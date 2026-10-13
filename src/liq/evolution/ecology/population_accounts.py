"""Evaluation accounts for a population that turns over while it walks.

Every living agent keeps its own evaluation account, at the same normalised
notional, opened when the agent is born and closed when it dies. What this
module adds to the account is only what a changing population needs: who has an
account, the order a decision point is taken in for all of them at once, and the
point inside that order where who is alive may change.

**One driver, not two.** This is the only thing that advances an evaluation
account through a decision point. One agent evaluated on its own in
:func:`~liq.evolution.ecology.accounts.evaluate` is a population of one driven
here, never a second loop over the same account: two drivers of one account
can drift into two orders, and only one of them would be the one checked.

**One order, not a second one.** A decision point is taken by the shared
:class:`~liq.evolution.ecology.step.DecisionStep`, once for the whole population:
settle what the last reading earned, account for the bar, then decide. Every
account is handed the same step, so each carries what it held over the gap from
the same previous close the population settled against, and each account's own
accounting — carry into the open, act on the target formed a bar ago, carry to
the close, publish — is the account class's, unchanged. Settling and deciding are
population-wide because the population's rule is: one observation shows every
living agent what its last reading earned, and one step forms every living
agent's next wish.

**Where turnover happens.** Between accounting for the bar and deciding at it. By
then every reading has been settled, so the population may admit and cull, and
every account has published the bar, so a turnover decision reads accounts as
they stand at this decision point and nothing later. An agent born there opens
its account at that instant and forms its first wish at the same decision point
as everyone else; an agent that dies there forms none.

**One exit.** A death holds a flat target for the next bar, through the account's
own :meth:`~liq.evolution.ecology.account_state._EvaluationAccount.close`. The
next bar acts on it like any target — at that bar's open, under the scenario the
execution model was handed, charged once — and the account is final after it.
This module is never told why an agent died: a cull and a retirement are closed
by the same call, so the kind of a death cannot become a branch in accounting.

**The reference goes to the population.** Once per decision point after a
pass's first, before turnover, the instrument's close-to-close return over that
decision point is appended to the population's reference history — the series
every agent's beta descriptor is estimated against. It is computed from the
prices the accounts were just marked on, never re-read from a bar source, and
it lines up with every agent's outcomes because each living agent records an
outcome at every decision point after its first.

**Outcomes go to their own agent.** Each account's per-bar outcome — the change in
its equity from the state it last published — is appended to that agent's own
outcome history in the population, and to nobody else's. An account opened at a
birth publishes its opening state at the birth instant, so an agent of any age
has exactly as many outcomes as bars it has lived.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import WalkReport, walk
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.ports import BarSource, ExecutionSimulator, RiskSizer
from liq.evolution.ecology.step import DecisionStep, what_the_reading_earned
from liq.evolution.ecology.types import (
    AgentId,
    Bar,
    BarWindow,
    InstrumentId,
    Intent,
    UtcTimestamp,
)

from .account_state import Evaluation, _EvaluationAccount

__all__ = [
    "PopulationAccounts",
    "PopulationEvaluation",
    "SettledPopulationWide",
]


class SettledPopulationWide(EcologyError):
    """One member of a population was asked to settle on its own.

    A population's readings are settled by one observation for every living
    agent; an agent shown its outcome alone would leave the rest a bar behind.
    """


class _Decided(Protocol):
    """What a population formed at one decision point, as an account reads it."""

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Who the wishes belong to, positionally."""
        ...

    def intents(self) -> tuple[Intent, ...]:
        """Every wish formed, aligned with :attr:`agent_ids`."""
        ...


class _Accounted(Protocol):
    """What the accounts need of the population whose wishes they run.

    The living agents, one settlement of all of them, one decision for all of
    them, and somewhere to write down what each account now holds and what each
    bar earned it. :class:`~liq.evolution.ecology.agent.PopulationState` is one;
    a single agent evaluated alone is another, of one member.
    """

    def agent_ids(self) -> tuple[AgentId, ...]:
        """The living agents, in the order they are accounted."""
        ...

    def observe(self, as_of: UtcTimestamp, outcomes: Mapping[AgentId, float]) -> object:
        """Show every living agent what its last reading earned."""
        ...

    def hold(self, agent_id: AgentId, exposure: float) -> None:
        """Record what one agent's account now holds."""
        ...

    def record_outcome(self, agent_id: AgentId, value: float) -> None:
        """Record what one bar earned one agent's account."""
        ...

    def record_reference_return(self, value: float) -> None:
        """Record the traded instrument's return over one decision point."""
        ...

    def step(self, window: BarWindow, instrument: InstrumentId) -> _Decided:
        """Form every living agent's wish at ``window``."""
        ...


def _no_turnover(window: BarWindow) -> None:
    """Change nobody at a decision point."""


@dataclass(frozen=True)
class PopulationEvaluation:
    """What every account did over one pass, alive or dead at its end.

    Attributes:
        accounts: Every account opened during the pass, by agent, in the order
            the accounts were opened: founders first, then each birth.
        opened_at: The instant each account opened at, or ``None`` for a
            founder, whose account opened before the walk began.
        closed_at: The instant each dead agent's closing bar was accounted at,
            after which its account took no further state. An agent dead at the
            pass's last decision point has no closing bar and is absent.
        history: What the pass covered, as the walk reported it.
    """

    accounts: Mapping[AgentId, Evaluation]
    opened_at: Mapping[AgentId, UtcTimestamp | None]
    closed_at: Mapping[AgentId, UtcTimestamp]
    history: WalkReport


@dataclass(frozen=True)
class _Member:
    """One living agent as its account sees it: its identity and its last wish.

    The wish is read from the step the population took at this decision point,
    so an account forms its target from what the population's rule formed for
    that agent and nothing is formed twice.
    """

    agent_id: AgentId
    accounts: PopulationAccounts

    def intend(self, window: BarWindow, instrument: InstrumentId) -> Intent:
        """Return the wish the population formed for this agent at this decision point."""
        del window, instrument
        return self.accounts.wish(self.agent_id)

    def observe(self, as_of: UtcTimestamp, outcome: float) -> object:
        """Refuse: a population is settled as one, never through one member."""
        del outcome
        raise SettledPopulationWide(
            f"{self.agent_id!r} was asked to settle alone at {as_of.isoformat()}; "
            "a population's readings are settled by one observation of every "
            "living agent"
        )


class PopulationAccounts:
    """One evaluation account per agent, opened at birth and closed at death.

    Args:
        population: Whose wishes are run. Its living agents at construction are
            the founders, and each gets an account at once.
        instrument: The single instrument every account follows.
        sizer: The mandate, applied to each account's wishes.
        simulator: The execution model every account acts and is charged
            through, under the scenario it was handed by name.
        opening_equity: What every account opens with, founders and births
            alike, in normalised notional units.
        log: Where each account's boundary crossings are written.

    Raises:
        ValueError: If accounts would open with nothing.
    """

    def __init__(
        self,
        population: _Accounted,
        *,
        instrument: InstrumentId,
        sizer: RiskSizer,
        simulator: ExecutionSimulator,
        opening_equity: float,
        log: RunLog = SILENT,
    ) -> None:
        if opening_equity <= 0.0:
            raise ValueError(
                f"an evaluation account must open with something: {opening_equity} "
                "leaves no notional for an exposure to be a fraction of and no "
                "figure for a score to be earned on"
            )
        self.population = population
        self.instrument = instrument
        self.sizer = sizer
        self.simulator = simulator
        self.opening_equity = opening_equity
        self.log = log
        self.step = DecisionStep()
        self._accounts: dict[AgentId, _EvaluationAccount] = {}
        self._opened_at: dict[AgentId, UtcTimestamp | None] = {}
        self._closing: set[AgentId] = set()
        self._closed_at: dict[AgentId, UtcTimestamp] = {}
        self._wishes: Mapping[AgentId, Intent] = MappingProxyType({})
        self._history: WalkReport | None = None
        for agent_id in population.agent_ids():
            self._accounts[agent_id] = self._account(agent_id)
            self._opened_at[agent_id] = None

    def member(self, agent_id: AgentId) -> _Member:
        """The member an account's wishes are read through."""
        return _Member(agent_id=agent_id, accounts=self)

    def wish(self, agent_id: AgentId) -> Intent:
        """The wish the population formed for ``agent_id`` at the current decision point."""
        return self._wishes[agent_id]

    def open(self, agent_id: AgentId, *, as_of: UtcTimestamp) -> None:
        """Open an account for an agent just admitted, at the founders' notional.

        Raises:
            KeyError: If the agent is not alive in the population.
            ValueError: If the agent already has an account.
        """
        if agent_id not in self.population.agent_ids():
            raise KeyError(
                f"{agent_id!r} is not alive; an account is opened for an agent the "
                "population has admitted"
            )
        if agent_id in self._accounts:
            raise ValueError(f"{agent_id!r} already has an account")
        account = self._account(agent_id)
        account.opened_at(as_of)
        self._accounts[agent_id] = account
        self._opened_at[agent_id] = as_of

    def close(self, agent_id: AgentId, *, as_of: UtcTimestamp) -> None:
        """Close a dead agent's account at the next bar, whatever it died of.

        Raises:
            KeyError: If the agent never had an account.
            ValueError: If the agent is still alive, or its account is already
                closing or closed.
        """
        if agent_id not in self._accounts:
            raise KeyError(f"{agent_id!r} has no account to close")
        if agent_id in self.population.agent_ids():
            raise ValueError(
                f"{agent_id!r} is still alive; an account is closed for an agent "
                "the population has removed"
            )
        if agent_id in self._closing or agent_id in self._closed_at:
            raise ValueError(f"{agent_id!r}'s account is already closed")
        self._accounts[agent_id].close(as_of)
        self._closing.add(agent_id)

    def run(
        self,
        source: BarSource,
        config: EcologyConfig,
        *,
        turnover: Callable[[BarWindow], None] = _no_turnover,
    ) -> PopulationEvaluation:
        """Walk ``source`` once, every account advanced at every decision point."""
        self._history = walk(
            source,
            config,
            visit=lambda window: self.visit(window, turnover=turnover),
            log=self.log,
        )
        return PopulationEvaluation(
            accounts=MappingProxyType(
                {agent_id: self.evaluation(agent_id) for agent_id in self._accounts}
            ),
            opened_at=MappingProxyType(dict(self._opened_at)),
            closed_at=MappingProxyType(dict(self._closed_at)),
            history=self._history,
        )

    def evaluation(self, agent_id: AgentId) -> Evaluation:
        """One agent's account as it stands, alive or dead."""
        history = self._history or WalkReport(
            run_id="", visits=(), crossings=(), window_carries_across_segments=True
        )
        return self._accounts[agent_id].evaluation(
            history, opening_equity=self.opening_equity
        )

    def visit(
        self,
        window: BarWindow,
        *,
        turnover: Callable[[BarWindow], None] = _no_turnover,
    ) -> None:
        """Advance every account through one decision point, in the shared order."""
        living = self.population.agent_ids()
        bar = self._accounts[living[0]]._priced(window)
        for agent_id in living:
            self._accounts[agent_id]._record_what_the_updates_consumed(window.as_of)
        self.step.take(
            window,
            bar,
            settle=self._settle,
            meanwhile=lambda: self._account_then_turn_over(bar, window, turnover),
            decide=lambda: self._decide(window),
        )

    def _settle(self, as_of: UtcTimestamp, outcome: float) -> None:
        """Show every living agent what its last reading earned, at once."""
        self.population.observe(
            as_of, dict.fromkeys(self.population.agent_ids(), outcome)
        )

    def _account_then_turn_over(
        self, bar: Bar, window: BarWindow, turnover: Callable[[BarWindow], None]
    ) -> None:
        """Account every open account for the bar, record outcomes, then turn over."""
        living = self.population.agent_ids()
        for agent_id in (*living, *sorted(self._closing)):
            self._accounts[agent_id]._account_for_the_bar(bar, window.as_of)
        for agent_id in sorted(self._closing):
            self._closed_at[agent_id] = window.as_of
        self._closing.clear()
        for agent_id in living:
            account = self._accounts[agent_id]
            self.population.hold(agent_id, account.held)
            if len(account.states) < 2:
                # A founder at the first decision point: nothing came before it.
                continue
            earlier, later = account.states[-2:]
            self.population.record_outcome(
                agent_id, later.equity / earlier.equity - 1.0
            )
        if self.step.previous_close is not None:
            # The reference every agent's beta is estimated against: the move
            # the instrument made over the interval this decision point's
            # outcomes span, on the prices the accounts were just marked on.
            # Once per decision point, before turnover, and none at a pass's
            # first, which has no previous close and records no outcome.
            self.population.record_reference_return(
                what_the_reading_earned(self.step.previous_close, bar.close)
            )
        turnover(window)

    def _decide(self, window: BarWindow) -> None:
        """Form every living agent's wish at once, and each account's target from it."""
        step = self.population.step(window, self.instrument)
        self._wishes = MappingProxyType(
            dict(zip(step.agent_ids, step.intents(), strict=True))
        )
        for agent_id in step.agent_ids:
            account = self._accounts[agent_id]
            account._form_the_target_the_next_bar_acts_on(window, account.states[-1])

    def _account(self, agent_id: AgentId) -> _EvaluationAccount:
        return _EvaluationAccount(
            agent=self.member(agent_id),
            instrument=self.instrument,
            sizer=self.sizer,
            simulator=self.simulator,
            equity=self.opening_equity,
            log=self.log,
            step=self.step,
        )
