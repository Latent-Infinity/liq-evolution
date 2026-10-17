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

**The book is kept beside the accounts, when an allocation is passed.** Once
per decision point, after turnover and before the population decides, the
allocation is handed exactly the living, which is the instant its weights are
updated at. After the decision it is handed what every living agent's mandate
permitted it — the target formed now, or, where a wish was refused, what it was
last permitted, flat before anything was — and the aggregate book's own account,
and it returns the book's target. That target is acted on at the next bar, like
any agent's, through the same execution model and the same account class, under
the book's reserved identity; it is never acted on at the bar it was formed at.
The book reads no agent's account, and no agent's account reads the book. With
no allocation passed there is no book, and the walk is the one above.

**The fixed-share allocation.** :class:`FixedShareAllocation` is the hook the
steady-state loop allocates through. At the update instant it reads what the
accounts recorded at this decision point and nothing else — each living
agent's own outcome and the instrument's reference return, as
:meth:`PopulationAccounts.recorded` holds them — and updates the one
aggregator over exactly the living. At the book instant it hands the book's
ledger those weights and what the living were permitted, and returns the
target the ledger sent. It reads no wish, fill, holding, fitness figure,
descriptor or record entry, and holds plain values only.

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

import numpy as np

from liq.evolution.ecology.book_ledger import BookLedger, BookRow
from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import WalkReport, walk
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.fixed_share import FixedShare, FixedShareState
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.ports import (
    BarSource,
    ExecutionSimulator,
    PortfolioRisk,
    RiskSizer,
)
from liq.evolution.ecology.step import DecisionStep, what_the_reading_earned
from liq.evolution.ecology.types import (
    BOOK_ACCOUNT_ID,
    AccountState,
    AgentId,
    Bar,
    BarWindow,
    InstrumentId,
    Intent,
    PositionTarget,
    UtcTimestamp,
)

from .account_state import Evaluation, _EvaluationAccount
from .accounts import FLAT

__all__ = [
    "AllocationHook",
    "AllocationRecord",
    "FixedShareAllocation",
    "PopulationAccounts",
    "PopulationEvaluation",
    "Recorded",
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


class AllocationHook(Protocol):
    """Where the aggregate book is allocated and formed, at a decision point.

    Two instants, each once per decision point. :meth:`allocate` is called
    after settlement and the trigger's turnover, and before the population
    decides, with exactly the living: the instant the allocation's weights are
    updated at. :meth:`book` is called after the decision, with what each
    living agent was permitted, and returns the book's target, which the next
    bar acts on.
    """

    def allocate(self, window: BarWindow, living: tuple[AgentId, ...]) -> None:
        """Update the allocation at ``window``, over exactly ``living``."""
        ...

    def book(
        self,
        window: BarWindow,
        permitted: Mapping[AgentId, float],
        account: AccountState,
    ) -> PositionTarget | None:
        """Return the book's target formed at ``window``, or ``None`` to send none.

        Args:
            window: The decision point.
            permitted: Every living agent to the exposure its mandate permits
                it, as a fraction of its own equity, which the book reads as a
                fraction of the capital it allocates the agent.
            account: The book's own account, as published at ``window``.
        """
        ...


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
        book: The aggregate book's account over the pass, or ``None`` where
            no allocation was passed and there is no book.
    """

    accounts: Mapping[AgentId, Evaluation]
    opened_at: Mapping[AgentId, UtcTimestamp | None]
    closed_at: Mapping[AgentId, UtcTimestamp]
    history: WalkReport
    book: Evaluation | None = None


@dataclass(frozen=True)
class Recorded:
    """What the accounts recorded at one decision point, before anyone decides at it.

    Attributes:
        as_of: The decision instant (UTC), or ``None`` before the first.
        outcomes: Each agent that recorded an outcome there, to that outcome:
            the change in its account's equity over the bar just accounted.
            Founders at a pass's first decision point and agents born there
            have none.
        reference: The instrument's return over the decision point, or
            ``None`` at a pass's first, which has no previous close.
    """

    as_of: UtcTimestamp | None
    outcomes: Mapping[AgentId, float]
    reference: float | None


_NOTHING_RECORDED = Recorded(as_of=None, outcomes=MappingProxyType({}), reference=None)


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
        self._permitted: Mapping[AgentId, float] = MappingProxyType({})
        self._book: _EvaluationAccount | None = None
        self._history: WalkReport | None = None
        self._recorded: Recorded = _NOTHING_RECORDED
        for agent_id in population.agent_ids():
            self._accounts[agent_id] = self._account(agent_id)
            self._opened_at[agent_id] = None

    def recorded(self) -> Recorded:
        """What the latest decision point recorded: outcomes and the reference return."""
        return self._recorded

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
        allocation: AllocationHook | None = None,
    ) -> PopulationEvaluation:
        """Walk ``source`` once, every account advanced at every decision point."""
        self._history = walk(
            source,
            config,
            visit=lambda window: self.visit(
                window, turnover=turnover, allocation=allocation
            ),
            log=self.log,
        )
        return PopulationEvaluation(
            accounts=MappingProxyType(
                {agent_id: self.evaluation(agent_id) for agent_id in self._accounts}
            ),
            opened_at=MappingProxyType(dict(self._opened_at)),
            closed_at=MappingProxyType(dict(self._closed_at)),
            history=self._history,
            book=None if self._book is None else self._evaluated(self._book),
        )

    def evaluation(self, agent_id: AgentId) -> Evaluation:
        """One agent's account as it stands, alive or dead."""
        return self._evaluated(self._accounts[agent_id])

    def _evaluated(self, account: _EvaluationAccount) -> Evaluation:
        history = self._history or WalkReport(
            run_id="", visits=(), crossings=(), window_carries_across_segments=True
        )
        return account.evaluation(history, opening_equity=self.opening_equity)

    def visit(
        self,
        window: BarWindow,
        *,
        turnover: Callable[[BarWindow], None] = _no_turnover,
        allocation: AllocationHook | None = None,
    ) -> None:
        """Advance every account through one decision point, in the shared order.

        The aggregate book's account opens at the first decision point an
        allocation is passed at, at the founders' notional.
        """
        if allocation is not None and self._book is None:
            self._book = self._account(BOOK_ACCOUNT_ID)
        living = self.population.agent_ids()
        bar = self._accounts[living[0]]._priced(window)
        for agent_id in living:
            self._accounts[agent_id]._record_what_the_updates_consumed(window.as_of)
        self.step.take(
            window,
            bar,
            settle=self._settle,
            meanwhile=lambda: self._account_then_turn_over(
                bar, window, turnover, allocation
            ),
            decide=lambda: self._decide(window, allocation),
        )

    def _settle(self, as_of: UtcTimestamp, outcome: float) -> None:
        """Show every living agent what its last reading earned, at once."""
        self.population.observe(
            as_of, dict.fromkeys(self.population.agent_ids(), outcome)
        )

    def _account_then_turn_over(
        self,
        bar: Bar,
        window: BarWindow,
        turnover: Callable[[BarWindow], None],
        allocation: AllocationHook | None,
    ) -> None:
        """Account every open account for the bar, record outcomes, turn over, allocate."""
        living = self.population.agent_ids()
        for agent_id in (*living, *sorted(self._closing)):
            self._accounts[agent_id]._account_for_the_bar(bar, window.as_of)
        if self._book is not None:
            self._book._account_for_the_bar(bar, window.as_of)
        for agent_id in sorted(self._closing):
            self._closed_at[agent_id] = window.as_of
        self._closing.clear()
        outcomes: dict[AgentId, float] = {}
        for agent_id in living:
            account = self._accounts[agent_id]
            self.population.hold(agent_id, account.held)
            if len(account.states) < 2:
                # A founder at the first decision point: nothing came before it.
                continue
            earlier, later = account.states[-2:]
            outcomes[agent_id] = later.equity / earlier.equity - 1.0
            self.population.record_outcome(agent_id, outcomes[agent_id])
        reference = None
        if self.step.previous_close is not None:
            # The reference every agent's beta is estimated against: the move
            # the instrument made over the interval this decision point's
            # outcomes span, on the prices the accounts were just marked on.
            # Once per decision point, before turnover, and none at a pass's
            # first, which has no previous close and records no outcome.
            reference = what_the_reading_earned(self.step.previous_close, bar.close)
            self.population.record_reference_return(reference)
        self._recorded = Recorded(
            as_of=window.as_of,
            outcomes=MappingProxyType(outcomes),
            reference=reference,
        )
        turnover(window)
        if allocation is not None:
            allocation.allocate(window, self.population.agent_ids())

    def _decide(self, window: BarWindow, allocation: AllocationHook | None) -> None:
        """Form every living agent's wish at once, each account's target, then the book's."""
        step = self.population.step(window, self.instrument)
        self._wishes = MappingProxyType(
            dict(zip(step.agent_ids, step.intents(), strict=True))
        )
        permitted: dict[AgentId, float] = {}
        for agent_id in step.agent_ids:
            account = self._accounts[agent_id]
            account._form_the_target_the_next_bar_acts_on(window, account.states[-1])
            # A refusal holds, so what was last permitted stands; flat before
            # anything was.
            permitted[agent_id] = (
                self._permitted.get(agent_id, FLAT)
                if account.pending is None
                else account.pending.target_exposure
            )
        self._permitted = MappingProxyType(permitted)
        if allocation is not None and self._book is not None:
            self._form_the_book(window, allocation, self._book)

    def _form_the_book(
        self, window: BarWindow, allocation: AllocationHook, book: _EvaluationAccount
    ) -> None:
        """Hold the book's target for the next bar, refusing one that is not the book's now.

        Raises:
            ValueError: If the target is not keyed by the book's identity, is in
                another instrument, or was not formed at this decision point.
        """
        target = allocation.book(window, self._permitted, book.states[-1])
        if target is not None and (
            target.agent_id != BOOK_ACCOUNT_ID
            or target.instrument != self.instrument
            or target.as_of != window.as_of
        ):
            raise ValueError(
                f"the book is sent its own target in {self.instrument!r}, formed "
                f"at {window.as_of.isoformat()}; got {target.agent_id!r}'s in "
                f"{target.instrument!r}, formed at {target.as_of.isoformat()}"
            )
        book.pending = target

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


@dataclass(frozen=True)
class AllocationRecord:
    """What a fixed-share allocation did over a pass, one entry per decision point.

    Attributes:
        weights: The weights the book consumed at each decision point, over
            exactly the living after that point's turnover.
        rows: The book's ledger row at each decision point.
        recorded: What each update read: the outcomes and reference return
            the accounts recorded at that decision point.
    """

    weights: tuple[FixedShareState, ...]
    rows: tuple[BookRow, ...]
    recorded: tuple[Recorded, ...]


class FixedShareAllocation:
    """The allocation hook: the one aggregator's weights, and the book's ledger.

    Args:
        accounts: The accounts the hook is handed to, read only through
            :meth:`PopulationAccounts.recorded` and their configured
            instrument, notional and execution model's cost provenance.
        aggregator: The aggregator, at the frozen record's values.
        risk: The portfolio-risk port the netted book is handed to.
        band: The book's rebalance band, as configured.

    The weights start uniform over the founders, and the ledger opens flat at
    the accounts' notional, charging what the execution model states it
    charges. Both are frozen values, replaced at each decision point.
    """

    def __init__(
        self,
        accounts: PopulationAccounts,
        *,
        aggregator: FixedShare,
        risk: PortfolioRisk,
        band: float,
    ) -> None:
        self._accounts = accounts
        self._aggregator = aggregator
        self._risk = risk
        self.weights: FixedShareState = aggregator.start(
            accounts.population.agent_ids()
        )
        self.ledger: BookLedger = BookLedger.open(
            instrument=accounts.instrument,
            band=band,
            cost=accounts.simulator.cost_provenance(),
            opening_equity=accounts.opening_equity,
        )
        self._weights: list[FixedShareState] = []
        self._rows: list[BookRow] = []
        self._read: list[Recorded] = []

    def allocate(self, window: BarWindow, living: tuple[AgentId, ...]) -> None:
        """Update the weights over exactly ``living``, from what this point recorded.

        Raises:
            ValueError: If what the accounts recorded is not this decision
                point's.
        """
        recorded = self._accounts.recorded()
        if recorded.as_of != window.as_of:
            raise ValueError(
                f"the update at {window.as_of.isoformat()} reads what the accounts "
                "recorded there, but they last recorded at "
                f"{'nothing' if recorded.as_of is None else recorded.as_of.isoformat()}"
            )
        outcomes = recorded.outcomes
        ids = np.asarray(living, dtype=np.str_)
        has = np.isin(ids, np.asarray(tuple(outcomes), dtype=np.str_))
        # An agent with no outcome is asleep; its slot is never read.
        returns = np.full(ids.size, np.nan)
        returns[has] = [outcomes[agent_id] for agent_id in ids[has].tolist()]
        self.weights = self._aggregator.update(
            self.weights,
            living=living,
            returns=returns,
            recorded=has,
            reference=recorded.reference,
        )
        self._weights.append(self.weights)
        self._read.append(recorded)
        self._accounts.log.record(
            "decision_point_shown", bar_ts=window.as_of, stage="allocate"
        )

    def book(
        self,
        window: BarWindow,
        permitted: Mapping[AgentId, float],
        account: AccountState,
    ) -> PositionTarget | None:
        """Net, bound and band the book from this point's weights.

        Raises:
            ValueError: If ``permitted`` is not keyed by exactly the agents the
                weights were updated over.
        """
        weights = self.weights
        if set(permitted) != set(weights.agent_ids):
            raise ValueError(
                "the book is netted over exactly the agents weighted at "
                f"{window.as_of.isoformat()}; permitted names "
                f"{sorted(set(permitted) ^ set(weights.agent_ids))} besides"
            )
        self.ledger, row = self.ledger.decide(
            bar=window.bars[self._accounts.instrument],
            account=account,
            agent_ids=weights.agent_ids,
            weights=weights.weights,
            permitted=np.fromiter(
                (permitted[agent_id] for agent_id in weights.agent_ids),
                dtype=np.float64,
                count=len(weights.agent_ids),
            ),
            risk=self._risk,
        )
        self._rows.append(row)
        self._accounts.log.record(
            "intent_sized",
            bar_ts=window.as_of,
            stage="allocate",
            agent_id=BOOK_ACCOUNT_ID,
        )
        return row.decision.target

    def record(self) -> AllocationRecord:
        """Everything the allocation did so far, one entry per decision point."""
        return AllocationRecord(
            weights=tuple(self._weights),
            rows=tuple(self._rows),
            recorded=tuple(self._read),
        )
