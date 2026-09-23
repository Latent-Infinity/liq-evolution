"""One agent's evaluation account, joined through the mandate and the model.

This is the lower level of the two-level accounting model: one agent, its own
account at a normalised notional, scored on what it actually held. The aggregate
ledger — allocation weights, netting, the portfolio-level mandate — is a
different ledger and is not here. Scoring an agent on its own account is what
keeps its fitness from depending on the allocation its fitness is about to
determine.

**Normalised notional.** Every agent opens with the same equity, so two agents'
figures are comparable to each other and neither is a figure about a book. The
account is a fraction of itself: exposure is a signed fraction of equity, a
charge is a fraction of equity, and no currency amount appears anywhere.

**What this module decides is nothing.** A wish is formed by the agent, what may
be held is decided by the mandate behind :class:`~liq.evolution.ecology.ports.RiskSizer`,
what is reached is decided by the model behind
:class:`~liq.evolution.ecology.ports.ExecutionSimulator`, and what it costs is
decided by the cost scenario that model was handed by name. This module joins
them in order and writes down what happened. There is no sizing arithmetic here,
no fill mechanics here and no cost number here — not as a default, not as a
fallback, not as a rounding rule — because each of those is a decision somebody
else has already made and a second copy of it would be a second answer.

**The acting bar is not the forming bar.** A target formed at one decision point
is acted on at the *next* one. A target acted on inside the bar it was formed on
would fill at prices that printed before the decision, so the execution model
refuses it outright rather than absorbing it, and this loop is what keeps the
two a bar apart: the target a decision point produces is held and handed to the
bar that follows it.

**How a bar is accounted, in full.** Three steps, in the order they happen:

1. What was held over the gap is carried to this bar's **open**, which is where
   the execution model acts. A position held from the last close to this open
   earns that move and nothing else.
2. The target formed a bar ago is acted on there, and whatever it is charged is
   charged there, on the equity as it then stands.
3. What is held *after* that is carried to the bar's **close**.

A bar that traded nothing therefore steps close to close, which is the familiar
form. Only a bar that traded differs, and it differs in the direction that stops
a newly established position being credited with a move it was not held over.

**Realised, never wished.** The exposure the account goes on holding is read
back from the account state the execution model hands out, not from the target
that was sent. Where a target is not reached the account holds what it held, is
charged nothing, and is scored on that — which is the whole of the claim this
account exists to make good.

**Why the event trace is emitted here.** Each decision point records what its
learning and fitness updates consumed: the decision point before it, or nothing
at the first, where there is no prior outcome to be shown. The record lives with
the account rather than with the walk because the walk knows nothing about
agents, accounts or scores, and a loop that had to would stop being one loop.
What the trace makes checkable is an ordering that cannot be checked by reading
code: a fitness update told the outcome of the bar it is scoring has been handed
the answer, and so has everything downstream of it. V1 takes no cull, birth or
allocation decision, so nothing yet *acts* on the refreshed score; the ordering
is established before the decisions that will consume it, because a trace begun
only when its consumer arrived would be asserting about a first run nobody
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import WalkReport, walk
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.logctx import SILENT, Event, RunLog
from liq.evolution.ecology.ports import BarSource, ExecutionSimulator, RiskSizer
from liq.evolution.ecology.types import (
    AccountState,
    AgentId,
    Bar,
    BarWindow,
    CostScenarioId,
    Fill,
    InstrumentId,
    Intent,
    NotFilled,
    PositionTarget,
    Rejection,
    SizingOutcome,
    UtcTimestamp,
    _UtcTimestampMixin,
)

__all__ = [
    "FITNESS",
    "FLAT",
    "LEARNING",
    "NOTHING_CHARGED",
    "AccountEvent",
    "DecisionPointNotPriced",
    "Evaluation",
    "IntentFormer",
    "evaluate",
]

#: Kind of event: what one agent was shown of an outcome so that it could update
#: on it.
LEARNING = "learning"

#: Kind of event: what one agent's score was refreshed from.
FITNESS = "fitness"

#: What an account holds in an instrument it is not in.
FLAT = 0.0

#: What an account has been charged before anything has been acted on.
NOTHING_CHARGED = 0.0


def _what_the_mandate_decided(sized: SizingOutcome) -> Event:
    """Which of the three well-formed sizing shapes this outcome is.

    Three, not two. A wish permitted whole and a wish reduced to what the
    mandate allows are both targets and would read alike under one label, and
    the difference between them is the whole of what the mandate did.
    """
    if sized.target is None:
        return "intent_refused"
    return "intent_bound" if sized.rejections else "intent_sized"


def _what_execution_did(outcome: Fill | NotFilled) -> Event:
    """Whether acting on a target reached it.

    Read from the outcome's type rather than by comparing exposures, because
    that is where the distinction is made: an outcome that traded nothing is a
    different value from one that traded, and it cannot be mistaken for a fill
    of nothing.
    """
    return "target_not_reached" if isinstance(outcome, NotFilled) else "target_reached"


class IntentFormer(Protocol):
    """Whatever forms one agent's wish at a decision point.

    Named structurally rather than as the agent type, because what this account
    needs of whoever forms its wishes is an identity and a rule — and a stand-in
    that exists to want something the mandate refuses is exactly what the
    long-only claim has to be checked against. A nominal dependency here would
    make that check impossible to write without the ecology's own agent learning
    to misbehave.
    """

    agent_id: AgentId

    def intend(self, window: BarWindow, instrument: InstrumentId) -> Intent:
        """Return what this agent wants to hold in ``instrument`` at ``window``."""
        ...


class DecisionPointNotPriced(EcologyError):
    """The decision point carried no completed bar for the instrument followed.

    An account follows one instrument, and a decision point that prices it is
    what makes the account expressible there: without a bar there is no open to
    act at and no close to carry to. Marking the position at the last price it
    had would forward-fill a mark, and holding the account still would report an
    unchanged equity as though the market had not moved, so neither is done.

    It is raised rather than recorded, for the reason a refused target is: it
    says the account was pointed at an instrument this stream does not carry,
    which is a wiring defect and not a market outcome.
    """


@dataclass(frozen=True)
class AccountEvent(_UtcTimestampMixin):
    """One decision taken in one bar, and the instant of what it consumed.

    Construction refuses an event that consumed the bar it was emitted in, or
    anything after it. That is not defensiveness: it is the single defect this
    record exists to make visible, and an implementation with it would not
    crash — every update would read a number that already existed because
    something in the same bar had just produced it, and the run would look
    healthy while being told the outcome of the decision it was about to take.
    A value that cannot be written down is a defect that cannot be recorded as
    normal.

    Attributes:
        agent_id: The agent whose decision this was.
        as_of: The decision instant the event was emitted in (UTC).
        kind: Stable code for which decision it was — :data:`LEARNING` or
            :data:`FITNESS`.
        consumed_as_of: The decision instant of the information the update
            consumed (UTC), or ``None`` at the first decision point of a run,
            where there is no prior outcome to be shown or scored.
    """

    agent_id: AgentId
    as_of: UtcTimestamp
    kind: str
    consumed_as_of: UtcTimestamp | None

    def __post_init__(self) -> None:
        """Refuse an update that consumed its own bar, or one after it."""
        self._normalize_timestamps("as_of")
        if self.consumed_as_of is None:
            return
        self._normalize_timestamps("consumed_as_of")
        if self.consumed_as_of >= self.as_of:
            raise ValueError(
                f"a {self.kind!r} update at {self.as_of.isoformat()} consumed "
                f"{self.consumed_as_of.isoformat()}, which is not information "
                "that existed before the bar the update was taken in"
            )


@dataclass(frozen=True)
class Evaluation:
    """What one agent's account did over one pass, and what it scored.

    Attributes:
        agent_id: The agent the account belongs to.
        instrument: The single instrument the account follows.
        cost_scenario_id: The scenario every charge in this account was drawn
            from, carried so the result can be reproduced and re-costed. What
            that scenario effectively charges is the execution model's to state
            and is recorded where the model is built.
        opening_equity: What the account opened with, in normalised notional.
        states: The account as of each decision point, in order — one per
            decision point the pass covered.
        fills: What acting on each target produced, in order. A target that was
            not reached appears here as the outcome saying so, never as a fill.
        rejections: Every bound that bound, in order, across the whole pass.
        events: Every learning and fitness update the pass emitted, in order,
            each naming the instant of the information it consumed.
        history: What the pass covered and where it crossed a walk-forward
            boundary, as the walk reported it.
    """

    agent_id: AgentId
    instrument: InstrumentId
    cost_scenario_id: CostScenarioId
    opening_equity: float
    states: tuple[AccountState, ...]
    fills: tuple[Fill | NotFilled, ...]
    rejections: tuple[Rejection, ...]
    events: tuple[AccountEvent, ...]
    history: WalkReport

    @property
    def closing_equity(self) -> float:
        """What the account was worth after the last decision point."""
        if not self.states:
            return self.opening_equity
        return self.states[-1].equity

    @property
    def fitness(self) -> float:
        """What the account earned on what it opened with.

        The score over the whole pass, which consumes every decision point
        including the last. It is not the per-bar figure the fitness events
        name: those are what a decision *inside* a bar may consume, and they
        stop a bar's own outcome being fed to the decision that produced it.
        """
        return self.closing_equity / self.opening_equity - 1.0


@dataclass
class _EvaluationAccount:
    """The account as it stands part-way through a pass, and how it advances.

    Mutable and private. Everything it hands out is an immutable value; what is
    held here is the running state a walk advances one decision point at a time,
    which is exactly the state the port contract says is handed in and handed
    back rather than kept by a provider.
    """

    agent: IntentFormer
    instrument: InstrumentId
    sizer: RiskSizer
    simulator: ExecutionSimulator
    equity: float
    log: RunLog = SILENT
    held: float = FLAT
    charged: float = NOTHING_CHARGED
    previous_close: float | None = None
    previous_as_of: UtcTimestamp | None = None
    pending: PositionTarget | None = None
    states: list[AccountState] = field(default_factory=list)
    fills: list[Fill | NotFilled] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    events: list[AccountEvent] = field(default_factory=list)

    def visit(self, window: BarWindow) -> None:
        """Advance the account through one decision point, in order.

        The order is the accounting: the previous outcome is delivered before
        anything acts, what was held is carried into the open, the target formed
        a bar ago is acted on and charged there, what is then held is carried to
        the close, the account is published, and only then is the next target
        formed.
        """
        bar = self._priced(window)
        self._deliver_the_previous_outcome(window.as_of)
        self._carry(self.previous_close, bar.open)
        self._act_on_what_was_decided_a_bar_ago(bar, window.as_of)
        self._carry(bar.open, bar.close)
        published = self._publish(window.as_of)
        self._form_the_target_the_next_bar_acts_on(window, published)
        self.previous_close = bar.close
        self.previous_as_of = window.as_of

    def _priced(self, window: BarWindow) -> Bar:
        """The bar this decision point prices the followed instrument at."""
        bar = window.bars.get(self.instrument)
        if bar is None:
            raise DecisionPointNotPriced(
                f"the decision point at {window.as_of.isoformat()} carries no "
                f"completed bar for '{self.instrument}', which this account "
                f"follows; it prices {sorted(window.bars)}"
            )
        return bar

    def _deliver_the_previous_outcome(self, as_of: UtcTimestamp) -> None:
        """Record what this bar's learning and fitness updates consumed.

        Both consume the decision point before this one, because a decision
        taken inside a bar may not be told that bar's own outcome. The two are
        recorded apart because they are two update paths, and a run can get one
        right while the other reaches forward.
        """
        self.events.extend(
            AccountEvent(
                agent_id=self.agent.agent_id,
                as_of=as_of,
                kind=kind,
                consumed_as_of=self.previous_as_of,
            )
            for kind in (LEARNING, FITNESS)
        )

    def _carry(self, from_price: float | None, to_price: float) -> None:
        """Move the account on what it holds, between two prices.

        Both legs of a bar are this one step: the gap from the last close to
        this open, and the move from this open to this close. A flat account
        does not move, and the first decision point of a pass has no price
        behind it to be carried from.
        """
        if from_price is None or self.held == FLAT:
            return
        self.equity *= 1.0 + self.held * (to_price / from_price - 1.0)

    def _act_on_what_was_decided_a_bar_ago(self, bar: Bar, as_of: UtcTimestamp) -> None:
        """Act on the target formed at the previous decision point, if any.

        What the account then holds, what it is worth and what it has been
        charged are read back from the state the execution model hands out,
        never from the target that was sent: the difference between the two is
        exactly the case where a target was not reached, and reading the target
        would book a position nobody held.

        The line written at this boundary is stamped at ``as_of`` — the
        decision point the account is being advanced through — and not at the
        acting bar's own start, so that every line of one decision point sorts
        together. Which bar the target was acted on against is the account's
        own record to keep, and the fill keeps it.
        """
        if self.pending is None:
            return
        outcome, reached = self.simulator.execute(
            self.pending, bar, self._as_of(bar.period_start)
        )
        self.log.record(
            _what_execution_did(outcome),
            bar_ts=as_of,
            stage="fill",
            agent_id=self.agent.agent_id,
        )
        self.fills.append(outcome)
        self.equity = reached.equity
        self.held = reached.exposures.get(self.instrument, FLAT)
        self.charged = reached.costs_charged
        self.pending = None

    def _form_the_target_the_next_bar_acts_on(
        self, window: BarWindow, published: AccountState
    ) -> None:
        """Ask the agent what it wants, and the mandate what it may have.

        The wish is offered against the account as this decision point published
        it, which is what the agent has; the target it produces is held for the
        bar that follows, which is what keeps the acting bar a bar behind the
        forming one. A mandate that permits nothing leaves the account holding
        what it holds — a refusal is not an instruction to go flat.
        """
        sized = self.sizer.size(self.agent.intend(window, self.instrument), published)
        self.log.record(
            _what_the_mandate_decided(sized),
            bar_ts=window.as_of,
            stage="size",
            agent_id=self.agent.agent_id,
        )
        self.rejections.extend(sized.rejections)
        self.pending = sized.target

    def _publish(self, as_of: UtcTimestamp) -> AccountState:
        """Record and return the account as of one decision point."""
        published = self._as_of(as_of)
        self.states.append(published)
        return published

    def _as_of(self, as_of: UtcTimestamp) -> AccountState:
        """The account as it stands, stamped at one instant."""
        return AccountState(
            agent_id=self.agent.agent_id,
            as_of=as_of,
            equity=self.equity,
            exposures=MappingProxyType({self.instrument: self.held}),
            costs_charged=self.charged,
        )


def evaluate(
    source: BarSource,
    config: EcologyConfig,
    *,
    agent: IntentFormer,
    instrument: InstrumentId,
    sizer: RiskSizer,
    simulator: ExecutionSimulator,
    opening_equity: float,
    log: RunLog = SILENT,
) -> Evaluation:
    """Run one agent over ``source`` once and return its own evaluation account.

    The pass is driven by the walk rather than by a loop written here, so the
    forward-once discipline the walk enforces holds of this account for free: a
    decision point offered out of order, twice, or in a segment already left
    stops the run where it is refused rather than being scored.

    Args:
        source: Where decision points come from. Only the port is used.
        config: How the pass is driven, and the record of what it was allowed to
            be.
        agent: Whose wishes are run. An identity and a rule for forming them.
        instrument: The single instrument the account follows. Every decision
            point of the pass must price it.
        sizer: The mandate. What may be held, and every bound that bound.
        simulator: The execution model. What was reached, and what it charged
            under the scenario it was handed by name.
        opening_equity: What the account opens with, in normalised notional
            units. The same for every agent, so two agents' scores are
            comparable to each other.
        log: Where the two boundaries this account crosses write what they did:
            what the mandate decided about each wish, and whether acting on
            each permitted target reached it. Both lines name the agent, since
            both concern one member rather than the population. The log is
            handed to the walk as well, so one run's lines come from one log
            and not from two configured apart. Silent until a run wires a
            writer, and silent is the same run: nothing an account computes is
            read from, or conditioned on, what it wrote down.

    Returns:
        Evaluation: The account as of every decision point, what it traded, what
        was refused, what each bar's updates consumed, and what it scored.

    Raises:
        ValueError: If the account opens with nothing. A score is what an
            account earned on what it opened with, and an account that opened
            with nothing has no such figure.
        DecisionPointNotPriced: If a decision point carries no completed bar for
            ``instrument``.
    """
    if opening_equity <= 0.0:
        raise ValueError(
            f"an evaluation account must open with something: {opening_equity} "
            "leaves no notional for an exposure to be a fraction of and no "
            "figure for a score to be earned on"
        )
    account = _EvaluationAccount(
        agent=agent,
        instrument=instrument,
        sizer=sizer,
        simulator=simulator,
        equity=opening_equity,
        log=log,
    )
    history = walk(source, config, visit=account.visit, log=log)
    return Evaluation(
        agent_id=agent.agent_id,
        instrument=instrument,
        cost_scenario_id=simulator.cost_scenario_id,
        opening_equity=opening_equity,
        states=tuple(account.states),
        fills=tuple(account.fills),
        rejections=tuple(account.rejections),
        events=tuple(account.events),
        history=history,
    )
