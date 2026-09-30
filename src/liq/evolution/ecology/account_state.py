from dataclasses import dataclass, field
from types import MappingProxyType

from liq.evolution.ecology.driver import WalkReport
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.ports import ExecutionSimulator, RiskSizer
from liq.evolution.ecology.types import (
    AccountState,
    AgentId,
    Bar,
    BarWindow,
    CostScenarioId,
    Fill,
    InstrumentId,
    NotFilled,
    PositionTarget,
    Rejection,
    UtcTimestamp,
    _UtcTimestampMixin,
)

from .accounts import (
    FITNESS,
    FLAT,
    LEARNING,
    NOTHING_CHARGED,
    DecisionPointNotPriced,
    IntentFormer,
    _what_execution_did,
    _what_the_mandate_decided,
    what_the_reading_earned,
)


@dataclass(frozen=True)
class AccountEvent(_UtcTimestampMixin):
    """
    One decision taken in one bar, and the instant of what it consumed.

    Construction refuses an event that consumed the bar it was emitted in, or anything after it. That is not defensiveness: it is the single
    defect this record exists to make visible, and an implementation with it would not crash — every update would read a number that already
    existed because something in the same bar had just produced it, and the run would look healthy while being told the outcome of the decision
    it was about to take. A value that cannot be written down is a defect that cannot be recorded as normal.

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
    """
    The account as it stands part-way through a pass, and how it advances.

    Mutable and private. Everything it hands out is an immutable value; what is held here is the running state a walk advances one decision
    point at a time, which is exactly the state the port contract says is handed in and handed back rather than kept by a provider.
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
        """
        Advance the account through one decision point, in order.

        The order is the accounting: the previous outcome is delivered before anything acts, what was held is carried into the open, the target
        formed a bar ago is acted on and charged there, what is then held is carried to the close, the account is published, and only then is the
        next target formed.
        """
        bar = self._priced(window)
        self._deliver_the_previous_outcome(window.as_of, bar)
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

    def _deliver_the_previous_outcome(self, as_of: UtcTimestamp, bar: Bar) -> None:
        """
        Show the agent what its last reading earned, and record what was consumed.

        Both update paths consume the decision point before this one, because a decision taken inside a bar may not be told the outcome of the
        decision it is about to take. The two are recorded apart because they are two update paths, and a run can get one right while the other
        reaches forward.

        The learning line is no longer only a record. The reading the agent formed at the previous decision point is settled here, against the move
        the instrument made over the bar it was acted on; the agent is shown that outcome stamped at this instant, which is when it became knowable,
        and it is paired with the reading by the agent's own refusal rather than by this loop's care. At the first decision point of a pass there is
        no reading behind it and nothing is shown.
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
        if self.previous_close is None:
            return
        self.agent.observe(
            as_of, what_the_reading_earned(self.previous_close, bar.close)
        )

    def _carry(self, from_price: float | None, to_price: float) -> None:
        """
        Move the account on what it holds, between two prices.

        Both legs of a bar are this one step: the gap from the last close to this open, and the move from this open to this close. A flat account
        does not move, and the first decision point of a pass has no price behind it to be carried from.
        """
        if from_price is None or self.held == FLAT:
            return
        self.equity *= 1.0 + self.held * (to_price / from_price - 1.0)

    def _act_on_what_was_decided_a_bar_ago(self, bar: Bar, as_of: UtcTimestamp) -> None:
        """
        Act on the target formed at the previous decision point, if any.

        What the account then holds, what it is worth and what it has been charged are read back from the state the execution model hands out, never
        from the target that was sent: the difference between the two is exactly the case where a target was not reached, and reading the target
        would book a position nobody held.

        The line written at this boundary is stamped at ``as_of`` — the decision point the account is being advanced through — and not at the acting
        bar's own start, so that every line of one decision point sorts together. Which bar the target was acted on against is the account's own
        record to keep, and the fill keeps it.
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
        """
        Ask the agent what it wants, and the mandate what it may have.

        The wish is offered against the account as this decision point published it, which is what the agent has; the target it produces is held for
        the bar that follows, which is what keeps the acting bar a bar behind the forming one. A mandate that permits nothing leaves the account
        holding what it holds — a refusal is not an instruction to go flat.
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
