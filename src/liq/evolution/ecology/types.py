"""Immutable value types that cross the ecology's provider boundaries.

Every type here is a *domain* value: it says what the ecology needs, not what a
providing library happens to return. Adapters translate library values into
these at the boundary, so no library type ever reaches domain or use-case code.

All timestamps are timezone-aware and expressed in UTC. A naive timestamp is not
a valid value for any field annotated :data:`UtcTimestamp`; conversion to a local
zone belongs at a display boundary, never in transit.

Every value is frozen. Values are shared by many readers within one decision
point, so a mutable value would let one reader change what another sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

__all__ = [
    "AccountState",
    "AgentId",
    "ArchiveEntry",
    "Bar",
    "BarWindow",
    "CostProvenance",
    "CostScenarioId",
    "Descriptor",
    "Fill",
    "Genome",
    "InstrumentId",
    "Intent",
    "NotFilled",
    "NullDeclaration",
    "PositionTarget",
    "Rejection",
    "SegmentRole",
    "SizingOutcome",
    "SurrogateSeries",
    "UtcTimestamp",
]

# A timezone-aware timestamp in UTC. Naive timestamps are not valid values.
type UtcTimestamp = datetime


class _UtcTimestampMixin:
    def _normalize_timestamps(self, *names: str) -> None:
        for name in names:
            value = getattr(self, name)
            if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))


# Stable identity of one agent for the life of a run, unique across rebirths.
type AgentId = str

# Stable identity of one tradable instrument, as the run's universe names it.
type InstrumentId = str

# Name of a cost scenario in the platform's cost book. Costs are referred to by
# this name and resolved outside the ecology; a cost is never a literal here.
type CostScenarioId = str

# What a walk-forward segment is used for. A run visits segments in order and
# never revisits one.
type SegmentRole = Literal["train", "validate", "test"]


@dataclass(frozen=True)
class Bar(_UtcTimestampMixin):
    """One completed bar of real market history.

    A value of this type exists only for a bar that was complete at
    ``period_end``. An incomplete bar is excluded by the source that produced
    it rather than represented here with missing or filled fields.

    Attributes:
        instrument: The instrument the bar describes.
        period_start: Inclusive start of the bar's period (UTC).
        period_end: Exclusive end of the bar's period (UTC); the instant at
            which the bar became complete and readable.
        open: First traded price in the period.
        high: Highest traded price in the period.
        low: Lowest traded price in the period.
        close: Last traded price in the period.
        volume: Traded volume over the period.
    """

    instrument: InstrumentId
    period_start: UtcTimestamp
    period_end: UtcTimestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        self._normalize_timestamps("period_start", "period_end")


@dataclass(frozen=True)
class BarWindow(_UtcTimestampMixin):
    """Everything the population may see at one decision point.

    A window is the unit of the forward pass: bars that have completed, the
    feature values computed from them once and shared by every reader, and the
    walk-forward segment the decision point falls in. Lookback lives inside the
    feature values, not in a list of past bars, so a reader cannot recompute
    history and cannot reach past what the window states.

    Attributes:
        as_of: The decision instant (UTC). Nothing in the window depends on
            information that became available after this instant.
        segment_id: Identity of the walk-forward segment this decision point
            belongs to. A split boundary is observable as a change of value.
        segment_role: What that segment is used for.
        bars: Instrument to the bar that completed at ``as_of``. Instruments
            with no complete bar at this instant are absent, never filled.
        features: Instrument to feature name to value, computed once for this
            decision point. Read-only: a reader that mutates it is a defect,
            not a private copy.
        feature_schema_version: Version of the feature vocabulary the names in
            ``features`` belong to, so an agent is never silently evaluated
            under a vocabulary it was not born under.
    """

    as_of: UtcTimestamp
    segment_id: str
    segment_role: SegmentRole
    bars: Mapping[InstrumentId, Bar]
    features: Mapping[InstrumentId, Mapping[str, float]]
    feature_schema_version: str

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class Intent(_UtcTimestampMixin):
    """What one agent wants to hold, before any mandate is applied.

    An intent is an exposure wish, not an order: it says how much of its own
    evaluation account the agent wants working in an instrument and when it
    decided. Order mechanics belong to whatever executes it.

    Attributes:
        agent_id: The agent that formed the intent.
        instrument: The instrument the exposure is wanted in.
        target_exposure: Wanted exposure as a signed fraction of the agent's
            evaluation-account equity. Negative values express a short wish,
            which the mandate may refuse.
        as_of: The decision instant the intent was formed at (UTC).
    """

    agent_id: AgentId
    instrument: InstrumentId
    target_exposure: float
    as_of: UtcTimestamp

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class PositionTarget(_UtcTimestampMixin):
    """An exposure an agent is permitted to hold.

    A target carries what the mandate allows, which is not always what the
    agent wished for. Where a bound on the *size* of a wish binds, the target
    carries the reduced exposure and the :class:`SizingOutcome` that delivers
    it names the bound that reduced it, so a smaller number is never a silent
    one. Where a mandate is breached in kind rather than in size, no target is
    produced at all.

    Attributes:
        agent_id: The agent the target belongs to.
        instrument: The instrument the exposure is in.
        target_exposure: Permitted exposure as a signed fraction of the agent's
            evaluation-account equity.
        as_of: The decision instant the target was formed at (UTC).
    """

    agent_id: AgentId
    instrument: InstrumentId
    target_exposure: float
    as_of: UtcTimestamp

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class Rejection(_UtcTimestampMixin):
    """A refusal to act, with the reason that produced it.

    A rejection is an outcome, not an error: it is returned, recorded and
    counted. Silence and silent adjustment are both defects.

    Attributes:
        agent_id: The agent whose intent was refused.
        instrument: The instrument the refused intent named.
        as_of: The decision instant of the refused intent (UTC).
        reason: Stable machine-readable code for why the intent was refused.
            Codes are compared and counted, so they do not carry free text.
        detail: Optional human-readable elaboration. Never carries a payload,
            a credential or anything not safe to log.
    """

    agent_id: AgentId
    instrument: InstrumentId
    as_of: UtcTimestamp
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class SizingOutcome:
    """What a mandate decided about one intent, and what bound it.

    A decision is not a choice between permitting and refusing, because bounds
    are not all of one kind. A bound on *direction* — a short wish where the
    mandate is long-only — is a breach of the mandate itself: there is no
    smaller version of that wish which would have been allowed, so nothing may
    be held and the refusal is the whole of the outcome. A bound on *magnitude*
    — too large, too levered — is the mandate working as intended: the exposure
    it leaves is what would be acted on and therefore what the agent is scored
    on, so the constrained target is carried and every bound that produced it
    is recorded beside it. Both parts are needed, which is why an outcome
    carries both rather than being one or the other.

    Three shapes are well formed. A target with no rejections: nothing bound,
    and the wish stands as the agent formed it. A target with rejections: the
    exposure was constrained, and each bound that constrained it is named. No
    target with rejections: nothing may be traded, and the reason is named. The
    fourth shape — neither a target nor a rejection — would withhold action
    without saying why, which is the silent adjustment this boundary exists to
    prevent, so construction refuses it.

    Attributes:
        target: The exposure that may be held, or ``None`` where nothing may be
            traded — either because the mandate was breached in kind, or
            because a bound on size left nothing tradable. ``None`` leaves
            whatever is already held untouched; it is not an instruction to go
            flat.
        rejections: Every bound that bound, each naming a stable reason code.
            Empty only where the intent passed through unaltered.
    """

    target: PositionTarget | None
    rejections: tuple[Rejection, ...]

    def __post_init__(self) -> None:
        """Refuse the one shape that would withhold action for no stated reason."""
        if self.target is None and not self.rejections:
            raise ValueError(
                "a sizing outcome that permits nothing must name at least one "
                "rejection: an intent refused for no recorded reason is the "
                "silent adjustment this value exists to make impossible"
            )


@dataclass(frozen=True)
class Fill(_UtcTimestampMixin):
    """What actually happened when a permitted target was acted on and traded.

    A fill exists only where something traded. ``requested_exposure`` and
    ``filled_exposure`` are both recorded so that a target which was not reached
    stays visible: scoring reads what was realised, never what was wanted.

    Attributes:
        agent_id: The agent the fill belongs to.
        instrument: The instrument traded.
        as_of: The instant the fill is accounted at (UTC).
        requested_exposure: Exposure the target asked for, as a signed fraction
            of evaluation-account equity.
        filled_exposure: Exposure actually reached, on the same scale. A value
            differing from ``requested_exposure`` records that the requested
            exposure was **not reached** — not that it was partially reached.
            Whether any difference between the two is even expressible depends
            on the execution model behind the port: under an adapter over a
            simulator that fills an order in full or not at all, the outcome is
            all-or-nothing, a partial quantity cannot arise, and a target that
            was not acted on is reported as :class:`NotFilled` rather than as a
            fill of nothing. No statistic that presupposes partiality — a fill
            ratio, a partial-fill rate — is in scope for such a model.
        price: Price the fill traded at.
        cost: Cost charged for the fill, in the account's units, drawn entirely
            from the named cost scenario.
        cost_scenario_id: The cost scenario every charge in this fill came
            from, carried so a result can be re-costed and reproduced.
    """

    agent_id: AgentId
    instrument: InstrumentId
    as_of: UtcTimestamp
    requested_exposure: float
    filled_exposure: float
    price: float
    cost: float
    cost_scenario_id: CostScenarioId

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class NotFilled(_UtcTimestampMixin):
    """A permitted target that was acted on and traded nothing, and why.

    This is the other half of what acting on a target can produce, and it exists
    because the alternative — returning a fill anyway — cannot be written down
    honestly. A fill that did not trade would carry a price nothing traded at
    and a charge for a trade nobody made, and it would report the request back
    as though it were the outcome, which is the single failure the realised-fill
    rule exists to catch.

    Two shapes are well formed and the difference between them is read from the
    numbers, not from the type. ``held_exposure`` equal to ``requested_exposure``
    says the target was already reached and nothing needed trading.
    ``held_exposure`` differing from it says the target was **not** reached, and
    ``reason`` says what stopped it.

    Three things are true of every value of this type, and two of them are true
    because of what the type does not have. There is no ``cost`` field, so
    nothing can be charged for a trade that did not happen. There is no
    ``price`` field, so no price can be presented as one something traded at.
    And ``reason`` is required to be a non-empty stable code, so withholding a
    trade in silence cannot be expressed at all.

    Attributes:
        agent_id: The agent the target belonged to.
        instrument: The instrument the target named.
        as_of: The instant the outcome is accounted at (UTC).
        requested_exposure: Exposure the target asked for, as a signed fraction
            of evaluation-account equity.
        held_exposure: Exposure still held in that instrument, unchanged by this
            call, on the same scale.
        reason: Stable machine-readable code for why nothing traded. Codes are
            compared and counted across runs, so they carry no free text.
        detail: Optional human-readable elaboration. Never carries a payload, a
            credential or anything not safe to log.
        cost_scenario_id: The cost scenario in force when the target was acted
            on. Nothing was charged under it; it is carried so that a run's
            record of what it did is uniform across both outcomes.
    """

    agent_id: AgentId
    instrument: InstrumentId
    as_of: UtcTimestamp
    requested_exposure: float
    held_exposure: float
    reason: str
    cost_scenario_id: CostScenarioId
    detail: str = ""

    def __post_init__(self) -> None:
        """Refuse the one shape that would withhold a trade for no stated reason."""
        self._normalize_timestamps("as_of")
        if not self.reason:
            raise ValueError(
                "an outcome in which nothing traded must name a stable reason: "
                "a target withheld for no recorded reason is the silent "
                "adjustment this value exists to make impossible"
            )


@dataclass(frozen=True)
class CostProvenance:
    """What the named cost scenario effectively charges, as the adapter applies it.

    A scenario is applied as written, including legs the book being modelled
    does not trade: silently dropping a parameter would make the charge a
    number the harness chose rather than one the scenario states, which is an
    inline cost decision wearing a scenario's name. The difference between what
    the scenario's headline says and what is actually charged therefore has to
    be readable rather than reasoned about, which is what this value is for.

    Attributes:
        cost_scenario_id: The scenario these figures were resolved from.
        effective_round_trip_bps: What a full round trip is actually charged, in
            basis points of traded notional, after every leg of the scenario is
            applied. One call charges one side of this.
        hedge_leg: Stable code for how the scenario's hedge leg was treated —
            charged although no hedge is traded, or absent from the scenario.
    """

    cost_scenario_id: CostScenarioId
    effective_round_trip_bps: float
    hedge_leg: str


@dataclass(frozen=True)
class AccountState(_UtcTimestampMixin):
    """One agent's evaluation account as of an instant.

    Each agent is scored on its own account at a normalised notional, so that
    agents are comparable to each other and separable from whatever the whole
    book holds.

    Attributes:
        agent_id: The agent the account belongs to.
        as_of: The instant the state describes (UTC).
        equity: Account value in normalised notional units.
        exposures: Instrument to realised signed exposure as a fraction of
            ``equity``. Absent instruments are flat.
        costs_charged: Costs charged to the account so far, in the same units
            as ``equity``.
    """

    agent_id: AgentId
    as_of: UtcTimestamp
    equity: float
    exposures: Mapping[InstrumentId, float]
    costs_charged: float

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class Genome:
    """The heritable part of an agent.

    Genes are named rather than positional so that a reader asks for the trait
    it means — the forgetting factor, say — instead of an index into a layout.
    The genome does not change during an agent's life; what the agent learns is
    held separately, because inheritance must be able to carry one without the
    other.

    Attributes:
        genes: Gene name to value.
        schema_version: Version of the gene vocabulary the names belong to.
    """

    genes: Mapping[str, float]
    schema_version: str


@dataclass(frozen=True)
class Descriptor:
    """Where an agent sits in behaviour space.

    Describes what an agent *does*, not how well it does it, so that diversity
    can be kept independently of quality.

    Attributes:
        values: Descriptor name to value, each normalised to ``[0, 1]``.
        schema_version: Version of the descriptor vocabulary the names belong
            to, recorded with every use so archives are never compared across
            incompatible vocabularies.
    """

    values: Mapping[str, float]
    schema_version: str


@dataclass(frozen=True)
class ArchiveEntry(_UtcTimestampMixin):
    """One agent's place in the diversity record.

    An entry is a proposal as much as a record: it is offered to the record,
    and the record decides whether to keep it.

    Attributes:
        agent_id: The agent the entry describes.
        genome: The heritable part of that agent at the time of the entry.
        descriptor: Where the agent sat in behaviour space.
        objectives: Objective name to value. Whether a value is better high or
            low is the platform's single objective vocabulary's business, not
            the entry's.
        recorded_at: When the entry was offered (UTC).
    """

    agent_id: AgentId
    genome: Genome
    descriptor: Descriptor
    objectives: Mapping[str, float]
    recorded_at: UtcTimestamp

    def __post_init__(self) -> None:
        self._normalize_timestamps("recorded_at")


@dataclass(frozen=True)
class NullDeclaration:
    """What a null keeps, what it breaks, and what that makes it a test of.

    A null that is not declared cannot be interpreted: the same number means
    different things depending on which structure survived the construction.
    Declaring it is therefore part of the null, not documentation about it.

    Attributes:
        null_id: Stable identity of this null construction.
        preserves: Properties of the real series the construction keeps.
        destroys: Properties of the real series the construction removes.
        hypothesis: The hypothesis a draw from this null tests — what it would
            mean for a real result to exceed the distribution it generates.
    """

    null_id: str
    preserves: tuple[str, ...]
    destroys: tuple[str, ...]
    hypothesis: str


@dataclass(frozen=True)
class SurrogateSeries:
    """One reproducible draw from a declared null.

    Attributes:
        null_id: The declaration this draw realises.
        source_id: Identity of the real series the draw was derived from.
        replicate: Index of this draw within the requested budget, from zero.
        seed: Seed that reproduces this exact draw.
        values: The drawn observations, positionally aligned with the source
            series.
    """

    null_id: str
    source_id: str
    replicate: int
    seed: int
    values: tuple[float, ...]
