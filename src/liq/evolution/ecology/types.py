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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, get_args

__all__ = [
    "AccountState",
    "AgentId",
    "ArchiveEntry",
    "Bar",
    "BarWindow",
    "CostProvenance",
    "CostScenarioId",
    "DeflatedFitness",
    "Descriptor",
    "Fill",
    "FitnessSample",
    "Genome",
    "InstrumentId",
    "Intent",
    "LivingOccupancy",
    "NotFilled",
    "NullDeclaration",
    "Offspring",
    "PositionTarget",
    "RecordCoverage",
    "RecordPayload",
    "Rejection",
    "SegmentRole",
    "SizingOutcome",
    "Standing",
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


# Where a deflator places one agent against the population it was compared
# with, at the deflator's declared confidence: told apart below it, not told
# apart at all, or told apart above it. Only "worse" licenses a cull.
type Standing = Literal["worse", "indistinguishable", "better"]


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
class Offspring:
    """One birth: the child, and the variation that produced it.

    A bare genome cannot say what was done to make it, so neither the scale of
    variation in force nor a boundary rule acting on a gene could be recorded
    with the birth — and a birth whose variation cannot be read back cannot be
    checked against the rule that was declared for it. The child and its
    variation therefore travel together.

    Attributes:
        genome: The child's heritable part, in its parents' vocabulary.
        mutation_scale: The mutation scale in force when this child was drawn,
            or ``None`` where no adaptive scale is in force at all — variation
            that introduces no magnitude has none to report. Where present it
            is a positive finite number.
        boundary_firings: Gene name to how many times the boundary rule acted
            on that gene during this birth. Every gene the child carries is
            named, zero where the rule did not act, so an absent count can
            never be mistaken for a rule that did not fire.
    """

    genome: Genome
    mutation_scale: float | None
    boundary_firings: Mapping[str, int]

    def __post_init__(self) -> None:
        """Refuse a report that does not describe the child it travels with."""
        scale = self.mutation_scale
        if scale is not None and not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                "a mutation scale in force must be positive and finite; a birth "
                "with no adaptive scale reports None rather than a number"
            )
        if set(self.boundary_firings) != set(self.genome.genes):
            raise ValueError(
                "boundary firings must name exactly the genes the child carries, "
                "zero where the rule did not act"
            )
        if any(count < 0 for count in self.boundary_firings.values()):
            raise ValueError("a boundary firing count cannot be negative")


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
class RecordCoverage:
    """How much of behaviour space the diversity record holds an entry in.

    Read from the record itself, never from a list kept beside it. It is
    cumulative over a run: a record never empties a cell, so the figure counts
    the cells ever reached by an entry it kept, and it is never evidence of how
    spread out the living population is.

    Attributes:
        axes: The descriptor names the record bins on, in its order.
        filled: How many cells hold at least one entry.
        total: How many cells the record has.
        histograms: Per axis, in :attr:`axes` order, how many held entries sit
            at each of that axis's cells.

    Raises:
        ValueError: If there is not one histogram per axis, if a count is
            negative, or if the cells filled are not between none and all.
    """

    axes: tuple[str, ...]
    filled: int
    total: int
    histograms: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.histograms) != len(self.axes):
            raise ValueError(
                f"{len(self.histograms)} histograms for {len(self.axes)} axes; "
                "every axis has exactly one"
            )
        if any(count < 0 for counts in self.histograms for count in counts):
            raise ValueError("a cell cannot hold a negative number of entries")
        if not 0 <= self.filled <= self.total:
            raise ValueError(
                f"{self.filled} of {self.total} cells filled; a record fills "
                "between none of its cells and all of them"
            )


@dataclass(frozen=True)
class LivingOccupancy:
    """How many of the record's cells the living agents sit in, now.

    Read from the living population under the record's own binning, never from
    the record: unlike :class:`RecordCoverage` it falls when the living crowd
    together, so it is the figure that can show a collapse.

    Attributes:
        axes: The descriptor names the record bins on, in its order.
        occupied: How many distinct cells hold at least one living agent.
        total: How many cells the record has.
        described: Living agents placed in a cell.
        undescribed: Living agents the population cannot yet describe.
        excluded: Living agents whose coordinate the record cannot bin; they
            are in no cell, never clamped into one.

    Raises:
        ValueError: If a count is negative, if more cells are occupied than
            there are or than agents were placed, or if agents were placed
            and no cell is occupied.
    """

    axes: tuple[str, ...]
    occupied: int
    total: int
    described: int
    undescribed: int
    excluded: int

    def __post_init__(self) -> None:
        if min(self.occupied, self.described, self.undescribed, self.excluded) < 0:
            raise ValueError("a count of cells or agents cannot be negative")
        if self.occupied > min(self.total, self.described):
            raise ValueError(
                f"{self.occupied} cells occupied by {self.described} placed agents "
                f"out of {self.total}; no more cells than either"
            )
        if self.described and not self.occupied:
            raise ValueError(
                f"{self.described} agents placed and no cell occupied; a placed "
                "agent occupies its cell"
            )


@dataclass(frozen=True)
class RecordPayload:
    """A diversity record as it is persisted, inside the envelope that says whose it is.

    The envelope names everything a restore must agree with before it loads a
    single entry: the version of this payload's layout, the run that wrote it,
    the descriptor vocabulary, and the record's axes, cells per axis and cell
    capacity. Each entry is held as text, so the payload refers to no live
    object and restoring it is a decoding, not a copy.

    Attributes:
        payload_version: Version of the payload's layout.
        run_id: The run that wrote it, and the only run it restores into.
        descriptor_schema_version: The vocabulary every entry is described in.
        axes: The descriptor names the record bins on, in its order.
        bins: Cells per axis, in :attr:`axes` order.
        bin_capacity: How many entries one cell holds.
        entries: Each held entry, encoded, in the order the record holds them.
    """

    payload_version: str
    run_id: str
    descriptor_schema_version: str
    axes: tuple[str, ...]
    bins: tuple[int, ...]
    bin_capacity: int
    entries: tuple[str, ...]


@dataclass(frozen=True)
class FitnessSample:
    """One agent's own evidence, handed in to be deflated against its rivals'.

    Attributes:
        agent_id: The agent the evidence belongs to.
        outcomes: The window of per-bar outcomes of the agent's own evaluation
            account, in bar order. Read as handed in; nothing is filled.
        independent_outcomes: How many of those outcomes are independent of one
            another — fewer than the window where outcomes overlap — and so how
            much evidence the window really carries.
    """

    agent_id: AgentId
    outcomes: tuple[float, ...]
    independent_outcomes: int


@dataclass(frozen=True)
class DeflatedFitness:
    """One agent's figure, deflated for the comparison it was part of.

    The undeflated figure travels beside the deflated one so the size of the
    deflation is read rather than reasoned about, and the standing is the
    deflator's verdict rather than something a reader infers by comparing
    figures — inferring it would be computing a statistic of one's own.

    A figure that is ``None`` means the agent was not assessed. It is never a
    number in waiting: nothing is substituted for it, and a reader orders on
    ``deflated`` only where it is present
    (``plans/oracle-v3-unassessed-fitness-2026-10-01.md``).

    Attributes:
        agent_id: The agent the figure belongs to.
        undeflated: The agent's per-period figure, before any deflation, or
            ``None`` where there is no per-period figure for the window.
        deflated: The figure the population is ordered by, deflated for how
            many agents were compared and how much independent evidence this
            one has, or ``None`` where the deflator could not assess the agent.
        compared: How many agents the figure was deflated against, this one
            included.
        independent_outcomes: The independent-outcome count the figure rests
            on, as the sample stated it.
        standing: Whether the figure is told apart from the population's at the
            deflator's declared confidence, and in which direction. An agent
            that was not assessed is ``"indistinguishable"``.
    """

    agent_id: AgentId
    undeflated: float | None
    deflated: float | None
    compared: int
    independent_outcomes: int
    standing: Standing

    @property
    def assessed(self) -> bool:
        """Whether the deflator returned a figure for this agent."""
        return self.deflated is not None

    def __post_init__(self) -> None:
        """Refuse a verdict no reader could act on without guessing.

        Raises:
            ValueError: If the standing is not one of the three a deflator may
                give, if a figure is present and not finite, if an agent with no
                deflated figure is told apart in either direction, or if a
                deflated figure is present without the figure it deflates.
        """
        if self.standing not in get_args(Standing.__value__):
            raise ValueError(
                "a standing must be one of worse, indistinguishable or better; "
                f"got {self.standing!r}"
            )
        for name, figure in (
            ("undeflated", self.undeflated),
            ("deflated", self.deflated),
        ):
            if figure is not None and not math.isfinite(figure):
                raise ValueError(
                    f"a present {name} figure must be finite, got {figure!r}: "
                    "an agent that was not assessed carries None, never a "
                    "number standing in for one"
                )
        if self.deflated is None and self.standing != "indistinguishable":
            raise ValueError(
                "an agent the deflator could not assess must be "
                f"indistinguishable, got {self.standing!r}: an agent nobody "
                "assessed is told apart from nobody"
            )
        if self.undeflated is None and self.deflated is not None:
            raise ValueError(
                "a deflated figure needs an undeflated one: nothing is deflated "
                "from a figure that does not exist"
            )


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
