"""One conforming, self-consistent stand-in per port.

These are test doubles, not providers. Each satisfies its port in full: every
method returns a well-formed domain value and nothing here raises on a normal
call. They exist so that the contract suite every adapter is run against has
something to run before a provider-backed adapter exists, and so every later
adapter has a reference behaviour to be compared against.

Nothing here is market data and nothing here may be read as market data. The
bars :class:`NullBarSource` yields are a declared arithmetic ramp at a constant
half-range and a constant volume — they are not any instrument's history, they
are not a fixture, and no number computed from them is evidence of anything. A
real bar source reads real history through the platform's data layer; this one
exists only so a suite can check the shape of the contract without one.

For the same reason :class:`NullExecutionSimulator` charges nothing by default.
A cost belongs to the cost scenario its adapter was configured with, and a
stand-in must not put a plausible-looking number where a resolved cost belongs.
The one bound it applies to what it will reach is a declared placeholder of the
same kind, and exists so that the outcome in which nothing trades is producible
here rather than only behind a provider.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import ClassVar

from liq.evolution.ecology.types import (
    AccountState,
    AgentId,
    ArchiveEntry,
    Bar,
    BarWindow,
    CostProvenance,
    CostScenarioId,
    Descriptor,
    Fill,
    Genome,
    InstrumentId,
    Intent,
    NotFilled,
    NullDeclaration,
    PositionTarget,
    Rejection,
    SegmentRole,
    SizingOutcome,
    SurrogateSeries,
    UtcTimestamp,
)

__all__ = [
    "NullAgentPopulation",
    "NullBarSource",
    "NullExecutionSimulator",
    "NullRiskSizer",
    "NullSurrogateSource",
]

# Declared placeholders, not prices: a unit ramp with a constant half-range and
# a constant volume, chosen to be obviously non-representative. See the module
# docstring.
_FIRST_CLOSE = 100.0
_CLOSE_STEP = 1.0
_HALF_RANGE = 1.0
_VOLUME = 1_000.0

# The walk-forward roles a run visits, in the order it visits them.
_SEGMENT_ROLES: tuple[SegmentRole, ...] = ("train", "validate", "test")

# An instant to count decision points from. It anchors the ramp above and says
# nothing about when anything happened.
_ORIGIN = datetime(2000, 1, 1, tzinfo=UTC)

# Unit arithmetic, not costs: a target is reached by moving to it once, so one
# call charges one of a round trip's two sides, and a rate in basis points is a
# fraction of notional once divided by this.
_SIDES_PER_ROUND_TRIP = 2
_BASIS_POINTS_PER_UNIT = 10_000.0


@dataclass(frozen=True)
class NullBarSource:
    """Walk a fixed number of decision points forward over a declared ramp.

    Every window carries one bar, one feature view computed from it, an as-of
    at the instant the bar completed, and the walk-forward segment it falls in.
    Segments are visited in order and never revisited, and the stream is a
    generator, so history cannot be rewound or replayed.

    Attributes:
        instrument: The single instrument every window carries a bar for.
        window_count: How many decision points the stream yields.
        segment_length: How many decision points share one segment before the
            next role begins.
        feature_schema_version: Version stamped on every window's feature view.
        segment_id_prefix: Prefix the generated segment identities share.
        origin: Instant the first bar's period starts at (UTC).
        bar_duration: Length of each bar's period.
    """

    instrument: InstrumentId = "NULL"
    window_count: int = 9
    segment_length: int = 3
    feature_schema_version: str = "null-features-1"
    segment_id_prefix: str = "null-segment"
    origin: UtcTimestamp = _ORIGIN
    bar_duration: timedelta = timedelta(minutes=1)

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point in chronological order, exactly once."""
        for index in range(self.window_count):
            period_start = self.origin + index * self.bar_duration
            period_end = period_start + self.bar_duration
            close = _FIRST_CLOSE + index * _CLOSE_STEP
            bar = Bar(
                instrument=self.instrument,
                period_start=period_start,
                period_end=period_end,
                open=close,
                high=close + _HALF_RANGE,
                low=close - _HALF_RANGE,
                close=close,
                volume=_VOLUME,
            )
            segment = min(
                index // self.segment_length,
                len(_SEGMENT_ROLES) - 1,
            )
            yield BarWindow(
                as_of=period_end,
                segment_id=f"{self.segment_id_prefix}-{segment}",
                segment_role=_SEGMENT_ROLES[segment],
                bars=MappingProxyType({self.instrument: bar}),
                features=MappingProxyType(
                    {
                        self.instrument: MappingProxyType(
                            {"level": close, "step": float(index)}
                        )
                    }
                ),
                feature_schema_version=self.feature_schema_version,
            )


@dataclass(frozen=True)
class NullRiskSizer:
    """Hold one declared bound of each kind, and apply each as its kind requires.

    What is applied here stands in for a mandate rather than being one: it
    permits long exposure only, and it caps gross exposure across the account —
    the wanted exposure plus everything already held elsewhere — at a declared
    fraction of equity. Neither bound is anyone's limit; like the ramp above
    they are declared placeholders, chosen so that a bound of each kind can be
    exercised.

    A short wish breaches the mandate in kind. No smaller short would have been
    allowed either, so it is refused whole: nothing may be traded and the
    reason is named. A long wish that does not fit under the cap breaches it in
    size, so it is reduced to what the cap leaves and the bound is recorded
    beside the reduced target, because that reduced exposure is the one that
    would be traded and scored. A wish that fits passes through unaltered, with
    nothing recorded.

    Attributes:
        max_gross_exposure: The cap on gross exposure across the account, as a
            fraction of evaluation-account equity.
    """

    #: A wish this mandate does not permit in any size.
    SHORT_NOT_PERMITTED: ClassVar[str] = "short_exposure_not_permitted"
    #: A wish reduced to the exposure the cap leaves.
    BOUND_EXCEEDED: ClassVar[str] = "gross_exposure_bound_exceeded"

    max_gross_exposure: float = 1.0

    def size(self, intent: Intent, account: AccountState) -> SizingOutcome:
        """Decide what ``intent`` may hold against ``account``, and what bound it."""
        if intent.target_exposure < 0.0:
            return SizingOutcome(
                target=None,
                rejections=(
                    self._record(
                        intent,
                        self.SHORT_NOT_PERMITTED,
                        f"short wish {intent.target_exposure} under a long-only mandate",
                    ),
                ),
            )
        held_elsewhere = sum(
            abs(exposure)
            for instrument, exposure in account.exposures.items()
            if instrument != intent.instrument
        )
        headroom = max(self.max_gross_exposure - held_elsewhere, 0.0)
        if intent.target_exposure > headroom:
            return SizingOutcome(
                target=self._permit(intent, headroom),
                rejections=(
                    self._record(
                        intent,
                        self.BOUND_EXCEEDED,
                        f"wanted {intent.target_exposure} with {headroom} left "
                        f"under cap {self.max_gross_exposure}",
                    ),
                ),
            )
        return SizingOutcome(
            target=self._permit(intent, intent.target_exposure),
            rejections=(),
        )

    @staticmethod
    def _permit(intent: Intent, exposure: float) -> PositionTarget:
        """Return the target letting ``intent``'s agent hold ``exposure``."""
        return PositionTarget(
            agent_id=intent.agent_id,
            instrument=intent.instrument,
            target_exposure=exposure,
            as_of=intent.as_of,
        )

    @staticmethod
    def _record(intent: Intent, reason: str, detail: str) -> Rejection:
        """Return the record of one bound binding on ``intent``."""
        return Rejection(
            agent_id=intent.agent_id,
            instrument=intent.instrument,
            as_of=intent.as_of,
            reason=reason,
            detail=detail,
        )


@dataclass(frozen=True)
class NullExecutionSimulator:
    """Reach a permitted target in full at the close of the bar that acts on it.

    Account state is handed in and handed back; nothing is kept between calls,
    so the same target acted on against the same bar and the same state always
    produces the same outcome.

    Both outcomes an execution model can produce are producible here, because a
    stand-in that could only ever fill would let the half of the contract about
    *not* filling pass without ever being run. Neither of the two declared
    bounds below is anyone's limit; like the ramp above they are declared
    placeholders, chosen so that each outcome can be exercised:

    * more exposure than :attr:`reachable_exposure` is out of reach, which
      stands in for the position bound a venue applies to an order it will not
      accept. Nothing trades and the reason is named.
    * a target already held needs no trade, so nothing trades and that is what
      is said. A fill here means something traded, always.

    Attributes:
        cost_scenario_id: The scenario every charge is drawn from. Resolved
            outside the ecology and carried into every outcome.
        effective_round_trip_bps: What a round trip is charged in basis points
            of traded notional, one side of it per call. Zero by default: a
            stand-in must not put a number where a resolved cost belongs, and a
            real adapter draws its charges from the named scenario instead.
        reachable_exposure: The largest exposure, in absolute value, this
            stand-in will reach.
    """

    #: A target larger than this stand-in says it can reach.
    TARGET_NOT_REACHABLE: ClassVar[str] = "target_exposure_not_reachable"
    #: A target that is already held, so there is nothing to trade.
    NOTHING_TO_TRADE: ClassVar[str] = "target_already_held"
    #: This stand-in resolves no scenario, so it applies no hedge leg either.
    HEDGE_LEG: ClassVar[str] = "not_in_scenario"

    cost_scenario_id: CostScenarioId = "null-no-charge"
    effective_round_trip_bps: float = 0.0
    reachable_exposure: float = 1.0

    def cost_provenance(self) -> CostProvenance:
        """Return what the named scenario effectively charges, as applied here."""
        return CostProvenance(
            cost_scenario_id=self.cost_scenario_id,
            effective_round_trip_bps=self.effective_round_trip_bps,
            hedge_leg=self.HEDGE_LEG,
        )

    def execute(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
    ) -> tuple[Fill | NotFilled, AccountState]:
        """Act on ``target`` at ``bar`` and return the outcome and the new state."""
        held = account.exposures.get(target.instrument, 0.0)
        if abs(target.target_exposure) > self.reachable_exposure:
            return self._nothing_traded(
                target,
                bar,
                account,
                held,
                self.TARGET_NOT_REACHABLE,
                f"wanted {target.target_exposure} beyond reach {self.reachable_exposure}",
            )
        moved = abs(target.target_exposure - held)
        if moved == 0.0:
            return self._nothing_traded(
                target, bar, account, held, self.NOTHING_TO_TRADE, ""
            )
        cost = (
            moved
            * account.equity
            * self.effective_round_trip_bps
            / _SIDES_PER_ROUND_TRIP
            / _BASIS_POINTS_PER_UNIT
        )
        fill = Fill(
            agent_id=target.agent_id,
            instrument=target.instrument,
            as_of=bar.period_end,
            requested_exposure=target.target_exposure,
            filled_exposure=target.target_exposure,
            price=bar.close,
            cost=cost,
            cost_scenario_id=self.cost_scenario_id,
        )
        exposures = dict(account.exposures)
        exposures[target.instrument] = fill.filled_exposure
        new_state = AccountState(
            agent_id=account.agent_id,
            as_of=fill.as_of,
            equity=account.equity - cost,
            exposures=MappingProxyType(exposures),
            costs_charged=account.costs_charged + cost,
        )
        return fill, new_state

    def _nothing_traded(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
        held: float,
        reason: str,
        detail: str,
    ) -> tuple[NotFilled, AccountState]:
        """Report that nothing traded, leaving the account exactly as handed in."""
        outcome = NotFilled(
            agent_id=target.agent_id,
            instrument=target.instrument,
            as_of=bar.period_end,
            requested_exposure=target.target_exposure,
            held_exposure=held,
            reason=reason,
            cost_scenario_id=self.cost_scenario_id,
            detail=detail,
        )
        unchanged = AccountState(
            agent_id=account.agent_id,
            as_of=outcome.as_of,
            equity=account.equity,
            exposures=MappingProxyType(dict(account.exposures)),
            costs_charged=account.costs_charged,
        )
        return outcome, unchanged


@dataclass
class NullAgentPopulation:
    """Hold a small fixed population and a keep-the-better diversity record.

    Each agent's heritable genes and its learned state are held apart, so a
    birth can carry one without the other. An offspring is the mean of its
    parents' genes with a seeded perturbation, which makes spawning
    reproducible without making it trivial. The record keeps one entry per
    coarse descriptor cell and replaces an incumbent only when the entry
    offered scores higher.

    Attributes:
        agent_count: How many agents are alive.
        gene_schema_version: Vocabulary the gene names belong to.
        descriptor_schema_version: Vocabulary the descriptor names belong to.
        variation: Half-width of the seeded perturbation applied on spawn.
    """

    agent_count: int = 3
    gene_schema_version: str = "null-genes-1"
    descriptor_schema_version: str = "null-descriptors-1"
    variation: float = 0.05

    _genomes: dict[AgentId, Genome] = field(init=False, repr=False)
    _learned: dict[AgentId, Mapping[str, float]] = field(init=False, repr=False)
    _descriptors: dict[AgentId, Descriptor] = field(init=False, repr=False)
    _archive: dict[tuple[float, ...], ArchiveEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Populate the living agents and start with an empty record."""
        span = max(self.agent_count - 1, 1)
        ids = tuple(f"null-agent-{index}" for index in range(self.agent_count))
        self._genomes = {
            agent_id: Genome(
                genes=MappingProxyType(
                    {"scale": index / span, "offset": 1.0 - index / span}
                ),
                schema_version=self.gene_schema_version,
            )
            for index, agent_id in enumerate(ids)
        }
        self._learned = {
            agent_id: MappingProxyType({"observations": float(index)})
            for index, agent_id in enumerate(ids)
        }
        self._descriptors = {
            agent_id: Descriptor(
                values=MappingProxyType({"activity": index / span}),
                schema_version=self.descriptor_schema_version,
            )
            for index, agent_id in enumerate(ids)
        }
        self._archive = {}

    def agent_ids(self) -> tuple[AgentId, ...]:
        """Return the identities of the living agents."""
        return tuple(self._genomes)

    def genome(self, agent_id: AgentId) -> Genome:
        """Return the heritable part of one agent."""
        return self._genomes[agent_id]

    def learned_state(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return what one agent has learned, separately from its genome."""
        return self._learned[agent_id]

    def descriptor(self, agent_id: AgentId) -> Descriptor:
        """Return where one agent currently sits in behaviour space."""
        return self._descriptors[agent_id]

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Genome:
        """Return one offspring genome of ``parents``, with variation applied."""
        lineage = [self._genomes[agent_id] for agent_id in parents]
        rng = random.Random(seed)
        genes = {
            name: sum(genome.genes[name] for genome in lineage) / len(lineage)
            + rng.uniform(-self.variation, self.variation)
            for name in sorted(lineage[0].genes)
        }
        return Genome(
            genes=MappingProxyType(genes),
            schema_version=lineage[0].schema_version,
        )

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the diversity record; report whether it was kept."""
        cell = self._cell(entry.descriptor)
        incumbent = self._archive.get(cell)
        if incumbent is not None and self._score(incumbent) >= self._score(entry):
            return False
        self._archive[cell] = entry
        return True

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the diversity record currently holds."""
        return tuple(self._archive.values())

    @staticmethod
    def _cell(descriptor: Descriptor) -> tuple[float, ...]:
        """Return the coarse behaviour cell ``descriptor`` falls in."""
        return tuple(round(value, 1) for _, value in sorted(descriptor.values.items()))

    @staticmethod
    def _score(entry: ArchiveEntry) -> float:
        """Return the single quality figure the record compares entries on."""
        return sum(entry.objectives.values())


@dataclass(frozen=True)
class NullSurrogateSource:
    """Draw a reordering of the observations it is handed.

    The construction keeps every observed value and its multiplicity and breaks
    the order they arrived in, so a draw tests whether a result survives when
    the sequence carries no information beyond its marginal values. Nothing is
    invented: a draw is a permutation of the real series handed in.

    Attributes:
        null_id: Identity this construction is declared and recorded under.
    """

    null_id: str = "null-order-permutation"

    def declaration(self) -> NullDeclaration:
        """Return what this null preserves, destroys and therefore tests."""
        return NullDeclaration(
            null_id=self.null_id,
            preserves=("observed_values", "series_length"),
            destroys=("temporal_ordering", "serial_dependence"),
            hypothesis=(
                "A result exceeding this distribution is not accounted for by "
                "the marginal values of the series alone, and therefore rests "
                "on the order they arrived in."
            ),
        )

    def draw(
        self,
        observed: Sequence[float],
        *,
        source_id: str,
        replicate: int,
        seed: int,
    ) -> SurrogateSeries:
        """Return one reproducible surrogate of ``observed`` under this null."""
        reordered = list(observed)
        random.Random(f"{seed}:{replicate}").shuffle(reordered)
        return SurrogateSeries(
            null_id=self.null_id,
            source_id=source_id,
            replicate=replicate,
            seed=seed,
            values=tuple(reordered),
        )
