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

:class:`NullFitnessDeflator` likewise deflates nothing and claims nothing: it
reports each agent's mean outcome as both figures and tells no agent apart, so a
loop wired to it can retire by age but never cull on a figure nobody deflated.
The deflation itself belongs to the platform's statistics authority, reached
only from the consumer that composes a run.

For the same reason :class:`NullExecutionSimulator` charges nothing by default.
A cost belongs to the cost scenario its adapter was configured with, and a
stand-in must not put a plausible-looking number where a resolved cost belongs.
The one bound it applies to what it will reach is a declared placeholder of the
same kind, and exists so that the outcome in which nothing trades is producible
here rather than only behind a provider.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import ClassVar

from liq.evolution.ecology.adapters import archive_codec
from liq.evolution.ecology.ports import (
    refuse_a_book_of_more_than_one_instrument,
    refuse_an_unanswerable_selection,
    refuse_an_undeflatable_sample,
)
from liq.evolution.ecology.types import (
    AUTHORIZED_BY,
    GENERATED_LABEL,
    AccountState,
    AgentId,
    ArchiveEntry,
    Bar,
    BarWindow,
    ConstrainedBook,
    CostProvenance,
    CostScenarioId,
    DeflatedFitness,
    Descriptor,
    Fill,
    FitnessSample,
    GeneratorFunction,
    Genome,
    InstrumentId,
    Intent,
    LivingOccupancy,
    NettedBook,
    NotFilled,
    NullDeclaration,
    Offspring,
    PositionTarget,
    RecordCoverage,
    RecordPayload,
    Rejection,
    SegmentRole,
    SizingOutcome,
    SurrogateProvenance,
    SurrogateSeries,
    UtcTimestamp,
)

# The in-repository permutation has no source series of its own: what it states
# about the series it was handed is declared once, here, and is the same for
# every draw. The epoch is a declared constant, not a period of any market.
_PERMUTATION_SYMBOL = "NULL"
_PERMUTATION_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_PERMUTATION_PURPOSE = "declared probe series; no booked read"

__all__ = [
    "NullAgentPopulation",
    "NullBarSource",
    "NullExecutionSimulator",
    "NullFitnessDeflator",
    "NullPortfolioRisk",
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

# The stand-ins' declared floors (plans/oracle-v3-unassessed-fitness-2026-10-01.md
# §3.2-§3.3). A uniform draw can choose among a single agent with a figure, and a
# mean asks nothing of a window beyond the two outcomes the deflation port
# already refuses below.
_FEWEST_TO_DRAW_FROM = 1
_PORT_WINDOW_FLOOR = 2

# The stand-in population's one descriptor, and its record's coarse cells: a
# value is placed at its nearest tenth, so the unit interval has eleven cells.
_ACTIVITY = "activity"
_TENTHS = 10
_CELLS_PER_AXIS = _TENTHS + 1


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
    reproducible without making it trivial; the perturbation's half-width is
    reported as the scale in force, and no boundary rule exists here to fire.
    Parents are drawn uniformly, with replacement, from the agents carrying a
    figure, or from the living when none does — a stand-in that prefers nobody,
    so no selection pressure is implied by it. One figure is enough to choose
    among, so its ``minimum_candidates`` is one. The record
    keeps one entry per coarse descriptor cell and replaces an incumbent only
    when the entry offered scores higher. A coordinate outside the unit
    interval, or not a number, is an exclusion, kept aside. The record persists
    and restores through the same codec and envelope as the platform-backed
    one, so the two are held to one persistence contract.

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
    _excluded: list[ArchiveEntry] = field(init=False, repr=False)

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
        self._excluded = []

    @property
    def descriptor_declaration(self) -> None:
        """``None``: the stand-in is described in its own vocabulary, not the declared axes."""
        return None

    @property
    def minimum_candidates(self) -> int:
        """One: a uniform draw can choose among a single agent with a figure."""
        return _FEWEST_TO_DRAW_FROM

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

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        """Return one offspring of ``parents``, with the variation it applied."""
        lineage = [self._genomes[agent_id] for agent_id in parents]
        rng = random.Random(seed)
        genes = {
            name: sum(genome.genes[name] for genome in lineage) / len(lineage)
            + rng.uniform(-self.variation, self.variation)
            for name in sorted(lineage[0].genes)
        }
        return Offspring(
            genome=Genome(
                genes=MappingProxyType(genes),
                schema_version=lineage[0].schema_version,
            ),
            mutation_scale=self.variation or None,
            boundary_firings=MappingProxyType(dict.fromkeys(genes, 0)),
        )

    def select_parents(
        self, *, fitness: Mapping[AgentId, float | None], count: int, seed: int
    ) -> tuple[AgentId, ...]:
        """Return ``count`` parents, drawn uniformly and reproducibly.

        Drawn from the agents carrying a figure, or from the living when none
        does; an agent handed in as ``None`` is otherwise never chosen.
        """
        living = self.agent_ids()
        refuse_an_unanswerable_selection(
            fitness=fitness,
            living=living,
            count=count,
            minimum=self.minimum_candidates,
        )
        candidates = [agent_id for agent_id in living if fitness[agent_id] is not None]
        return tuple(random.Random(seed).choices(candidates or living, k=count))

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the diversity record; report whether it was kept.

        A coordinate outside the unit interval, or not a number, is excluded.
        """
        if not all(0.0 <= value <= 1.0 for value in entry.descriptor.values.values()):
            self._excluded.append(entry)
            return False
        return self._keep(entry)

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the diversity record currently holds."""
        return tuple(self._archive.values())

    def excluded(self) -> tuple[ArchiveEntry, ...]:
        """Return every entry refused for a coordinate the record cannot bin."""
        return tuple(self._excluded)

    def occupancy(self) -> LivingOccupancy:
        """Return the coarse cells the living occupy, by the record's own cell rule.

        A coordinate outside the unit interval, or not a number, is in no cell.
        """
        cells = set()
        excluded = 0
        for agent_id in self.agent_ids():
            descriptor = self.descriptor(agent_id)
            if all(0.0 <= value <= 1.0 for value in descriptor.values.values()):
                cells.add(self._cell(descriptor))
            else:
                excluded += 1
        return LivingOccupancy(
            axes=(_ACTIVITY,),
            occupied=len(cells),
            total=_CELLS_PER_AXIS,
            described=len(self._descriptors) - excluded,
            undescribed=0,
            excluded=excluded,
        )

    def coverage(self) -> RecordCoverage:
        """Return the coarse cells the record fills, counted from its own cells."""
        histogram = [0] * _CELLS_PER_AXIS
        for (value,) in self._archive:
            histogram[round(value * _TENTHS)] += 1
        return RecordCoverage(
            axes=(_ACTIVITY,),
            filled=len(self._archive),
            total=_CELLS_PER_AXIS,
            histograms=(tuple(histogram),),
        )

    def persisted(self, run_id: str) -> RecordPayload:
        """Return the record's entries, encoded in its order, inside their envelope."""
        return RecordPayload(
            payload_version=archive_codec.PAYLOAD_VERSION,
            run_id=run_id,
            descriptor_schema_version=self.descriptor_schema_version,
            axes=(_ACTIVITY,),
            bins=(_CELLS_PER_AXIS,),
            bin_capacity=1,
            entries=tuple(
                archive_codec.encode_entry(entry) for entry in self._archive.values()
            ),
        )

    def restore(self, payload: RecordPayload, run_id: str) -> None:
        """Replay ``payload`` into the empty record, for restart of ``run_id``.

        Raises:
            ValueError: If the payload is not a record of this run in this
                record's shape, or if the record already holds an entry.
        """
        archive_codec.refuse_a_foreign_payload(
            payload,
            run_id=run_id,
            descriptor_schema_version=self.descriptor_schema_version,
            axes=(_ACTIVITY,),
            bins=(_CELLS_PER_AXIS,),
            bin_capacity=1,
            held=len(self._archive),
        )
        for text in payload.entries:
            self._keep(archive_codec.decode_entry(text))

    def _keep(self, entry: ArchiveEntry) -> bool:
        """Place ``entry`` in its cell unless the incumbent scores at least as high."""
        cell = self._cell(entry.descriptor)
        incumbent = self._archive.get(cell)
        if incumbent is not None and self._score(incumbent) >= self._score(entry):
            return False
        self._archive[cell] = entry
        return True

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
        declaration = self.declaration()
        canonical = "\n".join(repr(float(value)) for value in observed) + "\n"
        shuffle = inspect.getsource(random.Random.shuffle)
        provenance = SurrogateProvenance(
            source_data_id=source_id,
            symbols=(_PERMUTATION_SYMBOL,),
            period_first_bar=_PERMUTATION_EPOCH,
            period_last_bar=_PERMUTATION_EPOCH,
            source_data_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            source_dataset_id=self.null_id,
            source_read_purpose=_PERMUTATION_PURPOSE,
            source_read_arm_id=self.null_id,
            restart_indices=(0,),
            method="permutation",
            generator_version=importlib.metadata.version("liq-evolution"),
            generator_functions=(
                GeneratorFunction(
                    name="random.Random.shuffle",
                    sha256=hashlib.sha256(shuffle.encode("utf-8")).hexdigest(),
                ),
            ),
            generator_seed=seed,
            root_seed=seed,
            cell_key=hashlib.sha256(f"{self.null_id}|{source_id}".encode()).hexdigest()[
                :16
            ],
            null_replicates=replicate + 1,
            instrument="*",
            block_length=None,
            block_length_derivation=None,
            row_plan=None,
            max_iters=None,
            converged=None,
            iterations=None,
            spectral_distance=None,
            declaration=declaration,
            authorized_by=AUTHORIZED_BY,
            signal_bearing=False,
            citable=False,
            generated_label=GENERATED_LABEL,
        )
        return SurrogateSeries(
            null_id=self.null_id,
            source_id=source_id,
            replicate=replicate,
            seed=seed,
            values=tuple(reordered),
            provenance=provenance,
        )


@dataclass(frozen=True)
class NullFitnessDeflator:
    """Deflate nothing, tell nobody apart, and say so.

    Each agent's figure is the mean of its own outcomes, reported as both the
    undeflated and the deflated figure, and every standing is
    ``"indistinguishable"``: with no deflation there is no evidence to tell any
    agent apart with, in either direction. A mean exists for every window the
    port accepts, so every agent is assessed and ``minimum_window`` is the
    port's own floor of two. Used by this library's own tests only; a composed
    run is wired to the platform-backed deflator.
    """

    @property
    def minimum_window(self) -> int:
        """Two: the port's own floor, below which every deflator refuses."""
        return _PORT_WINDOW_FLOOR

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        """Return each sample's mean outcome, undeflated, and claim nothing."""
        refuse_an_undeflatable_sample(samples)
        return tuple(
            DeflatedFitness(
                agent_id=each.agent_id,
                undeflated=math.fsum(each.outcomes) / len(each.outcomes),
                deflated=math.fsum(each.outcomes) / len(each.outcomes),
                compared=len(samples),
                independent_outcomes=each.independent_outcomes,
                standing="indistinguishable",
            )
            for each in samples
        )


@dataclass(frozen=True)
class NullPortfolioRisk:
    """Apply no aggregate bound to the book, and say so.

    Every book comes back as it was netted, with nothing recorded, because
    nothing bound it; :meth:`bound` names nothing and :meth:`not_bound` says
    that no aggregate bound is applied at all, so a run wired to this stand-in
    cannot be read as one whose book was constrained. It still refuses a book
    of more than one instrument, as every adapter of the port does.
    """

    #: The one entry :meth:`not_bound` reports: every aggregate bound at once.
    NOTHING_APPLIED: ClassVar[tuple[str, str]] = (
        "aggregate_bounds",
        "a stand-in applies no aggregate bound: the book stands as netted",
    )

    def constrain(self, book: NettedBook) -> ConstrainedBook:
        """Return ``book``'s netted target as its target, with nothing recorded."""
        refuse_a_book_of_more_than_one_instrument(book)
        return ConstrainedBook(
            as_of=book.as_of,
            instrument=book.instrument,
            netted=book.netted,
            target=book.netted,
            rejections=(),
        )

    def bound(self) -> tuple[str, ...]:
        """Nothing: this stand-in applies no bound."""
        return ()

    def not_bound(self) -> tuple[tuple[str, str], ...]:
        """Every aggregate bound, at once, with the reason none is applied."""
        return (self.NOTHING_APPLIED,)
