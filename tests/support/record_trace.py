"""A steady-state run over a declared population, and a trace of what it offered its record.

The loop is walked over the declared streams of ``tests/support/turnover_stream.py``
with the real array-genome adapter described on the four declared axes. What the
loop offers is never read from its own report: the adapter's ``record`` and the
platform archive's ``insert`` are intercepted at the instance, by calling module;
the store's ``depart`` is intercepted the same way; and the C-06 double's every
call is logged with what the population could describe at that moment. A check
reads the trace against the deflate calls, never against what the loop says it
did.

**The reference for the default run.** :data:`OFFER_FREE_REPORT_DIGEST` is the sha256 of
``repr`` of the turnover report of :func:`reference_walk`, taken from the loop
before the record's offers existed (``steady_state.py`` sha256
``523fee9e…700b``). It is a frozen fixture: a default run that differs from it
in any byte is not the run that loop made.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace

from liq.evolution import ecology
from liq.evolution.ecology import agent, descriptors
from liq.evolution.ecology.adapters import (
    ArrayGenomePopulation,
    NullExecutionSimulator,
    NullRiskSizer,
)
from liq.evolution.ecology.config import EcologyConfig, LearningConfig, PopulationConfig
from liq.evolution.ecology.types import (
    AgentId,
    ArchiveEntry,
    BarWindow,
    DeflatedFitness,
    Descriptor,
    FitnessSample,
    Genome,
)
from tests.support import turnover_stream as streams

#: The loop's module: the only module that may offer.
LOOP = "liq.evolution.ecology.steady_state"

#: The adapter's module: the only module that may insert into the archive.
ADAPTER = "liq.evolution.ecology.adapters.array_genome"

#: How many decision points a walk covers by default.
DECISION_POINTS = 120

#: How many founders a walk starts from.
FOUNDERS = 14

#: The descriptor window: the run's fitness window.
WINDOW = 8

#: The bins of the holding horizon, turnover and beta axes, as the
#: descriptor-parameter decision record states them
#: (``liq-docs/plans/agent-ecology-adr-descriptor-parameters.md``).
DECIDED_BINS = (3, 16, 9)

#: The one declared asset class of the one traded instrument.
ASSET_CLASS = "equity"

#: The sha256 of ``repr`` of :func:`reference_walk`'s turnover report under the
#: loop that offered nothing. See the module docstring.
OFFER_FREE_REPORT_DIGEST = (
    "6938df6de10ad956e63e707df931a651cc20e796d57dc6e1d09a24e91a2ae7d2"
)

#: A configuration key a walk is not handed, so the configuration's own default
#: is the one in force.
UNSET = object()


#: The mandate's turnover bound the declared walks normalise turnover by.
MAX_TURNOVER = 1.0


def declaration(
    max_turnover: float = MAX_TURNOVER,
) -> descriptors.DescriptorDeclaration:
    """The four declared axes over the one traded instrument, at the walk's window.

    A ``max_turnover`` below the turnover the stream's agents show puts their
    turnover outside the unit interval, where it is excluded, never clamped.
    """
    return descriptors.DescriptorDeclaration(
        window=WINDOW,
        history_capacity=streams.HISTORY_CAPACITY,
        asset_classes=(ASSET_CLASS,),
        universe={streams.INSTRUMENT: ASSET_CLASS},
        mandate=descriptors.Mandate(
            max_position=1.0,
            max_gross_exposure=NullRiskSizer().max_gross_exposure,
            max_weight_delta=2.0,
            max_turnover=max_turnover,
        ),
    )


def declared(
    state: agent.PopulationState, max_turnover: float = MAX_TURNOVER
) -> ArrayGenomePopulation:
    """The real adapter over ``state``, described on the declared axes."""
    return ArrayGenomePopulation(
        state=state,
        descriptor_schema_version=descriptors.VOCABULARY,
        descriptors=declaration(max_turnover),
        descriptor_bins=DECIDED_BINS,
    )


def turnover(**keys: object):
    """A turnover configuration; every key a check does not name is declared here."""
    declared_keys: dict[str, object] = {
        "band_lower": 14,
        "band_upper": 20,
        "replacement_count": 2,
        "trigger_period": 3,
        "minimum_age": 6,
        "replacement_window": 6,
        "maximum_replaced_fraction": 0.3,
        "fitness_window": WINDOW,
        "parents_per_birth": 2,
        "inheritance": "baldwinian",
        "admission_redraws": 8,
        "seed": 11,
    }
    declared_keys.update(keys)
    return ecology.steady_state.TurnoverConfig(**declared_keys)


def deflated_for_comparisons(
    samples: Sequence[FitnessSample],
) -> list[tuple[float, str]]:
    """The double's declaration for the traced walks: each window's sum over the number compared.

    Worse where the window lost on balance. The figure differs from the
    undeflated one (the window's sum) wherever more than one agent is
    compared, so a trace can tell an offer of one from an offer of the other.
    """
    compared = len(samples)
    return [
        (total / compared, "worse" if total < 0.0 else "indistinguishable")
        for total in (math.fsum(sample.outcomes) for sample in samples)
    ]


def ecology_config(*, run_id: str, turnover: object, **keys: object) -> EcologyConfig:
    """The pass's configuration, handed only the keys a check names."""
    return EcologyConfig(
        run_id=run_id,
        population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
        learning=LearningConfig(),
        turnover=turnover,  # ty: ignore[invalid-argument-type]
        **{name: value for name, value in keys.items() if value is not UNSET},  # ty: ignore[invalid-argument-type]
    )


class RecordPort(streams.Recording):
    """The recording port, answering the record's reporting surfaces as the real one."""

    def coverage(self):
        return self.inner.coverage()  # ty: ignore[unresolved-attribute]

    def occupancy(self):
        return self.inner.occupancy()  # ty: ignore[unresolved-attribute]

    def excluded(self):
        return self.inner.excluded()  # ty: ignore[unresolved-attribute]


@dataclass(frozen=True)
class Expected:
    """What an assessed, described agent may be offered with at one trigger."""

    figure: float
    descriptor: Descriptor
    genome: Genome


@dataclass(frozen=True)
class Offer:
    """One call of the adapter's ``record``, as intercepted."""

    bar: int
    caller: str
    entry: ArchiveEntry
    kept: bool
    living_before: frozenset[AgentId]
    living_after: frozenset[AgentId]
    held_before: tuple[ArchiveEntry, ...]
    held_after: tuple[ArchiveEntry, ...]


@dataclass
class Trace:
    """Everything intercepted over one walk, in the order it happened.

    Attributes:
        bars: Per decision point walked, its instant and segment.
        events: ``(kind, bar)`` in order: ``deflate``, ``offer``, ``death``.
        expected: Per bar with a deflate call, each agent that call assessed
            and the population could describe there, with its figure.
        offers: Every ``record`` call.
        inserts: Every archive ``insert``: its bar, its caller, and whether an
            offer was in progress.
        deaths: Every departure: its bar, the agent, and the record's entries
            immediately before and after it.
        occupancy_census: Per ``occupancy()`` call, its bar, the living it was
            read over, and the cells the census counts there.
        unassessed_carrying: Per bar with a deflate call, each agent that call
            left unassessed but handed a non-``None`` undeflated figure and the
            population could describe there, with that figure.
        held_after_bar: Per decision point walked, the record's entries once
            the loop has finished with it: after its deaths and births, before
            any later offer.
    """

    bars: list[BarWindow] = field(default_factory=list)
    events: list[tuple[str, int]] = field(default_factory=list)
    expected: dict[int, dict[AgentId, Expected]] = field(default_factory=dict)
    offers: list[Offer] = field(default_factory=list)
    inserts: list[tuple[int, str, bool]] = field(default_factory=list)
    deaths: list[
        tuple[int, AgentId, tuple[ArchiveEntry, ...], tuple[ArchiveEntry, ...]]
    ] = field(default_factory=list)
    occupancy_census: list[tuple[int, frozenset[AgentId], int]] = field(
        default_factory=list
    )
    unassessed_carrying: dict[int, dict[AgentId, float]] = field(default_factory=dict)
    held_after_bar: list[tuple[ArchiveEntry, ...]] = field(default_factory=list)
    offering: bool = False

    @property
    def bar(self) -> int:
        return len(self.bars) - 1


def _caller() -> str:
    """The first module on the stack above the interception, outside these supports."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith("tests.support"):
            return name
        frame = frame.f_back
    return ""  # pragma: no cover - the stack always reaches a caller


@dataclass
class TracedDeflator(streams.ScriptedDeflator):
    """The stream's C-06 double, logging what each call assessed and who it could describe.

    Each agent is described alone, as the descriptor module describes one
    agent: a reading of the whole population can differ from it in the last
    place of beta's estimate, because the slope is computed over a batch.

    ``unassessed_undeflated``, where handed, gives every agent the declaration
    leaves unassessed a real, non-``None`` undeflated figure (still with no
    deflated one), so "offered its undeflated figure" and "offered nothing"
    are different numbers; the call log holds the verdicts as returned.
    """

    trace: Trace | None = None
    population: ArrayGenomePopulation | None = None
    unassessed_undeflated: Callable[[FitnessSample], float] | None = None

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        called = super().deflate(samples)
        if self.unassessed_undeflated is not None:
            figure = self.unassessed_undeflated
            called = tuple(
                replace(result, undeflated=figure(sample))
                if result.deflated is None
                else result
                for result, sample in zip(called, samples, strict=True)
            )
            self.calls[-1] = called
        trace, population = self.trace, self.population
        assert trace is not None and population is not None
        trace.events.append(("deflate", trace.bar))
        expected: dict[AgentId, Expected] = {}
        carrying: dict[AgentId, float] = {}
        for result in called:
            try:
                described = descriptors.describe(
                    population.state, population.descriptors, result.agent_id
                )
            except descriptors.DescriptorUnavailable:
                continue
            if result.deflated is None:
                if result.undeflated is not None:
                    carrying[result.agent_id] = result.undeflated
                continue
            expected[result.agent_id] = Expected(
                figure=result.deflated,
                descriptor=described.descriptor(),
                genome=population.genome(result.agent_id),
            )
        trace.expected[trace.bar] = expected
        trace.unassessed_carrying[trace.bar] = carrying
        return called


@dataclass
class Walked:
    """One walk and everything that watched it."""

    state: agent.PopulationState
    population: ArrayGenomePopulation
    port: RecordPort
    deflator: TracedDeflator
    loop: object
    watched: streams.Watched
    run: object
    trace: Trace


def _intercept(population: ArrayGenomePopulation, state, trace: Trace) -> None:
    """Intercept ``record``, ``insert`` and ``depart`` at their instances."""
    record = population.record
    insert = population.archive.insert
    depart = state.depart

    def traced_record(entry):
        living = frozenset(state.agent_ids())
        held = population.recorded()
        caller = _caller()
        trace.events.append(("offer", trace.bar))
        trace.offering = True
        try:
            kept = record(entry)
        finally:
            trace.offering = False
        trace.offers.append(
            Offer(
                bar=trace.bar,
                caller=caller,
                entry=entry,
                kept=kept,
                living_before=living,
                living_after=frozenset(state.agent_ids()),
                held_before=held,
                held_after=population.recorded(),
            )
        )
        return kept

    def traced_insert(*args, **kwargs):
        trace.inserts.append((trace.bar, _caller(), trace.offering))
        return insert(*args, **kwargs)

    def traced_depart(agent_id, *, died_at, kind):
        held = population.recorded()
        trace.events.append(("death", trace.bar))
        departed = depart(agent_id, died_at=died_at, kind=kind)
        trace.deaths.append((trace.bar, agent_id, held, population.recorded()))
        return departed

    population.record = traced_record  # ty: ignore[invalid-assignment]
    population.archive.insert = traced_insert  # ty: ignore[invalid-assignment]
    state.depart = traced_depart  # ty: ignore[invalid-assignment]


def _census_occupancy(port: RecordPort, population, trace: Trace) -> None:
    """Count the living's cells independently at every ``occupancy()`` call."""
    from tests.support import niche_census

    occupancy = port.occupancy

    def censused():
        trace.occupancy_census.append(
            (
                trace.bar,
                frozenset(population.agent_ids()),
                niche_census.living_cells(population),
            )
        )
        return occupancy()

    port.occupancy = censused  # ty: ignore[invalid-assignment]


def walk(
    *,
    windows: Sequence[BarWindow] | None = None,
    founders: int = FOUNDERS,
    turnover_config: object | None = None,
    verdicts: streams.Verdicts | None = None,
    diversity_record: object = UNSET,
    record_carries_across_segments: object = UNSET,
    run_id: str = "record-trace",
    look: Callable[[int], None] | None = None,
    discard_offers: bool = False,
    before_run: Callable[[Walked], None] | None = None,
    max_turnover: float = MAX_TURNOVER,
    unassessed_undeflated: Callable[[FitnessSample], float] | None = None,
) -> Walked:
    """Found, describe, intercept and walk once.

    The descriptor declaration is handed to the loop only where the walk names
    ``diversity_record``, so the default walk is wired exactly as it was before the offers existed.
    ``discard_offers`` makes ``record`` keep nothing and say so, beneath the
    interception, so a check can compare a run whose every offer is discarded.
    ``unassessed_undeflated`` is handed to the double (:class:`TracedDeflator`).
    """
    config = turnover() if turnover_config is None else turnover_config
    keys = {
        "diversity_record": diversity_record,
        "record_carries_across_segments": record_carries_across_segments,
    }
    pass_config = ecology_config(run_id=run_id, turnover=config, **keys)
    state = agent.PopulationState.founded(
        streams.founders(founders),
        population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
        learning=LearningConfig(),
    )
    population = declared(state, max_turnover)
    if discard_offers:
        population.record = lambda entry: False  # ty: ignore[invalid-assignment]
    trace = Trace()
    deflator = TracedDeflator(
        deflated_for_comparisons if verdicts is None else verdicts,  # ty: ignore[invalid-argument-type]
        trace=trace,
        population=population,
        unassessed_undeflated=unassessed_undeflated,
    )
    port = RecordPort(population)
    _intercept(population, state, trace)
    _census_occupancy(port, population, trace)
    extra = (
        {}
        if diversity_record is UNSET
        else {"descriptor_declaration": declaration(max_turnover)}
    )
    loop = ecology.steady_state.SteadyState(
        population=port,
        state=state,
        deflator=deflator,
        config=pass_config,
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
        **extra,  # ty: ignore[invalid-argument-type]
    )
    turn_over = loop._turn_over

    def at_each_bar(window: BarWindow) -> None:
        trace.bars.append(window)
        turn_over(window)
        trace.held_after_bar.append(population.recorded())

    loop._turn_over = at_each_bar  # ty: ignore[invalid-assignment]
    stream = streams.priced(count=DECISION_POINTS) if windows is None else windows
    watched = streams.Watched(stream, state, look=look)
    walked = Walked(
        state=state,
        population=population,
        port=port,
        deflator=deflator,
        loop=loop,
        watched=watched,
        run=None,
        trace=trace,
    )
    if before_run is not None:
        before_run(walked)
    walked.run = loop.run(watched)
    return walked


def reference_walk(**keys: object) -> Walked:
    """The walk :data:`OFFER_FREE_REPORT_DIGEST` was taken from, with any key a check adds."""
    return walk(verdicts=streams.lowest_is_worse, **keys)  # ty: ignore[invalid-argument-type]


def report_digest(report: object) -> str:
    """The sha256 of ``repr(report)``: every byte of the report, as Python renders it."""
    return hashlib.sha256(repr(report).encode("utf-8")).hexdigest()


def _same_descriptor(one: Descriptor, other: Descriptor) -> bool:
    """Equal vocabulary, axes and values, an undefined (``nan``) value equal to itself."""
    return (
        one.schema_version == other.schema_version
        and list(one.values) == list(other.values)
        and all(
            value == other.values[axis]
            or (math.isnan(value) and math.isnan(other.values[axis]))
            for axis, value in one.values.items()
        )
    )


def offer_problems(walked: Walked) -> list[str]:
    """Everything the trace shows the loop offered other than as declared.

    Read against the deflate calls: every offer from the loop, inside a
    trigger, after its one deflate call and before its first death; at each
    trigger the offered agents exactly those the call assessed and the
    population could describe; each entry carrying that call's figure under
    ``"deflated"``, the agent's genome and descriptor there, and the trigger's
    instant; every insert from the adapter, inside an offer; and the record
    holding what the archive holds.
    """
    trace = walked.trace
    problems: list[str] = []
    seen: dict[int, set[str]] = {}
    for at, (kind, bar) in enumerate(trace.events):
        if kind != "offer":
            continue
        before = trace.events[:at]
        if ("deflate", bar) not in before:
            problems.append(f"an offer at bar {bar} came before any deflate call there")
        if ("death", bar) in before:
            problems.append(f"an offer at bar {bar} came after a death there")
    for offer in trace.offers:
        bar, entry = offer.bar, offer.entry
        where = f"bar {bar}, {entry.agent_id}"
        if offer.caller != LOOP:
            problems.append(f"{where}: offered by {offer.caller or 'nobody'}")
        expected = trace.expected.get(bar, {})
        if entry.agent_id in seen.setdefault(bar, set()):
            problems.append(f"{where}: offered twice at one trigger")
        seen[bar].add(entry.agent_id)
        if entry.agent_id not in expected:
            problems.append(f"{where}: neither assessed nor described there")
            continue
        declared_offer = expected[entry.agent_id]
        if dict(entry.objectives) != {"deflated": declared_offer.figure}:
            problems.append(
                f"{where}: carried {dict(entry.objectives)}, not that call's "
                f"deflated figure {declared_offer.figure}"
            )
        if not _same_descriptor(entry.descriptor, declared_offer.descriptor):
            problems.append(f"{where}: carried another descriptor")
        if entry.genome != declared_offer.genome:
            problems.append(f"{where}: carried another genome")
        if bar < 0 or entry.recorded_at != trace.bars[bar].as_of:
            problems.append(f"{where}: recorded at another instant")
    for bar, expected in trace.expected.items():
        if set(expected) != seen.get(bar, set()):
            problems.append(
                f"bar {bar}: offered {sorted(seen.get(bar, set()))}, assessed and "
                f"described {sorted(expected)}"
            )
    deflates = [bar for kind, bar in trace.events if kind == "deflate"]
    if len(deflates) != len(set(deflates)):
        problems.append("a trigger made more than one deflate call")
    for bar, caller, inside in trace.inserts:
        if caller != ADAPTER or not inside:
            problems.append(f"bar {bar}: an insert by {caller} outside an offer")
    if len(trace.inserts) != len(trace.offers):
        problems.append(f"{len(trace.inserts)} inserts for {len(trace.offers)} offers")
    if list(walked.population.recorded()) != walked.population.archive.elites():
        problems.append("the record does not hold what the archive holds")
    return problems


def decided(walked: Walked) -> dict[str, object]:
    """What the loop decided: births, deaths, the living, and each trigger's row as it is with the offers off."""
    report = walked.run.report  # ty: ignore[unresolved-attribute]
    row = ecology.steady_state.Trigger
    return {
        "births": [
            (
                birth.agent_id,
                birth.parents,
                birth.genome,
                birth.spawn_seed,
                birth.index,
                birth.attempts,
            )
            for birth in report.births
        ],
        "deaths": [(death.kind, death.agent_id) for death in report.deaths],
        "living": list(walked.watched.living),
        "triggers": [
            {each.name: getattr(trigger, each.name) for each in fields(row)}
            for trigger in report.triggers
        ],
    }


def with_figures(
    verdicts: Sequence[DeflatedFitness], figure: Callable[[DeflatedFitness], float]
) -> tuple[DeflatedFitness, ...]:
    """``verdicts`` with every present deflated figure replaced, for a seeded offer."""
    return tuple(
        verdict
        if verdict.deflated is None
        else replace(verdict, deflated=figure(verdict))
        for verdict in verdicts
    )
