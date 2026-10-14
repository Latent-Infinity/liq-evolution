"""Evidence that the diversity record is reported as it is, and restores into its own run.

``ECOLOGY.NICHE.COVERAGE_REPORTED.v1`` (EV-E81, library half): given a run with
the record's offers configured on (``diversity_record = "record_only"``), when
its report is read, then at every trigger it states the archive's coverage —
read from the record through the population port and labelled cumulative —
beside the living occupancy at that trigger, read through the port under the
archive's own binning; and the initial coverage after the first trigger at
which an entry was offered, with that trigger's index and instant, or, when no
entry was ever offered, absent with its reason and never as zero filled cells.
Each figure is checked against what is read from outside the loop: the
archive's own ``coverage_report()``, the cells of ``recorded()`` and of the
living counted by ``tests/support/niche_census.py`` under the archive's public
binning. Configuring the offers on is not niching, and nothing here is evidence
that diversity is retained.

``ECOLOGY.NICHE.ARCHIVE_ROUND_TRIPS.v1`` (EV-E18): given a diversity record
persisted by a run, when it is restored into that run, then every entry — its
agent, genome, descriptor, figure and recording instant — and the record's
coverage are recovered without loss; and a payload whose envelope names another
archive payload version or another run is refused rather than loaded.

**What is round-tripped.** A record on the four declared axes under
``ecology-descriptors-2``, binned at the descriptor-parameter decision record's
counts. It is filled by walking the declared stream S1 through the steady-state
loop. With the record's offers off the loop offers nothing, so this check builds the entries
itself, shaped as the loop's would be: at each replacement trigger, after that
trigger's one deflate call and before its first death, one entry for each agent
that call assessed and whose descriptor is defined — its genome and descriptor
as the population holds them there, the deflated figure that call returned under
``"deflated"``, and the trigger's instant. An agent short of the descriptor
window is not offered and is given no stand-in. This demonstrates the codec's
fidelity on entries shaped as the loop's; it does not demonstrate a record the
loop produced.

**Asserted, not assumed.** The platform archive restores by replaying insertion,
so each restored entry is compared with the one persisted, field by field, with
its cell and the coverage. The record includes evictions, and a second record of
two entries per cell is round-tripped too, so that the order inside a crowded
cell is exercised. Two wrong codecs — one that drops the genome, one that loads
whatever envelope it is handed — are each shown to turn these checks red.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType, ModuleType

import pytest

from liq.evolution import ecology
from liq.evolution.config import FitnessConfig
from liq.evolution.ecology import Genome, agent, descriptors
from liq.evolution.ecology.adapters import ArrayGenomePopulation, NullRiskSizer
from liq.evolution.ecology.config import LearningConfig, PopulationConfig
from liq.evolution.ecology.types import (
    ArchiveEntry,
    BarWindow,
    DeflatedFitness,
    FitnessSample,
    UtcTimestamp,
)
from liq.gp.evolution.qd_archive import QDArchive
from tests.support import niche_census, record_trace
from tests.support import turnover_stream as streams

#: How many decision points the walk covers: many triggers past the warm-up, so
#: agents are described, offered, evicted, die and are born.
DECISION_POINTS = 120

#: How many founders the walk starts from.
FOUNDERS = 14

#: The descriptor window: the run's fitness window.
WINDOW = 8

#: The bins of the holding horizon, turnover and beta axes, as the
#: descriptor-parameter decision record states them
#: (``liq-docs/plans/agent-ecology-adr-descriptor-parameters.md``).
DECIDED_BINS = (3, 16, 9)

#: The one declared asset class of the one traded instrument.
ASSET_CLASS = "equity"

#: The run the record is written under, and another run it must not enter.
RUN_ID = "archive-round-trip"
ANOTHER_RUN = "another-archive-round-trip"

#: The cell capacity of the second, crowded record: more than the archive's
#: default of one, so entries share a cell and the order inside it matters.
CROWDED_CAPACITY = 2


def _declaration() -> descriptors.DescriptorDeclaration:
    return descriptors.DescriptorDeclaration(
        window=WINDOW,
        history_capacity=streams.HISTORY_CAPACITY,
        asset_classes=(ASSET_CLASS,),
        universe={streams.INSTRUMENT: ASSET_CLASS},
        mandate=descriptors.Mandate(
            max_position=1.0,
            max_gross_exposure=NullRiskSizer().max_gross_exposure,
            max_weight_delta=2.0,
            max_turnover=1.0,
        ),
    )


def _declared(state: agent.PopulationState) -> ArrayGenomePopulation:
    return ArrayGenomePopulation(
        state=state,
        descriptor_schema_version=descriptors.VOCABULARY,
        descriptors=_declaration(),
        descriptor_bins=DECIDED_BINS,
    )


def _turnover():
    return ecology.steady_state.TurnoverConfig(
        band_lower=14,
        band_upper=20,
        replacement_count=2,
        trigger_period=3,
        minimum_age=6,
        replacement_window=6,
        maximum_replaced_fraction=0.3,
        fitness_window=WINDOW,
        parents_per_birth=2,
        inheritance="baldwinian",
        admission_redraws=8,
        seed=11,
    )


@dataclass
class Offering(streams.ScriptedDeflator):
    """The stream's deflator double, building an entry per assessed, described agent.

    Each call is one trigger's single deflate call. Right after it — inside the
    trigger, before any death there — each agent the call assessed and the
    declared population can describe is offered once, carrying that call's
    deflated figure and the trigger's instant. Nothing the loop decides reads
    what is offered.
    """

    population: ArrayGenomePopulation | None = None
    at: UtcTimestamp | None = None
    offered: list[ArchiveEntry] = field(default_factory=list)
    kept: list[bool] = field(default_factory=list)
    undescribed: int = 0

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        called = super().deflate(samples)
        assert self.population is not None and self.at is not None
        for result in called:
            if result.deflated is None:
                continue
            try:
                described = self.population.descriptor(result.agent_id)
            except descriptors.DescriptorUnavailable:
                self.undescribed += 1
                continue
            entry = ArchiveEntry(
                agent_id=result.agent_id,
                genome=self.population.genome(result.agent_id),
                descriptor=described,
                objectives=MappingProxyType({"deflated": result.deflated}),
                recorded_at=self.at,
            )
            self.offered.append(entry)
            self.kept.append(self.population.record(entry))
        return called


@dataclass
class Walked:
    """One walk, the declared population its record was filled on, and what was offered."""

    population: ArrayGenomePopulation
    deflator: Offering


def _walk() -> Walked:
    deflator = Offering(streams.per_sample(streams.negative_is_worse))
    wired = streams.wire(
        founding=streams.founders(FOUNDERS),
        turnover=_turnover(),
        deflator=deflator,
        run_id=RUN_ID,
    )
    deflator.population = _declared(wired.state)
    windows = streams.priced(count=DECISION_POINTS)
    turn_over = wired.loop._turn_over  # ty: ignore[unresolved-attribute]

    def at_the_trigger(window: BarWindow) -> None:
        deflator.at = window.as_of
        turn_over(window)

    wired.loop._turn_over = at_the_trigger  # ty: ignore[unresolved-attribute]
    wired.loop.run(streams.Watched(windows, wired.state))  # ty: ignore[unresolved-attribute]
    return Walked(population=deflator.population, deflator=deflator)


@pytest.fixture
def codec() -> ModuleType:
    """The record's codec, imported per case so each one reds on its own."""
    return importlib.import_module("liq.evolution.ecology.adapters.archive_codec")


@pytest.fixture(scope="module")
def walked() -> Walked:
    """The walk is deterministic and read-only below, so it is taken once."""
    return _walk()


def _fresh() -> ArrayGenomePopulation:
    """A fresh declared population: founded, nothing recorded."""
    return _declared(
        agent.PopulationState.founded(
            streams.founders(FOUNDERS),
            population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
            learning=LearningConfig(),
        )
    )


def _cells(
    archive: QDArchive,
) -> list[tuple[tuple[int, ...], list[float], list[float]]]:
    """Each occupant's cell, figures and coordinates, in the archive's own order."""
    return [
        (tuple(raw["bin"]), list(raw["objectives"]), list(raw["descriptors"]))
        for raw in archive.to_dict()["entries"]  # ty: ignore[not-iterable]
    ]


def round_trip_problems(
    persisted: Sequence[ArchiveEntry],
    restored: Sequence[ArchiveEntry],
    before: QDArchive,
    after: QDArchive,
) -> list[str]:
    """What the restored record lost or changed, entry by entry and cell by cell."""
    problems: list[str] = []
    if len(restored) != len(persisted):
        problems.append(f"{len(persisted)} entries persisted, {len(restored)} restored")
    for index, (was, now) in enumerate(zip(persisted, restored, strict=False)):
        for name in ("agent_id", "genome", "descriptor", "objectives", "recorded_at"):
            if getattr(was, name) != getattr(now, name):
                problems.append(f"entry {index} ({was.agent_id}) lost its {name}")
    if _cells(after) != _cells(before):
        problems.append("the restored entries sit in other cells, or in another order")
    if after.coverage_report() != before.coverage_report():
        problems.append("the restored coverage differs from the persisted coverage")
    return problems


def test_archive_round_trip(walked: Walked) -> None:
    """Every entry, its cell and the coverage come back exactly, evictions included."""
    population = walked.population
    held = population.recorded()
    assert len(held) >= 2, "the walk recorded too few entries to round-trip"
    assert sum(walked.deflator.kept) > len(held), (
        "no kept entry was later evicted, so the record never exercised eviction"
    )
    assert walked.deflator.undescribed > 0, (
        "no assessed agent was short of the window, so the no-stand-in rule was "
        "never exercised"
    )
    for entry in held:
        assert entry.descriptor.schema_version == descriptors.VOCABULARY
        assert tuple(entry.descriptor.values) == descriptors.AXES
        assert set(entry.objectives) == {"deflated"}

    payload = population.persisted(RUN_ID)
    fresh = _fresh()
    fresh.restore(payload, RUN_ID)

    assert (
        round_trip_problems(held, fresh.recorded(), population.archive, fresh.archive)
        == []
    )
    assert fresh.coverage() == population.coverage()


def _crowded(entries: Sequence[ArchiveEntry]) -> QDArchive:
    """The platform archive on the declared axes, two entries a cell, holding ``entries``."""
    archive = QDArchive(
        n_dims=len(descriptors.AXES),
        bins_per_dim=_declaration().archive_bins(
            holding_horizon=DECIDED_BINS[0],
            turnover=DECIDED_BINS[1],
            beta=DECIDED_BINS[2],
        ),
        descriptor_bounds=((0.0, 1.0),) * len(descriptors.AXES),
        objective_directions=list(FitnessConfig().objective_directions),
        bin_capacity=CROWDED_CAPACITY,
    )
    for entry in entries:
        archive.insert(
            entry,
            (entry.objectives["deflated"],),
            tuple(entry.descriptor.values[axis] for axis in descriptors.AXES),
        )
    return archive


def test_archive_round_trip_with_more_than_one_entry_a_cell(
    walked: Walked, codec: ModuleType
) -> None:
    """The order inside a crowded cell survives, because the restore replays it."""
    offered = [
        entry
        for entry in walked.deflator.offered
        if all(0.0 <= value <= 1.0 for value in entry.descriptor.values.values())
    ]
    before = _crowded(offered)
    shared = [cell for cell, *_ in _cells(before)]
    assert len(shared) > len(set(shared)), (
        "no cell holds two entries, so the crowded order was never exercised"
    )
    payload = codec.persisted(
        before,
        run_id=RUN_ID,
        descriptor_schema_version=descriptors.VOCABULARY,
        axes=descriptors.AXES,
    )
    after = _crowded(())
    codec.restore(
        after,
        payload,
        run_id=RUN_ID,
        descriptor_schema_version=descriptors.VOCABULARY,
        axes=descriptors.AXES,
    )

    assert (
        round_trip_problems(
            [entry for entry in before.elites() if isinstance(entry, ArchiveEntry)],
            [entry for entry in after.elites() if isinstance(entry, ArchiveEntry)],
            before,
            after,
        )
        == []
    )


def _refused(population: ArrayGenomePopulation, payload: object, run_id: str) -> bool:
    """Whether ``population`` refuses to restore ``payload`` into ``run_id``."""
    try:
        population.restore(payload, run_id)  # ty: ignore[invalid-argument-type]
    except ValueError:
        return True
    return False


def foreign_version_problems(population: ArrayGenomePopulation) -> list[str]:
    """What a payload under another version envelope loads into a fresh population."""
    foreign = replace(population.persisted(RUN_ID), payload_version="another-version")
    fresh = _fresh()
    if not _refused(fresh, foreign, RUN_ID):
        return ["a payload under another version envelope was loaded"]
    if fresh.recorded() != ():
        return ["a refused payload left entries behind"]
    return []


def another_run_problems(population: ArrayGenomePopulation) -> list[str]:
    """What another run's payload loads, whether named in the call or in the envelope."""
    payload = population.persisted(RUN_ID)
    problems: list[str] = []
    for restored_into, handed in (
        (ANOTHER_RUN, payload),
        (RUN_ID, replace(payload, run_id=ANOTHER_RUN)),
    ):
        fresh = _fresh()
        if not _refused(fresh, handed, restored_into):
            problems.append(
                f"a payload written by {handed.run_id!r} was loaded into "
                f"{restored_into!r}"
            )
        elif fresh.recorded() != ():
            problems.append("a refused payload left entries behind")
    return problems


def test_a_foreign_archive_version_is_refused(walked: Walked) -> None:
    """A payload under another version envelope loads nothing."""
    assert foreign_version_problems(walked.population) == []


def test_a_payload_from_another_run_is_refused(walked: Walked) -> None:
    """A record restores only into the run that wrote it."""
    assert another_run_problems(walked.population) == []


def test_a_payload_is_written_in_the_current_version(
    walked: Walked, codec: ModuleType
) -> None:
    """The envelope names the version, the run, the vocabulary and the cells."""
    payload = walked.population.persisted(RUN_ID)
    assert isinstance(payload, ecology.RecordPayload)  # ty: ignore[unresolved-attribute]
    assert payload.payload_version == codec.PAYLOAD_VERSION
    assert payload.run_id == RUN_ID
    assert payload.descriptor_schema_version == descriptors.VOCABULARY
    assert payload.axes == descriptors.AXES
    assert payload.bins == walked.population.archive.bins_per_dim
    assert payload.bin_capacity == walked.population.archive.bin_capacity
    assert all(isinstance(each, str) for each in payload.entries)


def test_a_codec_that_drops_the_genome_turns_the_round_trip_red(
    walked: Walked, codec: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeded: a restore that keeps everything but the genome is caught."""
    decode = codec.decode_entry

    def without_the_genome(text: str) -> ArchiveEntry:
        entry = decode(text)
        return replace(
            entry,
            genome=Genome(genes=MappingProxyType({}), schema_version=""),
        )

    monkeypatch.setattr(codec, "decode_entry", without_the_genome)
    population = walked.population
    fresh = _fresh()
    fresh.restore(population.persisted(RUN_ID), RUN_ID)
    problems = round_trip_problems(
        population.recorded(), fresh.recorded(), population.archive, fresh.archive
    )
    assert problems
    assert all("genome" in problem for problem in problems)


def test_a_codec_that_ignores_the_envelope_turns_the_refusals_red(
    walked: Walked, codec: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeded: a restore that loads whatever it is handed is caught by both refusals."""
    monkeypatch.setattr(codec, "refuse_a_foreign_payload", lambda *args, **kwargs: None)
    assert foreign_version_problems(walked.population) != []
    assert another_run_problems(walked.population) != []


# ---------------------------------------------------------------------------
# Coverage, as the turnover report states it
# ---------------------------------------------------------------------------


def _census_excluded_at_each_occupancy(
    walked: record_trace.Walked, censused: dict[int, tuple[int, int]]
) -> None:
    """At each ``occupancy()`` call, count the living the record cannot bin.

    Read beside ``record_trace``'s own census, at the same instant: each living
    agent the population port describes is placed with the platform archive's
    public ``cell_of``; one it refuses is excluded. Stored per bar as
    ``(cells, excluded)``.
    """
    population, trace = walked.population, walked.trace
    occupancy = walked.port.occupancy

    def counted():
        excluded = 0
        for agent_id in population.agent_ids():
            try:
                described = population.descriptor(agent_id)
            except descriptors.DescriptorUnavailable:
                continue
            try:
                population.archive.cell_of(
                    tuple(described.values[axis] for axis in descriptors.AXES)
                )
            except ValueError:
                excluded += 1
        made = occupancy()
        censused[trace.bar] = (trace.occupancy_census[-1][2], excluded)
        return made

    walked.port.occupancy = counted  # ty: ignore[invalid-assignment]


def test_the_turnover_report_states_coverage_from_the_record() -> None:
    """Every trigger row's coverage is the record's own, beside the living's cells.

    A second walk is the record-only walk with ``max_turnover=1.0e-3``, the
    walk the exclusion case in ``tests/ecology/test_steady_state.py`` uses:
    its agents' turnover lies outside the unit interval, so living agents with
    a defined descriptor are refused a cell by ``cell_of``. Its premise — at
    least one trigger with such an agent — is asserted, and there occupancy's
    ``occupied`` and ``excluded`` equal the census
    (``liq-docs/plans/oracle-v4-sufficiency-gaps-2026-10-14.md`` §2.2).
    """
    read: dict[int, tuple[object, dict[str, object], int]] = {}
    holder: dict[str, record_trace.Walked] = {}

    def look(index: int) -> None:
        if "walked" not in holder or index < 0:
            return
        population = holder["walked"].population
        read[index] = (
            population.coverage(),
            population.archive.coverage_report(),
            niche_census.record_cells(population),
        )

    walked = record_trace.walk(
        diversity_record="record_only",
        look=look,
        before_run=lambda walked: holder.update(walked=walked),
    )
    report = walked.run.report  # ty: ignore[unresolved-attribute]
    census = {bar: (who, count) for bar, who, count in walked.trace.occupancy_census}

    assert report.triggers, "no trigger fell, so no coverage was reported"
    for row in report.triggers:
        coverage, platform, cells = read[row.index]
        assert row.cumulative_coverage == coverage, row.index
        assert row.cumulative_coverage.filled == platform["filled_bins"] == cells
        assert row.cumulative_coverage.total == platform["total_bins"]
        living, occupied = census[row.index]
        assert living == walked.watched.living[row.index], (
            "living occupancy was not read over the living at the offer instant"
        )
        assert row.living_occupancy.occupied == occupied, row.index
    filled = [row.cumulative_coverage.filled for row in report.triggers]
    assert filled == sorted(filled), "a coverage labelled cumulative fell"
    assert filled[-1] > 0
    first = next(row for row in report.triggers if row.offers)
    initial = report.initial_coverage
    assert initial.absent is None
    assert initial.trigger is first
    assert (initial.trigger.index, initial.trigger.as_of) == (first.index, first.as_of)
    assert initial.coverage == read[first.index][0]

    censused: dict[int, tuple[int, int]] = {}
    excluding = record_trace.walk(
        diversity_record="record_only",
        max_turnover=1.0e-3,
        before_run=lambda walked: _census_excluded_at_each_occupancy(walked, censused),
    )
    rows = excluding.run.report.triggers  # ty: ignore[unresolved-attribute]
    with_excluded = [row for row in rows if censused[row.index][1] > 0]
    assert with_excluded, (
        "no trigger held a living, described agent the record cannot bin, so "
        "exclusion from occupancy was never exercised"
    )
    for row in with_excluded:
        cells, excluded = censused[row.index]
        assert row.living_occupancy.occupied == cells, row.index
        assert row.living_occupancy.excluded == excluded, row.index


def test_a_run_with_no_offer_reports_initial_coverage_absent() -> None:
    """A run too short for any agent to be assessed and described has none."""
    walked = record_trace.walk(
        diversity_record="record_only",
        windows=streams.priced(count=record_trace.WINDOW + 1),
    )
    report = walked.run.report  # ty: ignore[unresolved-attribute]

    assert report.triggers, "no trigger fell, so the absent case was never reached"
    assert any(row.not_offered_descriptor_undefined for row in report.triggers), (
        "no assessed agent went undescribed, so the run was not the declared one"
    )
    assert all(row.offers == 0 for row in report.triggers)
    initial = report.initial_coverage
    assert initial.trigger is None
    assert initial.coverage is None, "an absent initial coverage was given a figure"
    assert initial.absent == "no agent was both assessed and described"
