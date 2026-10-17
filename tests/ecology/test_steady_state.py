"""How the steady-state loop reads its verdicts, and how it breeds.

The facts the loop carries are evidenced in ``tests/facts/test_population.py``.
What is pinned here is the rest of the rule
(``liq-docs/plans/oracle-v3-indistinguishable-cull-2026-09-30.md`` §2.2–§2.3,
``liq-docs/plans/oracle-v3-unassessed-fitness-2026-10-01.md`` §2.2–§2.4):

* the cull orders on the **deflated** figure the C-06 port returns, never the
  undeflated; an agent the port marks indistinguishable, or does not assess at
  all, is never culled, only retired; retirements go oldest first, and an agent
  marked better is retired only when nobody else is left, as a protected
  retirement;
* a deflator whose verdicts respond to the number compared changes the cull set;
* every death, cull or retirement, leaves through the same exit;
* births happen at the configured trigger, from parents the population port
  selected on the same verdicts that decided the culls — no unassessed or
  too-young parent at an ordinary trigger, and a whole trigger drawn blind, at
  the same birth count, when too few agents carry a figure;
* the inheritance setting decides what crosses a birth, bit for bit;
* with the record's offers on, the loop decides exactly what it decides with them
  off, and every trigger row reports what was offered and read; with them off,
  the default, the report is the offer-free loop's to the byte;
* a walk-forward split boundary triggers nothing and resets nothing;
* the configurations the records refuse are refused at construction.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent, descriptors
from liq.evolution.ecology.adapters import (
    ArrayGenomePopulation,
    NullExecutionSimulator,
    NullFitnessDeflator,
    NullRiskSizer,
)
from liq.evolution.ecology.config import EcologyConfig, LearningConfig, PopulationConfig
from liq.evolution.ecology.types import ArchiveEntry, DeflatedFitness, Descriptor
from tests.support import level_shift_stream as s1
from tests.support import record_trace
from tests.support import turnover_stream as streams

DECISION_POINTS = 90


def _turnover(**keys: object):
    declared: dict[str, object] = {
        "band_lower": 14,
        "band_upper": 20,
        "replacement_count": 2,
        "trigger_period": 3,
        "minimum_age": 6,
        "replacement_window": 6,
        "maximum_replaced_fraction": 0.3,
        "fitness_window": 8,
        "parents_per_birth": 2,
        "inheritance": "baldwinian",
        "admission_redraws": 8,
        "seed": 5,
    }
    declared.update(keys)
    return ecology.steady_state.TurnoverConfig(**declared)


def _figure(
    agent_id: str,
    *,
    deflated: float | None,
    undeflated: float | None = 0.0,
    standing: str = "indistinguishable",
) -> DeflatedFitness:
    return DeflatedFitness(
        agent_id=agent_id,
        undeflated=undeflated,  # ty: ignore[invalid-argument-type]
        deflated=deflated,  # ty: ignore[invalid-argument-type]
        compared=6,
        independent_outcomes=6,
        standing=standing,  # ty: ignore[invalid-argument-type]
    )


def _candidate(agent_id: str, *, age: int, born: int, admitted: int, fitness=None):
    return ecology.steady_state.Candidate(
        agent_id=agent_id, age=age, born_index=born, admitted=admitted, fitness=fitness
    )


def _chosen(living, k: int = 2, m: int = 6) -> list[tuple[str, str]]:
    return [
        (each.agent_id, str(kind))
        for each, kind in ecology.steady_state.choose_deaths(
            living, replacement_count=k, minimum_age=m
        )
    ]


# ---------------------------------------------------------------------------
# The rule, at one trigger
# ---------------------------------------------------------------------------


def test_the_cull_orders_on_the_deflated_figure_never_the_undeflated() -> None:
    """Of two worse agents, the lower deflated figure dies, whatever the undeflated says."""
    living = [
        _candidate(
            "low-deflated",
            age=9,
            born=1,
            admitted=1,
            fitness=_figure(
                "low-deflated", deflated=-2.0, undeflated=5.0, standing="worse"
            ),
        ),
        _candidate(
            "low-undeflated",
            age=9,
            born=0,
            admitted=0,
            fitness=_figure(
                "low-undeflated", deflated=-1.0, undeflated=-5.0, standing="worse"
            ),
        ),
    ]
    assert _chosen(living, k=1) == [("low-deflated", "cull")]


def test_nobody_indistinguishable_or_unassessed_is_culled_only_retired() -> None:
    """With nobody worse, the oldest the evidence is silent on retire, assessed or not."""
    living = [
        _candidate("young", age=2, born=7, admitted=9),
        _candidate(
            "unassessed",
            age=8,
            born=1,
            admitted=1,
            fitness=_figure("unassessed", deflated=None, undeflated=None),
        ),
        _candidate(
            "low-but-silent",
            age=9,
            born=1,
            admitted=2,
            fitness=_figure("x", deflated=-9.0),
        ),
        _candidate(
            "oldest", age=9, born=0, admitted=5, fitness=_figure("o", deflated=3.0)
        ),
    ]
    assert _chosen(living) == [
        ("oldest", "retirement"),
        ("unassessed", "retirement"),
    ]


def test_retirement_breaks_ties_by_birth_bar_then_admission_order() -> None:
    living = [
        _candidate(
            name,
            age=9,
            born=born,
            admitted=admitted,
            fitness=_figure(name, deflated=0.0),
        )
        for name, born, admitted in (("c", 2, 0), ("b", 1, 7), ("a", 1, 3))
    ]
    assert [name for name, _ in _chosen(living, k=3)] == ["a", "b", "c"]


def test_better_agents_are_retired_only_when_nobody_else_is_left() -> None:
    """A better agent is protected until it is the only kind of agent left to retire."""
    living = [
        _candidate(
            "better-oldest",
            age=12,
            born=0,
            admitted=0,
            fitness=_figure("b", deflated=5.0, standing="better"),
        ),
        _candidate(
            "silent", age=7, born=5, admitted=4, fitness=_figure("s", deflated=0.0)
        ),
        _candidate(
            "worse",
            age=7,
            born=5,
            admitted=5,
            fitness=_figure("w", deflated=-1.0, standing="worse"),
        ),
    ]
    assert _chosen(living, k=2) == [("worse", "cull"), ("silent", "retirement")]
    assert _chosen(living, k=3) == [
        ("worse", "cull"),
        ("silent", "retirement"),
        ("better-oldest", "protected_retirement"),
    ]


def test_the_count_is_the_replacement_count_or_the_eligible_pool() -> None:
    """Exactly ``min(k, eligible)`` die, and nobody below the minimum age makes it up."""
    living = [
        _candidate(
            "old", age=6, born=0, admitted=0, fitness=_figure("o", deflated=0.0)
        ),
        _candidate("young", age=5, born=1, admitted=1),
        _candidate("younger", age=1, born=5, admitted=2),
    ]
    assert _chosen(living, k=2, m=6) == [("old", "retirement")]
    assert _chosen(living[1:], k=2, m=6) == []


def test_a_worse_agent_without_a_figure_is_not_culled() -> None:
    """A verdict with no figure is no evidence; the agent is only retired."""
    stray = _figure("stray", deflated=None, undeflated=None)
    living = [_candidate("stray", age=9, born=0, admitted=0, fitness=stray)]
    assert _chosen(living, k=1) == [("stray", "retirement")]


# ---------------------------------------------------------------------------
# The loop, over a declared stream
# ---------------------------------------------------------------------------


def _run(**keys: object) -> streams.Steady:
    deflator = keys.pop("deflator", None)
    windows = keys.pop("windows", None)
    founding = keys.pop("founding", None)
    learning = keys.pop("learning", None)
    look = keys.pop("look", None)
    refused = keys.pop("refused_spawns", frozenset())
    config = _turnover(**keys)
    return streams.steady(
        refused_spawns=refused,
        windows=streams.priced(count=DECISION_POINTS) if windows is None else windows,
        founding=streams.founders(config.band_lower) if founding is None else founding,
        turnover=config,
        deflator=(
            streams.ScriptedDeflator(streams.per_sample(streams.negative_is_worse))
            if deflator is None
            else deflator
        ),
        learning=learning,
        look=look,
    )


def test_births_and_deaths_happen_only_at_the_configured_trigger() -> None:
    run = _run()
    config = run.loop.turnover
    for index in range(DECISION_POINTS):
        scheduled = index > 0 and index % config.trigger_period == 0
        if not scheduled:
            assert not run.watched.deaths(index), index
            assert not run.watched.births(index), index
    assert any(run.watched.deaths(index) for index in range(DECISION_POINTS))


def test_a_deflator_that_responds_to_the_number_compared_changes_the_cull_set() -> None:
    """The structural twin of DEFLATED_FOR_COMPARISONS: more compared, a different cull set."""

    def harsher_for_fewer(samples):
        bar = 1.0e-3 / len(samples)
        return [
            (
                math.fsum(each.outcomes),
                "worse" if math.fsum(each.outcomes) < -bar else "indistinguishable",
            )
            for each in samples
        ]

    culled = []
    for founders in (14, 18):
        run = _run(
            band_lower=14,
            band_upper=20,
            founding=streams.founders(founders),
            deflator=streams.ScriptedDeflator(harsher_for_fewer),
        )
        culled.append(
            {death.agent_id for death in run.run.report.deaths if death.kind == "cull"}
        )
    assert culled[0] != culled[1]


def test_culls_and_retirements_leave_through_one_exit(monkeypatch) -> None:
    """Every death is one store departure and one account close, whatever decided it."""
    closes: list[str] = []
    departures: list[tuple[str, str]] = []
    close = ecology.population_accounts.PopulationAccounts.close
    depart = agent.PopulationState.depart

    def counting_close(self, agent_id, *, as_of):
        closes.append(agent_id)
        return close(self, agent_id, as_of=as_of)

    def counting_depart(self, agent_id, *, died_at, kind):
        departures.append((agent_id, str(kind)))
        return depart(self, agent_id, died_at=died_at, kind=kind)

    monkeypatch.setattr(
        ecology.population_accounts.PopulationAccounts, "close", counting_close
    )
    monkeypatch.setattr(agent.PopulationState, "depart", counting_depart)
    run = _run(deflator=streams.ScriptedDeflator(streams.lowest_is_worse))

    deaths = run.run.report.deaths
    assert {death.kind for death in deaths} >= {"cull", "retirement"}
    assert closes == [death.agent_id for death in deaths]
    assert departures == [(death.agent_id, str(death.kind)) for death in deaths]
    evaluation = run.run.evaluation
    for death in deaths:
        account = evaluation.accounts[death.agent_id]
        if death.agent_id not in evaluation.closed_at:
            continue
        assert account.fills[-1].requested_exposure == 0.0
        assert account.states[-1].as_of > death.died_at
        assert run.state.lineage(death.agent_id).death_kind == death.kind


def test_an_unassessed_agent_is_retired_and_its_death_carries_no_figure() -> None:
    def silent_on_the_flat(sample):
        if not any(sample.outcomes):
            return None
        return streams.negative_is_worse(sample)

    run = _run(
        deflator=streams.ScriptedDeflator(streams.per_sample(silent_on_the_flat))
    )
    unassessed = [death for death in run.run.report.deaths if death.deflated is None]
    assert unassessed, "no unassessed agent died, so the case was never reached"
    assert all(death.kind == "retirement" for death in unassessed)
    assert all(death.undeflated is None for death in unassessed)
    assert any(trigger.unassessed for trigger in run.run.report.triggers)


def test_an_ordinary_trigger_breeds_only_from_assessed_eligible_agents() -> None:
    """No parent is unassessed or below minimum age where enough agents carry a figure."""

    def silent_on_some(sample):
        if sample.agent_id.endswith(("0", "5")):
            return None
        return streams.negative_is_worse(sample)

    run = _run(deflator=streams.ScriptedDeflator(streams.per_sample(silent_on_some)))
    breeding = [t for t in run.run.report.triggers if t.deaths]
    assert len(run.port.handed) == len(breeding), "one selection per breeding trigger"
    ordinary = 0
    for trigger, handed, selected in zip(
        breeding, run.port.handed, run.port.selected, strict=True
    ):
        assert set(handed) == set(run.watched.living[trigger.index + 1]) - {
            birth.agent_id
            for birth in run.run.report.births
            if birth.index == trigger.index
        }, "the figures handed to selection were not keyed by exactly the living"
        if trigger.blind:
            assert all(value is None for value in handed.values())
            continue
        ordinary += 1
        assert any(value is None for value in handed.values()), (
            "no agent was handed without a figure, so exclusion was never tested"
        )
        assert all(handed[parent] is not None for parent in selected)
    assert ordinary, "no ordinary trigger bred"


def test_a_trigger_with_too_few_assessed_is_bred_blind_at_the_same_count() -> None:
    """Below the port's floor every parent is drawn blind, and as many are born."""

    def assesses_three(samples):
        assessed = {
            each.agent_id for each in sorted(samples, key=lambda s: s.agent_id)[:3]
        }
        return [
            (math.fsum(each.outcomes), "indistinguishable")
            if each.agent_id in assessed
            else None
            for each in samples
        ]

    blind_run = _run(deflator=streams.ScriptedDeflator(assesses_three))
    ordinary_run = _run()
    blind = [t for t in blind_run.run.report.triggers if t.births]
    ordinary = [t for t in ordinary_run.run.report.triggers if t.births]
    assert blind and all(trigger.blind for trigger in blind)
    assert [t.births for t in blind] == [t.births for t in ordinary]
    assert all(
        all(value is None for value in handed.values())
        for handed in blind_run.port.handed
    )
    assert all(birth.blind for birth in blind_run.run.report.births)
    assert all(
        trigger.candidates < blind_run.port.minimum_candidates for trigger in blind
    )


def test_one_deflate_call_per_trigger_decides_both_deaths_and_parents() -> None:
    deflator = streams.ScriptedDeflator(streams.per_sample(streams.negative_is_worse))
    run = _run(deflator=deflator)
    with_eligible = [t for t in run.run.report.triggers if t.eligible]
    assert len(deflator.calls) == len(with_eligible)
    for call, handed in zip(
        deflator.calls[-len(run.port.handed) :], run.port.handed, strict=True
    ):
        figures = {each.agent_id: each.deflated for each in call}
        assert all(
            value == figures[agent_id]
            for agent_id, value in handed.items()
            if value is not None
        )


def test_the_report_carries_what_each_trigger_and_birth_did() -> None:
    """With a refusal seeded at the run's first spawn and made good by a redraw."""
    config = _turnover()
    assert config.admission_redraws > 0
    run = _run(refused_spawns=streams.FIRST_SPAWN)
    report = run.run.report

    assert run.port.pushed == sorted(streams.FIRST_SPAWN)
    assert report.admission_refusals == run.state.admission_refusals() > 0
    assert len(report.per_bar) == DECISION_POINTS
    for trigger in report.triggers:
        assert trigger.culls + trigger.retirements + trigger.protected_retirements == (
            trigger.deaths
        )
        assert trigger.below_minimum_age >= 0
        assert trigger.candidates >= 0
        assert trigger.replacement_count == config.replacement_count
    genes = set(run.state.gene_names)
    assert set(report.boundary_firings) == genes
    for birth in report.births:
        assert set(birth.boundary_firings) == genes
        assert birth.mutation_scale is not None
        assert math.isfinite(birth.mutation_scale)
        assert birth.mutation_scale > 0.0
        assert birth.learned_state_disposition == "nothing_carried"
        assert len(birth.parents) == config.parents_per_birth
    assert report.parents_per_birth == config.parents_per_birth
    assert report.inheritance == "baldwinian"


def test_a_birth_lost_past_its_redraws_stops_a_run_at_the_band() -> None:
    """A refusal certain by construction, at the run's first spawn, with no redraw.

    The population is founded at the band's lower bound, so the first trigger
    that kills replaces its dead or leaves the band. Its first birth is
    refused by the real admission path and counted by the real counter, and
    with no redraw configured nothing makes it good.
    """
    config = _turnover(admission_redraws=0)
    windows = streams.priced(count=DECISION_POINTS)
    wired = streams.wire(
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
        refused_spawns=streams.FIRST_SPAWN,
    )
    assert config.admission_redraws == 0
    breached: ecology.steady_state.PopulationBandBreached | None = None
    try:
        wired.loop.run(streams.Watched(windows, wired.state))
    except ecology.steady_state.PopulationBandBreached as error:
        breached = error
    assert wired.port.pushed == sorted(streams.FIRST_SPAWN), (
        "the declared spawn was never asked for, so no refusal was seeded"
    )
    assert wired.state.admission_refusals() >= 1, (
        "the admission path refused nothing, so no birth was lost to a refusal"
    )
    assert breached is not None, (
        "a birth was refused with no redraw to make it good, and the run did "
        "not stop at the band"
    )


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def _admissions(monkeypatch) -> list[tuple[str, dict, dict, object]]:
    seen: list[tuple[str, dict, dict, object]] = []
    original = ecology.steady_state.SteadyState._admitted

    def watching(self, window, birth, offspring, blind, attempt):
        lineage = original(self, window, birth, offspring, blind, attempt)
        seen.append(
            (
                birth.agent_id,
                dict(self.state.learned_state(birth.agent_id)),
                dict(self.state.learned_state(birth.parents[0])),
                lineage,
            )
        )
        return lineage

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_admitted", watching)
    return seen


def test_a_lamarckian_birth_carries_its_parents_whole_learned_state(
    monkeypatch,
) -> None:
    seen = _admissions(monkeypatch)
    _run(inheritance="lamarckian")
    assert seen
    for agent_id, born_with, parent_had, lineage in seen:
        assert born_with == parent_had, agent_id
        assert lineage.learned_state_disposition == "carried"


def test_a_baldwinian_birth_starts_from_its_own_genome(monkeypatch) -> None:
    seen = _admissions(monkeypatch)
    run = _run(inheritance="baldwinian")
    assert seen
    for agent_id, born_with, parent_had, lineage in seen:
        fresh = agent.PopulationState.founded(
            (
                agent.AgentBirth(
                    agent_id=agent_id,
                    genome=_genome_of(run, agent_id),
                    learned_state=MappingProxyType({}),
                    feature_schema_version=s1.FEATURE_SCHEMA_VERSION,
                    model_version=s1.MODEL_VERSION,
                ),
            ),
            learning=LearningConfig(),
        )
        assert born_with == dict(fresh.learned_state(agent_id)), agent_id
        assert born_with != parent_had
        assert lineage.learned_state_disposition == "nothing_carried"


def _genome_of(run: streams.Steady, agent_id: str):
    births = {birth.agent_id: birth for birth in run.run.report.births}
    return births[agent_id].genome


# ---------------------------------------------------------------------------
# The diversity record
# ---------------------------------------------------------------------------
#
# With the offers off, the default, the loop offers nothing:
# ``tests/facts/test_archive_interaction.py::test_niching_off_offers_nothing``
# (EV-E79) carries that case, byte-identity included, and replaces the one that
# stood here (``liq-docs/plans/oracle-v4-archive-interaction-2026-10-02.md`` §7).
# That the record carries across a split boundary is EV-E79's
# ``test_the_record_carries_across_a_split_and_a_run_starts_it_empty``, cited
# rather than repeated. Configuring the offers on is not niching.


def _recorded(diversity_record: str, **keys: object) -> record_trace.Walked:
    return record_trace.walk(diversity_record=diversity_record, **keys)  # ty: ignore[invalid-argument-type]


def test_offering_to_the_record_changes_nothing_the_loop_decides() -> None:
    """With the offers on, every birth, death, survivor and trigger row is the default's."""
    on = _recorded("record_only")
    off = _recorded("none")

    assert on.trace.offers, "nothing was offered, so the comparison was vacuous"
    assert record_trace.decided(on) == record_trace.decided(off)


def test_with_the_offers_off_the_report_is_the_default_and_carries_no_record_field() -> (
    None
):
    walked = record_trace.reference_walk(diversity_record="none")
    report = walked.run.report  # ty: ignore[unresolved-attribute]

    assert walked.trace.offers == []
    assert type(report) is ecology.steady_state.TurnoverReport
    assert all(type(row) is ecology.steady_state.Trigger for row in report.triggers)
    assert record_trace.report_digest(report) == record_trace.OFFER_FREE_REPORT_DIGEST


def test_with_the_offers_on_every_trigger_row_reports_what_was_offered_and_read() -> (
    None
):
    walked = _recorded("record_only")
    report = walked.run.report  # ty: ignore[unresolved-attribute]
    trace = walked.trace
    windows = {window.as_of: window for window in trace.bars}
    census = {bar: count for bar, _, count in trace.occupancy_census}
    living = {bar: who for bar, who, _ in trace.occupancy_census}
    excluded = {id(entry) for entry in walked.population.excluded()}

    assert len(trace.occupancy_census) == len(report.triggers)
    readings = 0
    for row in report.triggers:
        assert row.segment_id == windows[row.as_of].segment_id
        assert row.cumulative_coverage.total == walked.population.archive.total_bins
        assert row.living_occupancy.occupied == census[row.index], row.index
        assert living[row.index] == walked.watched.living[row.index], row.index
        made = [offer for offer in trace.offers if offer.bar == row.index]
        assert row.offers == len(made), row.index
        assert row.offers_kept == sum(offer.kept for offer in made), row.index
        assert row.offers_excluded == sum(
            id(offer.entry) in excluded for offer in made
        ), row.index
        assert row.described + row.undescribed == len(
            walked.watched.living[row.index]
        ), row.index
        assert set(row.raw_beta) <= set(walked.watched.living[row.index])
        assert len(row.raw_beta) == row.described
        assert set(row.population_quantiles) == set(descriptors.BEHAVIOURAL_AXES)
        if row.drift is None:
            continue
        readings += 1
        assert set(row.drift) == set(descriptors.BEHAVIOURAL_AXES)
    assert report.triggers[0].drift is None, "drift was read before a second reading"
    assert readings == len(report.triggers) - 1
    initial = report.initial_coverage
    assert initial.absent is None
    assert initial.trigger is next(row for row in report.triggers if row.offers)


def test_a_descriptor_no_binning_accepts_is_counted_excluded_never_clamped() -> None:
    """A turnover past the mandate's bound reaches the record unaltered and is refused."""
    walked = _recorded("record_only", max_turnover=1.0e-3)
    rows = walked.run.report.triggers  # ty: ignore[unresolved-attribute]
    excluded = {id(entry) for entry in walked.population.excluded()}

    assert excluded, "no offer was excluded, so the exclusion count was untested"
    assert sum(row.offers_excluded for row in rows) == len(excluded)
    for entry in walked.population.excluded():
        assert entry.descriptor.values[descriptors.TURNOVER] > 1.0
    assert any(
        row.descriptor_exclusions.get(descriptors.TURNOVER, {}).get(
            descriptors.Exclusion.OUTSIDE_UNIT_INTERVAL
        )
        for row in rows
    )
    assert any(row.living_occupancy.excluded for row in rows)


def test_initial_coverage_is_a_trigger_s_or_absent_never_both() -> None:
    row = _recorded("record_only").run.report.initial_coverage.trigger  # ty: ignore[unresolved-attribute]
    for trigger, absent in ((row, "a reason"), (None, None)):
        with pytest.raises(ValueError, match="absent"):
            ecology.steady_state.InitialCoverage(trigger=trigger, absent=absent)


@pytest.mark.parametrize("value", ["niched", "on", "", "None", "record-only"])
def test_a_diversity_record_value_that_does_not_exist_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="diversity_record"):
        record_trace.ecology_config(
            run_id="keys", turnover=_turnover(), diversity_record=value
        )


def test_a_record_reset_at_split_boundaries_is_refused() -> None:
    with pytest.raises(ValueError, match="record_carries_across_segments"):
        record_trace.ecology_config(
            run_id="keys",
            turnover=_turnover(),
            record_carries_across_segments=False,
        )


def test_offers_without_the_declaration_their_drift_is_read_under_are_refused() -> None:
    config = record_trace.ecology_config(
        run_id="keys", turnover=_turnover(), diversity_record="record_only"
    )
    loop = _loop(turnover=_turnover())
    with pytest.raises(ValueError, match="diversity_record"):
        ecology.steady_state.SteadyState(
            population=loop.population,
            state=loop.state,
            deflator=NullFitnessDeflator(),
            config=config,
            instrument=streams.INSTRUMENT,
            sizer=NullRiskSizer(),
            simulator=NullExecutionSimulator(),
            opening_equity=1.0,
        )


# ---------------------------------------------------------------------------
# A walk-forward split boundary
# ---------------------------------------------------------------------------


def test_a_split_boundary_triggers_nothing_and_resets_nothing() -> None:
    """Across the boundary an agent keeps its history, age and learned state."""
    histories: dict[int, dict[str, tuple[float, ...]]] = {}
    holder: dict[str, agent.PopulationState] = {}

    def look(index: int) -> None:
        if index in (s1.BREAK_AT - 1, s1.BREAK_AT) and "state" in holder:
            state = holder["state"]
            histories[index] = {
                agent_id: state.history(agent_id) for agent_id in state.agent_ids()
            }

    windows = streams.priced(count=s1.BREAK_AT + 10)
    config = _turnover(trigger_period=7)
    assert s1.BREAK_AT % config.trigger_period != 0
    state_box = agent.PopulationState.founded(
        streams.founders(config.band_lower),
        population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
        learning=LearningConfig(),
    )
    holder["state"] = state_box
    port = ArrayGenomePopulation(
        state=state_box, descriptor_schema_version=streams.DESCRIPTOR_SCHEMA_VERSION
    )
    loop = ecology.steady_state.SteadyState(
        population=port,
        state=state_box,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
        config=EcologyConfig(
            run_id="split",
            population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
            learning=LearningConfig(),
            turnover=config,
        ),
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
    )
    watched = streams.Watched(windows, state_box, look=look)
    result = loop.run(watched)

    assert [crossing.as_of for crossing in result.evaluation.history.crossings] == [
        windows[s1.BREAK_AT].as_of
    ]
    assert not watched.deaths(s1.BREAK_AT)
    assert not watched.births(s1.BREAK_AT)
    before, after = histories[s1.BREAK_AT - 1], histories[s1.BREAK_AT]
    assert before.keys() == after.keys()
    for agent_id, history in before.items():
        assert after[agent_id][: len(history)] == history or (
            len(history) == streams.HISTORY_CAPACITY
            and after[agent_id][:-1] == history[1:]
        ), agent_id
        assert len(after[agent_id]) == min(len(history) + 1, streams.HISTORY_CAPACITY)


# ---------------------------------------------------------------------------
# Construction refusals
# ---------------------------------------------------------------------------


def _loop(
    *,
    turnover,
    deflator=None,
    founding=None,
    population=None,
    state=None,
    history=64,
    learning=True,
):
    learns = LearningConfig() if learning else None
    held = PopulationConfig(history_capacity=history)
    state = (
        agent.PopulationState.founded(
            streams.founders(turnover.band_lower) if founding is None else founding,
            population=held,
            learning=learns,
        )
        if state is None
        else state
    )
    port = (
        ArrayGenomePopulation(state=state, descriptor_schema_version="d")
        if population is None
        else population
    )
    return ecology.steady_state.SteadyState(
        population=port,
        state=state,
        deflator=NullFitnessDeflator() if deflator is None else deflator,
        config=EcologyConfig(
            run_id="refused",
            population=held,
            learning=learns or LearningConfig(),
            turnover=turnover,
        ),
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
    )


def test_a_configuration_that_would_be_blind_by_age_alone_is_refused() -> None:
    """``N_lower − k·⌈m/T⌉ < minimum_candidates`` is refused, derived from the port."""
    config = _turnover()
    floor = ArrayGenomePopulation.minimum_candidates.fget(None)  # ty: ignore[unresolved-attribute]
    unripe = config.replacement_count * math.ceil(
        config.minimum_age / config.trigger_period
    )
    roomy = {"maximum_replaced_fraction": 0.4}
    short = _turnover(
        band_lower=floor + unripe - 1, band_upper=floor + unripe + 4, **roomy
    )
    with pytest.raises(ValueError, match="blind"):
        _loop(turnover=short)
    _loop(
        turnover=_turnover(
            band_lower=floor + unripe, band_upper=floor + unripe + 4, **roomy
        )
    )


def test_a_minimum_age_too_short_for_the_deflator_is_refused() -> None:
    config = _turnover()
    needs_more = streams.ScriptedDeflator(
        streams.per_sample(streams.negative_is_worse),
        minimum_window=config.minimum_age + 1,
    )
    with pytest.raises(ValueError, match="minimum window"):
        _loop(turnover=config, deflator=needs_more)
    short_window = _turnover(fitness_window=1)
    with pytest.raises(ValueError, match="minimum window"):
        _loop(turnover=short_window)


def test_a_fitness_window_longer_than_the_history_kept_is_refused() -> None:
    with pytest.raises(ValueError, match="history"):
        _loop(turnover=_turnover(fitness_window=10), history=8)


def test_a_founding_outside_the_band_is_refused() -> None:
    config = _turnover()
    with pytest.raises(ValueError, match="band"):
        _loop(turnover=config, founding=streams.founders(config.band_upper + 1))


def test_a_port_and_a_store_holding_different_agents_are_refused() -> None:
    config = _turnover()
    other = agent.PopulationState.founded(
        streams.founders(config.band_lower + 1), learning=LearningConfig()
    )
    with pytest.raises(ValueError, match="same living"):
        _loop(
            turnover=config,
            population=ArrayGenomePopulation(
                state=other, descriptor_schema_version="d"
            ),
        )


def test_a_population_that_does_not_learn_is_refused() -> None:
    with pytest.raises(ValueError, match="learn"):
        _loop(turnover=_turnover(), learning=False)


def test_a_loop_needs_a_turnover_configuration() -> None:
    config = _turnover()
    loop = _loop(turnover=config)
    with pytest.raises(ValueError, match="turnover"):
        ecology.steady_state.SteadyState(
            population=loop.population,
            state=loop.state,
            deflator=NullFitnessDeflator(),
            config=EcologyConfig(run_id="none"),
            instrument=streams.INSTRUMENT,
            sizer=NullRiskSizer(),
            simulator=NullExecutionSimulator(),
            opening_equity=1.0,
        )


@pytest.mark.parametrize(
    ("keys", "match"),
    [
        ({"band_lower": 0}, "at least one"),
        ({"replacement_count": 0}, "at least one"),
        ({"trigger_period": 0}, "at least one"),
        ({"replacement_window": 0}, "at least one"),
        ({"fitness_window": 0}, "at least one"),
        ({"parents_per_birth": 0}, "at least one"),
        ({"minimum_age": -1}, "negative"),
        ({"admission_redraws": -1}, "negative"),
        ({"band_lower": 21}, "band"),
        ({"maximum_replaced_fraction": 1.0}, "fraction"),
        ({"maximum_replaced_fraction": 0.0}, "fraction"),
        ({"replacement_count": 14}, "whole population"),
        ({"inheritance": "neither"}, "inheritance"),
    ],
)
def test_configurations_the_records_refuse_are_refused(keys, match) -> None:
    with pytest.raises(ValueError, match=match):
        _turnover(**keys)


def test_no_turnover_key_holds_an_instant() -> None:
    """The trigger is a function of the bar index and configuration only."""
    config = _turnover()
    assert not any(isinstance(value, datetime) for value in vars(config).values()), (
        "a turnover key holds an instant, so the trigger could depend on the clock"
    )


# ---------------------------------------------------------------------------
# Construction refusals with the record's offers on
# ---------------------------------------------------------------------------


def _record_only(population, state, **keys: object):
    """A record-only loop over ``population``; ``keys`` reach the loop's constructor."""
    return ecology.steady_state.SteadyState(
        population=population,
        state=state,
        deflator=NullFitnessDeflator(),
        config=record_trace.ecology_config(
            run_id="refused", turnover=_turnover(), diversity_record="record_only"
        ),
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
        **keys,  # ty: ignore[invalid-argument-type]
    )


def _founded_for_the_record() -> agent.PopulationState:
    return agent.PopulationState.founded(
        streams.founders(_turnover().band_lower),
        population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
        learning=LearningConfig(),
    )


def test_a_record_only_pass_over_a_record_already_holding_an_entry_is_refused() -> None:
    """A run's record starts empty: one holding an entry at start came from elsewhere."""
    state = _founded_for_the_record()
    population = record_trace.declared(state)
    agent_id = population.agent_ids()[0]
    kept = population.record(
        ArchiveEntry(
            agent_id=agent_id,
            genome=population.genome(agent_id),
            descriptor=Descriptor(
                values=MappingProxyType(dict.fromkeys(descriptors.AXES, 0.5)),
                schema_version=descriptors.VOCABULARY,
            ),
            objectives=MappingProxyType({"deflated": 1.0}),
            recorded_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
    )
    assert kept and population.recorded(), "the record kept nothing, so it is empty"
    with pytest.raises(ValueError, match="empty"):
        _record_only(population, state, descriptor_declaration=population.descriptors)


def test_record_only_over_a_population_not_described_in_the_declared_vocabulary_is_refused() -> (
    None
):
    """The report's descriptors are the population's own; one described otherwise has none."""
    state = _founded_for_the_record()
    population = ArrayGenomePopulation(
        state=state, descriptor_schema_version=streams.DESCRIPTOR_SCHEMA_VERSION
    )
    assert population.descriptor(population.agent_ids()[0]).schema_version != (
        descriptors.VOCABULARY
    )
    with pytest.raises(ValueError, match="vocabulary"):
        _record_only(
            population, state, descriptor_declaration=record_trace.declaration()
        )


def test_a_declaration_handed_to_the_loop_that_is_not_the_populations_is_refused() -> (
    None
):
    """The loop reads the population's own declaration; a second copy cannot disagree."""
    state = _founded_for_the_record()
    population = record_trace.declared(state)
    other = record_trace.declaration(max_turnover=record_trace.MAX_TURNOVER / 2)
    assert other != population.descriptor_declaration
    with pytest.raises(ValueError, match="population's own"):
        _record_only(population, state, descriptor_declaration=other)
    loop = _record_only(population, state, descriptor_declaration=None)
    assert loop._declaration is population.descriptor_declaration


# ---------------------------------------------------------------------------
# Allocation in the loop
# ---------------------------------------------------------------------------
#
# With an allocation configured the loop updates the fixed-share weights once
# per decision point, after settlement and the trigger's births and deaths and
# before the population decides, and forms the aggregate book after the
# decision; the book's target is acted on at the next bar. With none configured
# the loop is the one it was, to the byte of its turnover report.

#: A test-local band; the run's value is the composition's to supply.
_BAND = 0.0


def _allocation_config():
    from liq.evolution.ecology.config import AllocationConfig
    from tests.support import weight_oracle

    record = weight_oracle.read_record()
    return AllocationConfig.from_tables(
        {
            "contract_sha256": record.sha256,
            "eta": record.contract["eta"],
            "share_rate": record.contract["share_rate"],
        },
        band=_BAND,
    )


class _Lines:
    """A line writer that keeps every line, in order, in a shared list."""

    def __init__(self, lines: list, bound: dict | None = None) -> None:
        self.lines = lines
        self.bound = bound or {}

    def bind(self, **values: str) -> _Lines:
        return _Lines(self.lines, {**self.bound, **values})

    def info(self, event: str, /) -> None:
        self.lines.append(("log", self.bound.get("stage"), event, self.bound))


class _Acting(NullExecutionSimulator):
    """The null execution model, keeping every target it acts on and the bar it acts at."""

    def __init__(self, acted: list) -> None:
        super().__init__()
        object.__setattr__(self, "_acted", acted)

    def execute(self, target, bar, account):
        self._acted.append((target, bar.period_end))  # ty: ignore[unresolved-attribute]
        return super().execute(target, bar, account)


def _allocated(
    *,
    allocation: object = None,
    risk: object = None,
    windows=None,
    log=None,
    simulator=None,
    events: list | None = None,
):
    """A loop over the declared stream with an allocation configured, walked once."""
    from liq.evolution.ecology.adapters import NullPortfolioRisk
    from liq.evolution.ecology.logctx import SILENT, RunLog

    config = _turnover()
    learns = LearningConfig()
    held = PopulationConfig(history_capacity=streams.HISTORY_CAPACITY)
    state = agent.PopulationState.founded(
        streams.founders(config.band_lower), population=held, learning=learns
    )
    port = ArrayGenomePopulation(
        state=state, descriptor_schema_version=streams.DESCRIPTOR_SCHEMA_VERSION
    )
    loop = ecology.steady_state.SteadyState(
        population=port,
        state=state,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
        config=EcologyConfig(
            run_id="allocated",
            population=held,
            learning=learns,
            turnover=config,
            allocation=_allocation_config() if allocation is None else allocation,  # ty: ignore[invalid-argument-type]
        ),
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator() if simulator is None else simulator,
        opening_equity=1.0,
        log=SILENT if log is None else RunLog(run_id="allocated", writer=log),
        portfolio_risk=NullPortfolioRisk() if risk is None else risk,  # ty: ignore[unknown-argument]
    )
    if events is not None:
        turn_over, observe, step = loop._turn_over, state.observe, state.step

        def turning(window):
            events.append(("turnover", window.as_of))
            turn_over(window)

        def observing(as_of, outcomes):
            events.append(("settle", as_of))
            return observe(as_of, outcomes)

        def stepping(window, instrument):
            events.append(("decide", window.as_of))
            return step(window, instrument)

        loop._turn_over = turning  # ty: ignore[invalid-assignment]
        state.observe = observing  # ty: ignore[invalid-assignment]
        state.step = stepping  # ty: ignore[invalid-assignment]
    stream = streams.priced(count=DECISION_POINTS) if windows is None else windows
    watched = streams.Watched(stream, state)
    return loop, watched, stream, loop.run(watched)


def test_an_allocated_loop_updates_after_settlement_and_turnover_and_before_deciding() -> (
    None
):
    """At every decision point: settle, turn over, allocate, decide, then form the book."""
    events: list = []
    _, _, stream, _ = _allocated(log=_Lines(events), events=events)
    allocations = [
        each for each in events if each[0] == "log" and each[1] == "allocate"
    ]
    assert allocations, "no allocation event was recorded"
    for window in stream:
        stamp = window.as_of.isoformat()
        at = [
            index
            for index, each in enumerate(events)
            if (each[0] != "log" and each[1] == window.as_of)
            or (each[0] == "log" and each[3].get("bar_ts") == stamp)
        ]
        kinds = [
            (
                events[i][0],
                events[i][1],
                events[i][2] if events[i][0] == "log" else None,
            )
            for i in at
        ]
        weighted = [
            k
            for k in kinds
            if k[:2] == ("log", "allocate") and k[2] == "decision_point_shown"
        ]
        booked = [
            k for k in kinds if k[:2] == ("log", "allocate") and k[2] == "intent_sized"
        ]
        assert len(weighted) == 1 and len(booked) == 1, (window.as_of, kinds)
        order = [
            k[0] if k[0] != "log" else k[2]
            for k in kinds
            if k[0] != "log" or k[1] == "allocate"
        ]
        expected = ["turnover", "decision_point_shown", "decide", "intent_sized"]
        if order[0] == "settle":
            expected = ["settle", *expected]
        assert order == expected, (window.as_of, order)


def test_the_books_target_formed_at_a_point_is_acted_on_at_the_next_bar() -> None:
    """Every target the book is sent was formed one decision point earlier."""
    acted: list = []
    loop, _, stream, run = _allocated(simulator=_Acting(acted))
    rows = run.allocation.rows  # ty: ignore[unresolved-attribute]
    assert [row.as_of for row in rows] == [window.as_of for window in stream]
    book = [
        (target, at)
        for target, at in acted
        if target.agent_id == ecology.BOOK_ACCOUNT_ID
    ]
    sent = {
        row.decision.target.as_of: row.decision.target
        for row in rows
        if row.decision.target is not None
    }
    assert book, "the book was never acted on, so the check is vacuous"
    instants = [window.as_of for window in stream]
    for target, at in book:
        index = instants.index(target.as_of)
        assert stream[index + 1].bars[streams.INSTRUMENT].period_end == at
        assert sent[target.as_of] == target
    assert len(book) == sum(1 for row in rows[:-1] if row.decision.target is not None)
    assert loop.allocation is not None


def test_births_and_deaths_move_weight_exactly_as_the_record_says() -> None:
    """Rebuilt from the record's text over the observed living: newborns at s/P exactly."""
    from tests.ecology.test_population_accounts import _rebuilt
    from tests.support import weight_oracle as oracle

    _, watched, stream, run = _allocated()
    living_at = [
        tuple(sorted(watched.living[index + 1])) for index in range(len(stream))
    ]
    rebuilt = _rebuilt(stream, run.evaluation, living_at)
    warmup = int(oracle.read_record().contract["scale_warmup"])  # ty: ignore[invalid-argument-type]
    after_warmup = [
        index
        for index in range(warmup, len(stream))
        if watched.births(index) and watched.deaths(index)
    ]
    assert after_warmup, (
        "no trigger fell after the warm-up, so no loss step met a turnover"
    )
    found: list[str] = []
    for index, (state, (expected, newborns, share)) in enumerate(
        zip(run.allocation.weights, rebuilt, strict=True)  # ty: ignore[unresolved-attribute]
    ):
        assert newborns == watched.births(index) or index == 0, index
        implemented = dict(zip(state.agent_ids, state.weights.tolist(), strict=True))
        found += [
            f"@{index} {each}"
            for each in oracle.mismatches(
                expected,
                implemented,
                newborns=newborns if index else frozenset(),
                share_rate=share,
            )
        ]
    assert not found, "\n".join(found[:10])


def test_weights_and_ledger_are_plain_values_in_the_loops_state() -> None:
    """Frozen values the loop holds, with no random state; one seed, one result."""
    import numpy as np

    loop, _, _, run = _allocated()
    held = loop.allocation
    assert held.weights is run.allocation.weights[-1]  # ty: ignore[unresolved-attribute]
    for value in (*vars(held).values(), *vars(held.ledger).values()):
        assert not isinstance(value, np.random.Generator | np.random.RandomState)
    _, _, _, again = _allocated()
    assert [w.weights.tobytes() for w in again.allocation.weights] == [  # ty: ignore[unresolved-attribute]
        w.weights.tobytes()
        for w in run.allocation.weights  # ty: ignore[unresolved-attribute]
    ]
    assert again.allocation.rows == run.allocation.rows  # ty: ignore[unresolved-attribute]


def test_allocation_changes_nothing_turnover_decides() -> None:
    """Turnover never reads the weights: the report is the same with or without them."""
    _, _, _, on = _allocated()
    off = _run()
    assert repr(on.report) == repr(off.run.report)  # ty: ignore[unresolved-attribute]


def _with_allocation(monkeypatch) -> None:
    """Route :mod:`record_trace`'s walks through a loop with an allocation configured."""
    import functools

    from liq.evolution.ecology.adapters import NullPortfolioRisk

    configured = _allocation_config()
    original = record_trace.ecology_config
    monkeypatch.setattr(
        record_trace,
        "ecology_config",
        lambda **keys: original(**keys, allocation=configured),
    )
    monkeypatch.setattr(
        ecology.steady_state,
        "SteadyState",
        functools.partial(
            ecology.steady_state.SteadyState, portfolio_risk=NullPortfolioRisk()
        ),
    )


def test_with_no_allocation_the_turnover_report_is_the_frozen_offer_free_one(
    monkeypatch,
) -> None:
    """The differential: unconfigured, the report is the frozen offer-free one, to the byte; configured, too."""
    off = record_trace.reference_walk()
    assert off.run.allocation is None  # ty: ignore[unresolved-attribute]
    assert off.run.evaluation.book is None  # ty: ignore[unresolved-attribute]
    assert (
        record_trace.report_digest(off.run.report)  # ty: ignore[unresolved-attribute]
        == record_trace.OFFER_FREE_REPORT_DIGEST
    )
    _with_allocation(monkeypatch)
    on = record_trace.reference_walk()
    assert on.run.allocation is not None  # ty: ignore[unresolved-attribute]
    assert (
        record_trace.report_digest(on.run.report)  # ty: ignore[unresolved-attribute]
        == record_trace.OFFER_FREE_REPORT_DIGEST
    )


def test_the_allocation_reads_no_record_contents(monkeypatch) -> None:
    """Discarding every offer changes no weight and no ledger row."""
    _with_allocation(monkeypatch)
    kept = record_trace.walk(diversity_record="record_only")
    discarded = record_trace.walk(diversity_record="record_only", discard_offers=True)
    assert kept.trace.offers, "nothing was offered, so the comparison was vacuous"
    assert kept.population.recorded() != discarded.population.recorded()
    one, other = kept.run.allocation, discarded.run.allocation  # ty: ignore[unresolved-attribute]
    assert [w.weights.tobytes() for w in one.weights] == [
        w.weights.tobytes() for w in other.weights
    ]
    assert one.rows == other.rows


def test_an_allocation_without_a_portfolio_risk_port_is_refused() -> None:
    """The book's bounds are the port's: an allocation with nothing to bound it is refused."""
    state = _founded_for_the_record()
    with pytest.raises(ValueError, match="portfolio-risk"):
        ecology.steady_state.SteadyState(
            population=ArrayGenomePopulation(
                state=state, descriptor_schema_version="d"
            ),
            state=state,
            deflator=NullFitnessDeflator(),
            config=record_trace.ecology_config(
                run_id="refused", turnover=_turnover(), allocation=_allocation_config()
            ),
            instrument=streams.INSTRUMENT,
            sizer=NullRiskSizer(),
            simulator=NullExecutionSimulator(),
            opening_equity=1.0,
        )


def test_a_portfolio_risk_port_without_an_allocation_is_refused() -> None:
    """A port handed to a loop with no book would be read by nothing."""
    from liq.evolution.ecology.adapters import NullPortfolioRisk

    state = _founded_for_the_record()
    with pytest.raises(ValueError, match="no allocation"):
        ecology.steady_state.SteadyState(
            population=ArrayGenomePopulation(
                state=state, descriptor_schema_version="d"
            ),
            state=state,
            deflator=NullFitnessDeflator(),
            config=record_trace.ecology_config(run_id="refused", turnover=_turnover()),
            instrument=streams.INSTRUMENT,
            sizer=NullRiskSizer(),
            simulator=NullExecutionSimulator(),
            opening_equity=1.0,
            portfolio_risk=NullPortfolioRisk(),  # ty: ignore[unknown-argument]
        )


def test_record_only_without_a_turnover_configuration_is_refused() -> None:
    """Offers are made at triggers; a pass with no turnover has none and offers nothing."""
    with pytest.raises(ValueError, match="turnover"):
        EcologyConfig(run_id="refused", diversity_record="record_only")
