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
* no birth, death, cull or retirement offers an entry to the diversity record;
* a walk-forward split boundary triggers nothing and resets nothing;
* the configurations the records refuse are refused at construction.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import math
from datetime import datetime
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent
from liq.evolution.ecology.adapters import (
    ArrayGenomePopulation,
    NullExecutionSimulator,
    NullFitnessDeflator,
    NullRiskSizer,
)
from liq.evolution.ecology.config import EcologyConfig, LearningConfig, PopulationConfig
from liq.evolution.ecology.types import DeflatedFitness
from tests.support import level_shift_stream as s1
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


def test_the_loop_offers_nothing_to_the_diversity_record() -> None:
    """No birth, death, cull or retirement offers an entry to the record.

    ``liq-docs/plans/oracle-v3-archive-offers-2026-10-02.md`` §1.5: the real
    adapter's ``record`` is intercepted at the instance, and the walk must
    both turn the population over and leave the record holding nothing.
    """
    config = _turnover()
    wired = streams.wire(
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=streams.ScriptedDeflator(streams.lowest_is_worse),
    )
    population = wired.port.inner
    assert isinstance(population, ArrayGenomePopulation)
    offers: list[object] = []
    record = population.record

    def intercepted(entry):
        offers.append(entry)
        return record(entry)

    population.record = intercepted  # ty: ignore[invalid-assignment]
    run = wired.loop.run(  # ty: ignore[unresolved-attribute]
        streams.Watched(streams.priced(count=DECISION_POINTS), wired.state)
    )

    assert run.report.births, "nobody was born, so no birth could have offered"
    assert {death.kind for death in run.report.deaths} >= {"cull", "retirement"}, (
        "a kind of death this stream yields never happened, so it was never tested"
    )
    assert offers == [], "the loop offered entries to the population's diversity record"
    assert population.recorded() == ()


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
