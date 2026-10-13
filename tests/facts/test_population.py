"""Evidence about who dies, who is born, and how many, in a steady-state population.

Five claims, each observed from outside the loop that turns the population over:

* ``MIN_AGE_ENFORCED`` — no agent younger than the configured minimum age dies,
  by cull or by retirement, and a trigger with too few eligible agents has
  fewer deaths rather than taking a young one.
* ``DEATH_COUNT_FROM_CONFIGURATION`` — two runs with identical configuration
  and seeds over streams whose outcomes differ have the same deaths, and the
  same births, at every decision point; the data decides who, never how many.
* ``SIZE_WITHIN_BAND`` — the living count lies within the configured band at
  founding and after every decision point, including when the admission path
  really does refuse a birth.
* ``REPLACEMENT_WITHIN_FRACTION`` — no run of the configured window's length,
  whatever bar it starts at, holds more deaths than the configured fraction of
  those living at its start.
* ``ONE_EVOLUTION_ENGINE``'s port clause — every birth's parents are a set the
  population port's selection returned, and its genome is exactly the offspring
  the port's spawn returned.

**Observation, not self-report.** Living counts, births and deaths are set
differences of ``agent_ids()`` read between decision points by the bar source
(``tests/support/turnover_stream.py``), never the loop's turnover report. Where
the report is checked it is checked *against* the observation.

**The rule these encode** is ``liq-docs/plans/oracle-v3-indistinguishable-cull-2026-09-30.md``
§2.2–§2.3 and §4.3, with the reproduce side of
``liq-docs/plans/oracle-v3-unassessed-fitness-2026-10-01.md`` §2.2–§2.3, and the
band and fraction semantics of ``liq-docs/plans/oracle-v3-fr4-facts-2026-09-30.md``
§3. Every configuration bound below is derived from the configuration's own keys
rather than written as a number.

**Seeded faults.** Each claim carries the faults the records name, injected by
replacing one piece of the loop and required to turn the claim's own oracle red,
so none of these checks can pass by being unable to fail.

Nothing here is market data: every stream is declared arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from liq.evolution import ecology
from liq.evolution.ecology.adapters import ArrayGenomePopulation
from tests.support import turnover_stream as streams

#: How many decision points the runs here walk. Short enough to be quick, long
#: enough for many triggers after the warm-up.
DECISION_POINTS = 120


def _turnover(**keys: object):
    """A turnover configuration; every key a check does not name is declared here."""
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
        "seed": 11,
    }
    declared.update(keys)
    return ecology.steady_state.TurnoverConfig(**declared)


def _ages_at_death(watched: streams.Watched) -> dict[str, int]:
    born = watched.born_at()
    return {
        agent_id: index - born[agent_id]
        for agent_id, index in watched.died_at().items()
    }


# ---------------------------------------------------------------------------
# MIN_AGE_ENFORCED (EV-E12)
# ---------------------------------------------------------------------------


def _min_age_config():
    """Trigger every bar, so agents exactly at, and a bar short of, the minimum age exist."""
    return _turnover(
        band_lower=14,
        band_upper=20,
        replacement_count=1,
        trigger_period=1,
        minimum_age=4,
        replacement_window=4,
        maximum_replaced_fraction=0.3,
    )


def _exactly_at_the_minimum_is_worse(config) -> streams.ScriptedDeflator:
    """Worse for a window exactly the minimum age long, lowest figure for the shortest.

    Every window handed in belongs to an agent the loop judged old enough. A
    loop that handed in a younger one would see it judged worse still, and
    culled first.
    """
    return streams.ScriptedDeflator(
        streams.per_sample(
            lambda sample: (
                float(len(sample.outcomes)),
                "worse"
                if len(sample.outcomes) <= config.minimum_age
                else "indistinguishable",
            )
        )
    )


def _min_age_run() -> tuple[object, streams.Steady]:
    config = _min_age_config()
    run = streams.steady(
        windows=streams.priced(count=DECISION_POINTS),
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=_exactly_at_the_minimum_is_worse(config),
    )
    return config, run


def _minimum_age_held(config, run: streams.Steady) -> None:
    """The oracle: nobody below the minimum age died; warm-up triggers took nobody."""
    ages = _ages_at_death(run.watched)
    assert ages, "nobody died, so the minimum age was never put to the test"
    young = {
        agent_id: age for agent_id, age in ages.items() if age < config.minimum_age
    }
    assert not young, f"agents died below the minimum age {config.minimum_age}: {young}"
    for index in range(config.minimum_age):
        assert not run.watched.deaths(index), (
            f"decision point {index} is in the warm-up — every founder is younger "
            "than the minimum age — and yet somebody died there"
        )


def test_min_age_blocks_cull() -> None:
    """No agent younger than the minimum age dies, by cull or by retirement."""
    config, run = _min_age_run()
    _minimum_age_held(config, run)

    ages = _ages_at_death(run.watched)
    assert config.minimum_age in ages.values(), (
        "nobody died exactly at the minimum age, so the boundary's inclusive side "
        "was never exercised"
    )
    born = run.watched.born_at()
    a_bar_short = [
        agent_id
        for index in range(config.minimum_age, len(run.watched.living) - 1)
        for agent_id in run.watched.living[index]
        if index - born[agent_id] == config.minimum_age - 1
    ]
    assert a_bar_short, "no agent was ever a bar short of the minimum age at a trigger"
    kinds = {death.kind for death in run.run.report.deaths}
    assert {"cull", "retirement"} <= kinds, (
        f"only {kinds} occurred, so one of the two paths a death takes was never "
        "put to the test"
    )


def test_min_age_checked_on_the_cull_path_only_turns_the_check_red(monkeypatch) -> None:
    """Seeded fault: retirements drawn from every living agent, young ones included."""

    def cull_path_only(living, *, replacement_count, minimum_age):
        eligible = [each for each in living if each.age >= minimum_age]
        culls = sorted(
            (
                each
                for each in eligible
                if each.fitness is not None and each.fitness.standing == "worse"
            ),
            key=lambda each: (each.fitness.deflated, each.born_index, each.admitted),
        )[:replacement_count]
        rest = sorted(
            (each for each in living if each not in culls),
            key=lambda each: (each.born_index, each.admitted),
        )[: replacement_count - len(culls)]
        return tuple((each, "cull") for each in culls) + tuple(
            (each, "retirement") for each in rest
        )

    monkeypatch.setattr(ecology.steady_state, "choose_deaths", cull_path_only)
    config, run = _min_age_run()
    with pytest.raises(AssertionError, match="minimum age|warm-up"):
        _minimum_age_held(config, run)


# ---------------------------------------------------------------------------
# DEATH_COUNT_FROM_CONFIGURATION
# ---------------------------------------------------------------------------


def _declared_verdict(sample):
    """Unassessed where the window never traded; otherwise worse where it lost on balance.

    A window that never traded has no dispersion, the class of agent the
    unassessed-fitness ruling expects most of (its §7). Which agents trade at
    a bar differs between S1 and S3, so who is assessed, who is worse and
    whether a trigger is blind all differ between the two runs.
    """
    if not any(sample.outcomes):
        return None
    return streams.negative_is_worse(sample)


#: The population port's floor, read from the adapter rather than restated: the
#: fewest agents carrying a figure its selection can rank.
SELECTION_FLOOR: int = ArrayGenomePopulation.minimum_candidates.fget(None)  # ty: ignore[unresolved-attribute]

#: The DEATH_COUNT runs' band, by a rule fixed before it was run: a lower bound
#: of three times the port's floor, and the band six wide.
DEATH_COUNT_BAND = (3 * SELECTION_FLOOR, 3 * SELECTION_FLOOR + 6)


def _death_count_runs() -> tuple[object, list[streams.Steady]]:
    """S1 and S3 under one configuration, large enough that blindness is rare."""
    config = _turnover(band_lower=DEATH_COUNT_BAND[0], band_upper=DEATH_COUNT_BAND[1])
    runs = [
        streams.steady(
            windows=streams.priced(sign=sign, count=DECISION_POINTS),
            founding=streams.founders(config.band_lower),
            turnover=config,
            deflator=streams.ScriptedDeflator(streams.per_sample(_declared_verdict)),
        )
        for sign in (1, -1)
    ]
    return config, runs


def _counts(run: streams.Steady) -> tuple[list[int], list[int]]:
    bars = range(len(run.watched.living) - 1)
    return (
        [len(run.watched.deaths(index)) for index in bars],
        [len(run.watched.births(index)) for index in bars],
    )


def _counts_identical(runs: list[streams.Steady]) -> None:
    """The oracle: deaths and births identical at every decision point across the runs."""
    (deaths_up, births_up), (deaths_down, births_down) = (_counts(run) for run in runs)
    assert sum(deaths_up) > 0, "nobody died, so the count was never put to the test"
    assert deaths_up == deaths_down, (
        "the number of deaths at some decision point depended on the stream: "
        f"{[(i, a, b) for i, (a, b) in enumerate(zip(deaths_up, deaths_down, strict=True)) if a != b]}"
    )
    assert births_up == births_down, (
        "the number of births at some decision point depended on the stream: "
        f"{[(i, a, b) for i, (a, b) in enumerate(zip(births_up, births_down, strict=True)) if a != b]}"
    )


def test_death_count_set_by_configuration_not_data() -> None:
    """Identical configuration and seeds over S1 and S3: the same counts, different dead."""
    config, runs = _death_count_runs()
    _counts_identical(runs)

    culled = [
        {death.agent_id for death in run.run.report.deaths if death.kind == "cull"}
        for run in runs
    ]
    assert culled[0] != culled[1], (
        "the two streams culled the same agents, so nothing shows the data "
        "deciding who dies"
    )
    worse_present = [
        [trigger.worse > 0 for trigger in run.run.report.triggers] for run in runs
    ]
    assert worse_present[0] != worse_present[1], (
        "no trigger had a worse agent in one run and none in the other, so a loop "
        "that skipped retirements when nobody is worse could not be told apart"
    )
    blind = [[trigger.blind for trigger in run.run.report.triggers] for run in runs]
    for each in blind:
        assert 4 * sum(each) < len(each), (
            f"{sum(each)} of {len(each)} triggers were blind, not fewer than a "
            "quarter, so blindness is not rare in this fixture"
        )
    assert blind[0] != blind[1], (
        "no trigger was blind in one run and not in the other, so a loop that "
        "skipped births when too few are assessed could not be told apart"
    )
    del config


def test_skipping_retirements_when_nobody_is_worse_turns_the_check_red(
    monkeypatch,
) -> None:
    """Seeded fault: a trigger with no worse agent kills nobody."""
    real = ecology.steady_state.choose_deaths

    def only_on_evidence(living, *, replacement_count, minimum_age):
        chosen = real(
            living, replacement_count=replacement_count, minimum_age=minimum_age
        )
        return tuple((each, kind) for each, kind in chosen if kind == "cull")

    monkeypatch.setattr(ecology.steady_state, "choose_deaths", only_on_evidence)
    _, runs = _death_count_runs()
    with pytest.raises(AssertionError, match="number of deaths"):
        _counts_identical(runs)


def test_a_trigger_fired_by_a_worse_verdict_turns_the_check_red(monkeypatch) -> None:
    """Seeded fault: a decision point off the schedule fires because someone is worse."""
    real = ecology.steady_state.choose_deaths
    on_schedule = ecology.steady_state.SteadyState._fires
    current = {"scheduled": True}

    def fires_every_bar(self, index):
        current["scheduled"] = on_schedule(self, index)
        return index > 0

    def deaths_when_scheduled_or_worse(living, *, replacement_count, minimum_age):
        chosen = real(
            living, replacement_count=replacement_count, minimum_age=minimum_age
        )
        if current["scheduled"] or any(kind == "cull" for _, kind in chosen):
            return chosen
        return ()

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_fires", fires_every_bar)
    monkeypatch.setattr(
        ecology.steady_state, "choose_deaths", deaths_when_scheduled_or_worse
    )
    _, runs = _death_count_runs()
    with pytest.raises(AssertionError, match="number of (deaths|births)"):
        _counts_identical(runs)


def _first_differing_blindness(runs: list[streams.Steady]) -> int:
    """The decision point of the first trigger blind in one run and not the other."""
    pairs = zip(runs[0].run.report.triggers, runs[1].run.report.triggers, strict=True)
    for up, down in pairs:
        assert up.index == down.index
        if up.blind != down.blind:
            return up.index
    raise AssertionError("no trigger's blindness differs between the clean runs")


def test_skipping_births_when_too_few_are_assessed_turns_the_check_red(
    monkeypatch,
) -> None:
    """Seeded fault: a blind trigger kills but breeds nobody.

    The fault acts at every blind trigger up to and including the first trigger
    at which the clean pair's blindness differs — the first point at which the
    oracle can see it — computed here from a clean pair under the same
    configuration. The bound is that rule, not a count chosen by trial.
    """
    _, clean = _death_count_runs()
    last = _first_differing_blindness(clean)
    real = ecology.steady_state.SteadyState._parents
    fired: list[tuple[object, int]] = []

    def none_when_blind(self, trigger, survivors, by_id, births):
        chunks, blind = real(self, trigger, survivors, by_id, births)
        if blind and trigger <= last:
            fired.append((self, trigger))
            return (), blind
        return chunks, blind

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_parents", none_when_blind)
    monkeypatch.setattr(
        ecology.steady_state.SteadyState, "_hold_the_band", lambda self, window: None
    )
    _, runs = _death_count_runs()
    blind_at_last = [
        run
        for run in runs
        if any(t.index == last and t.blind for t in run.run.report.triggers)
    ]
    assert len(blind_at_last) == 1, (
        f"the faulted runs are not blind in exactly one run at decision point {last}"
    )
    assert (blind_at_last[0].loop, last) in fired, (
        f"the fault did not fire at decision point {last} in the run blind there"
    )
    with pytest.raises(AssertionError, match="number of births"):
        _counts_identical(runs)


def test_a_configuration_whose_eligible_pool_can_fall_short_is_refused() -> None:
    """``N_lower < k·⌈m/T⌉`` is refused at construction, derived from the keys."""
    config = _turnover()
    floor = config.replacement_count * math.ceil(
        config.minimum_age / config.trigger_period
    )
    with pytest.raises(ValueError, match="eligible"):
        _turnover(band_lower=floor - 1, band_upper=config.band_upper)


# ---------------------------------------------------------------------------
# SIZE_WITHIN_BAND (EV-E75)
# ---------------------------------------------------------------------------


def _band_run() -> tuple[object, streams.Steady]:
    """Founded at the lower bound, with a refusal seeded at the run's first spawn."""
    config = _turnover()
    run = streams.steady(
        windows=streams.priced(count=DECISION_POINTS),
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
        refused_spawns=streams.FIRST_SPAWN,
    )
    return config, run


def _band_held(config, run: streams.Steady) -> None:
    """The oracle: every observed living count lies within the band."""
    counts = [len(living) for living in run.watched.living]
    outside = [
        (index - 1, count)
        for index, count in enumerate(counts)
        if not config.band_lower <= count <= config.band_upper
    ]
    assert not outside, (
        f"the living count left the band [{config.band_lower}, {config.band_upper}] "
        f"at (decision point, count) {outside[:5]}"
    )


def test_living_count_stays_within_band() -> None:
    """Founded at the band's lower bound, with real refusals, the count never leaves it."""
    config, run = _band_run()
    assert len(run.watched.living[0]) == config.band_lower
    _band_held(config, run)

    refused = run.state.admission_refusals()
    assert refused > 0, (
        "the admission path refused no birth, so the band was never tested "
        "against a refusal that had to be made good"
    )
    assert run.run.report.admission_refusals == refused
    assert sum(trigger.admission_refusals for trigger in run.run.report.triggers) == (
        refused
    )
    reported = [
        len(death_set)
        for death_set in (run.watched.deaths(i) for i in range(DECISION_POINTS))
    ]
    assert [bar.deaths for bar in run.run.report.per_bar] == reported, (
        "the turnover report's deaths per bar differ from the deaths observed"
    )


def test_a_refusal_not_made_good_turns_the_band_check_red(monkeypatch) -> None:
    """Seeded fault: a refused birth is dropped, and the loop carries on."""

    def admit_once(self, window, ordinal, chunks, blind, trigger):
        return self._admit_attempt(window, ordinal, chunks[ordinal][0], blind, 0)

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_born", admit_once)
    monkeypatch.setattr(
        ecology.steady_state.SteadyState, "_hold_the_band", lambda self, window: None
    )
    config, run = _band_run()
    with pytest.raises(AssertionError, match="left the band"):
        _band_held(config, run)


# ---------------------------------------------------------------------------
# REPLACEMENT_WITHIN_FRACTION (EV-E76)
# ---------------------------------------------------------------------------


def _fraction_config(**keys: object):
    """A window the trigger period does not divide, so aligned and sliding counts differ."""
    declared: dict[str, object] = {
        "band_lower": 12,
        "band_upper": 16,
        "replacement_count": 1,
        "trigger_period": 4,
        "minimum_age": 8,
        "replacement_window": 10,
        "maximum_replaced_fraction": 0.25,
    }
    declared.update(keys)
    return _turnover(**declared)


def _fraction_run(config) -> streams.Steady:
    return streams.steady(
        windows=streams.priced(count=DECISION_POINTS),
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=streams.ScriptedDeflator(streams.lowest_is_worse),
    )


def _fraction_held(config, run: streams.Steady) -> int:
    """The oracle: no span of the window's length holds more than its fraction."""
    over = [
        (start, deaths, living)
        for start, deaths, living in streams.spans(
            run.watched, config.replacement_window
        )
        if deaths > config.maximum_replaced_fraction * living
    ]
    assert not over, (
        f"spans of {config.replacement_window} bars replaced more than "
        f"{config.maximum_replaced_fraction} of those living at their start: "
        f"(start, deaths, living) {over[:5]}"
    )
    return max(
        deaths for _, deaths, _ in streams.spans(run.watched, config.replacement_window)
    )


def test_no_span_replaces_more_than_its_fraction() -> None:
    """Every span, whatever bar it starts at, holds at most its fraction of deaths."""
    config = _fraction_config()
    run = _fraction_run(config)
    most = _fraction_held(config, run)

    allowed = math.floor(config.maximum_replaced_fraction * config.band_lower)
    assert most == allowed, (
        f"the busiest span held {most} deaths against {allowed} allowed, so the "
        "check never came up against its bound"
    )
    straddling = [
        start
        for start, deaths, _ in streams.spans(run.watched, config.replacement_window)
        if deaths == most and start % config.replacement_window
    ]
    assert straddling, (
        "no span at the bound straddles an aligned window's edge, so this run "
        "cannot show that spans starting at every bar are counted"
    )


def test_a_configuration_whose_window_could_overflow_is_refused() -> None:
    """A configuration whose busiest span could exceed its fraction is refused."""
    loose = {"maximum_replaced_fraction": 0.2}
    with pytest.raises(ValueError, match="fraction"):
        _fraction_config(**loose)


def test_an_aligned_window_counter_turns_the_fraction_check_red(monkeypatch) -> None:
    """Seeded fault: the bound counted over aligned windows admits an overflowing span."""
    monkeypatch.setattr(
        ecology.config,
        "most_deaths_in_any_span",
        lambda replacement_count, trigger_period, window: (
            replacement_count * (window // trigger_period)
        ),
    )
    config = _fraction_config(maximum_replaced_fraction=0.2)
    run = _fraction_run(config)
    with pytest.raises(AssertionError, match="replaced more than"):
        _fraction_held(config, run)


def test_a_bound_on_culls_alone_turns_the_fraction_check_red(monkeypatch) -> None:
    """Seeded fault: retirements fill the count as if no cull had been made."""
    real = ecology.steady_state.choose_deaths

    def culls_bounded_alone(living, *, replacement_count, minimum_age):
        chosen = real(
            living, replacement_count=replacement_count, minimum_age=minimum_age
        )
        culls = tuple(each for each in chosen if each[1] == "cull")
        taken = {each.agent_id for each, _ in culls}
        retired = sorted(
            (
                each
                for each in living
                if each.age >= minimum_age and each.agent_id not in taken
            ),
            key=lambda each: (each.born_index, each.admitted),
        )[:replacement_count]
        return culls + tuple((each, "retirement") for each in retired)

    monkeypatch.setattr(ecology.steady_state, "choose_deaths", culls_bounded_alone)
    monkeypatch.setattr(
        ecology.steady_state.SteadyState, "_hold_the_band", lambda self, window: None
    )
    config = _fraction_config()
    run = _fraction_run(config)
    with pytest.raises(AssertionError, match="replaced more than"):
        _fraction_held(config, run)


# ---------------------------------------------------------------------------
# ONE_EVOLUTION_ENGINE — the port clause (EV-E73 node)
# ---------------------------------------------------------------------------


def _births_traced_to_the_port(run, genomes, parents) -> None:
    """The oracle: every birth's genome is a spawn return, its parents a selection's."""
    assert genomes, "nobody was born, so the port was never traced"
    for agent_id, genome in genomes.items():
        matches = [
            spawn_parents
            for spawn_parents, _, offspring in run.port.spawned
            if offspring.genome == genome and spawn_parents == parents[agent_id]
        ]
        assert matches, (
            f"{agent_id}'s genome and parents match no offspring the port's spawn "
            "returned for those parents"
        )
        assert any(
            _contains_run(selected, parents[agent_id]) for selected in run.port.selected
        ), (
            f"{agent_id}'s parents {parents[agent_id]} are no run of a selection's return"
        )


def _contains_run(selected: tuple[str, ...], run: tuple[str, ...]) -> bool:
    width = len(run)
    return any(
        selected[start : start + width] == run
        for start in range(0, len(selected) - width + 1, width)
    )


def _births_seen_at_their_birth(monkeypatch) -> tuple[object, dict, dict]:
    genomes: dict[str, object] = {}
    parents: dict[str, tuple[str, ...]] = {}
    original = ecology.steady_state.SteadyState._admitted

    def watching(self, window, birth, offspring, blind, attempt):
        lineage = original(self, window, birth, offspring, blind, attempt)
        genomes[birth.agent_id] = self.state.genome(birth.agent_id)
        parents[birth.agent_id] = self.state.lineage(birth.agent_id).parents
        return lineage

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_admitted", watching)
    config = _turnover()
    run = streams.steady(
        windows=streams.priced(count=DECISION_POINTS),
        founding=streams.founders(config.band_lower),
        turnover=config,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
        run_id="traced",
    )
    observed = {
        agent_id
        for index in range(DECISION_POINTS)
        for agent_id in run.watched.births(index)
    }
    assert observed == set(genomes), "a birth was observed that the trace did not see"
    return run, genomes, parents


def test_births_reach_parents_and_offspring_only_through_the_port(monkeypatch) -> None:
    """Every birth is a spawn return, of parents a selection returned."""
    run, genomes, parents = _births_seen_at_their_birth(monkeypatch)
    _births_traced_to_the_port(run, genomes, parents)


def test_admitting_a_parents_copy_turns_the_trace_red(monkeypatch) -> None:
    """Seeded fault: the loop admits the first parent's genome instead of the offspring."""
    real = ecology.inheritance.birth_of

    def copy_of_the_parent(**keys):
        born = real(**keys)
        parent = keys["population"].genome(keys["parents"][0])
        return replace(born, genome=parent)

    monkeypatch.setattr(ecology.inheritance, "birth_of", copy_of_the_parent)
    run, genomes, parents = _births_seen_at_their_birth(monkeypatch)
    with pytest.raises(AssertionError, match="match no offspring"):
        _births_traced_to_the_port(run, genomes, parents)


def test_inline_top_k_selection_turns_the_trace_red(monkeypatch) -> None:
    """Seeded fault: the loop picks the highest figures itself and never asks the port."""

    def top_k(self, trigger, survivors, by_id, births):
        ranked = sorted(
            (agent_id for agent_id in survivors if by_id.get(agent_id) is not None),
            key=lambda agent_id: -by_id[agent_id].deflated,
        )
        width = self.turnover.parents_per_birth
        chunk = tuple(ranked[:width])
        return [[chunk] * (self.turnover.admission_redraws + 1)] * births, False

    monkeypatch.setattr(ecology.steady_state.SteadyState, "_parents", top_k)
    run, genomes, parents = _births_seen_at_their_birth(monkeypatch)
    with pytest.raises(AssertionError, match="no run of a selection"):
        _births_traced_to_the_port(run, genomes, parents)
