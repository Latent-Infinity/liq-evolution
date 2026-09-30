"""The online update's own behaviour, below the level the facts assert it at.

The fact evidence in `tests/facts/test_learning.py` makes claims about what a
population does. What is checked here is the arithmetic underneath: the columns
the update declares, the cold start it seeds, the two bounds it applies and the
refusals a population raises when it is asked to learn without being able to.
Nothing here is evidence of anything and nothing here is market data — the
inputs are numbers written in this file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from liq.evolution.ecology import agent, config, learning
from liq.evolution.ecology.types import Bar, BarWindow, Genome

FEATURES = ("alpha", "beta")
GENE_SCHEMA_VERSION = "unit-genes-1"
FEATURE_SCHEMA_VERSION = "unit-features-1"
MODEL_VERSION = "unit-model-1"
INSTRUMENT = "UNIT"
FORGETTING = 0.95

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def _birth(
    agent_id: str = "one", *, forgetting: float = FORGETTING
) -> agent.AgentBirth:
    """One agent carrying both features and a forgetting gene inside the range."""
    genes: dict[str, float] = {agent.ENTRY_THRESHOLD: 0.0}
    for name in FEATURES:
        genes[f"{agent.MASK_PREFIX}{name}"] = 1.0
        genes[f"{agent.WEIGHT_PREFIX}{name}"] = 1.0
    genes[agent.FORGETTING_FACTOR] = forgetting
    return agent.AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=MappingProxyType(genes), schema_version=GENE_SCHEMA_VERSION
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def _window(index: int, values: dict[str, float]) -> BarWindow:
    """One decision point carrying ``values`` as its whole feature view."""
    start = ORIGIN + index * MINUTE
    bar = Bar(
        instrument=INSTRUMENT,
        period_start=start,
        period_end=start + MINUTE,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )
    return BarWindow(
        as_of=start + MINUTE,
        segment_id="unit-segment",
        segment_role="train",
        bars=MappingProxyType({INSTRUMENT: bar}),
        features=MappingProxyType({INSTRUMENT: MappingProxyType(values)}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def _founded(learns: config.LearningConfig | None = None) -> agent.PopulationState:
    """One learning agent, ready to be shown a decision point."""
    return agent.PopulationState.founded(
        (_birth(),), learning=config.LearningConfig() if learns is None else learns
    )


def test_the_declared_columns_cover_everything_the_update_writes() -> None:
    """Every name the update reads or writes is one an agent is born declaring."""
    declared = set(learning.declared_columns(FEATURES))
    assert set(learning.weight_names(FEATURES)) <= declared
    assert learning.SCALE_MASS in declared
    assert set(learning.cold_start(FEATURES, config.LearningConfig())) == declared


def test_the_cold_start_seeds_the_accumulator_the_weight_is_read_from() -> None:
    """A declared cold start survives the first observation rather than vanishing.

    FTRL recomputes its weights from an accumulator every bar, so a cold start
    written only into the weights would be discarded the first time anything
    arrived. Seeding the accumulator is what makes the declared value the
    recursion's starting point.
    """
    declared = config.LearningConfig(cold_start_weight=0.4)
    started = learning.cold_start(FEATURES, declared)
    for name in FEATURES:
        assert started[f"{learning.WEIGHT_PREFIX}{name}"] == pytest.approx(0.4)
        assert started[f"{learning.DRIFT_PREFIX}{name}"] != 0.0

    state = _founded(declared)
    assert set(state.learned_weights("one").values()) == {0.4}


def test_an_update_bound_to_columns_that_are_not_there_refuses() -> None:
    """A column the update owns and the population never declared is a KeyError."""
    with pytest.raises(KeyError, match="not among its learned names"):
        learning.OnlineUpdate(
            config=config.LearningConfig(),
            features=FEATURES,
            column_of={learning.SCALE_MASS: 0},
        )


def test_the_gain_is_held_under_its_bound_and_the_holding_is_recorded() -> None:
    """A state bound tighter than the arithmetic's own maximum is applied."""
    # The gain starts at step_scale / step_offset before any evidence arrives,
    # so a bound below that is one this pass reaches on its first observation.
    tight = config.LearningConfig(state_bound=0.05)
    state = _founded(tight)
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    reached = state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    assert [record.bound for record in reached] == [config.STATE_BOUND_KEY]
    assert state.bounds_reached() == reached
    size = sum(value**2 for value in state.estimator_state("one").values()) ** 0.5
    assert size <= tight.state_bound * (1.0 + 1e-9)


def test_the_weights_are_held_under_their_bound_and_the_holding_is_recorded() -> None:
    """A weight-norm bound is applied to the accumulator, so it survives the bar."""
    tight = config.LearningConfig(weight_norm_bound=1e-6)
    state = _founded(tight)
    reached: tuple[agent.BoundReached, ...] = ()
    for index, (alpha, beta, outcome) in enumerate(
        ((2.0, -2.0, 5.0), (-3.0, 4.0, -7.0), (1.0, 6.0, 9.0), (-5.0, 2.0, -4.0))
    ):
        state.step(_window(index, {"alpha": alpha, "beta": beta}), INSTRUMENT)
        reached += state.observe(ORIGIN + (index + 2) * MINUTE, {"one": outcome})
    assert config.WEIGHT_NORM_BOUND_KEY in {record.bound for record in reached}
    size = sum(value**2 for value in state.learned_weights("one").values()) ** 0.5
    assert size <= tight.weight_norm_bound * (1.0 + 1e-9)


def test_an_absolute_penalty_sets_an_unearned_weight_to_exactly_zero() -> None:
    """The closed form's shrinkage branch is reachable and sets zero, not nearly."""
    state = _founded(config.LearningConfig(absolute_penalty=1e9))
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    assert set(state.learned_weights("one").values()) == {0.0}


def test_the_reported_norms_are_the_sizes_of_what_they_name() -> None:
    """The two norms the bounds are judged against read off the same arrays."""
    state = _founded()
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    update = learning.OnlineUpdate(
        config=config.LearningConfig(),
        features=state.feature_names,
        column_of={name: index for index, name in enumerate(state.learned_names)},
    )
    held = np.asarray(
        [list(state.learned_state("one").values())], dtype=agent.STORAGE_DTYPE
    )
    assert float(update.state_norms(held)[0]) == pytest.approx(
        sum(v**2 for v in state.estimator_state("one").values()) ** 0.5
    )
    assert float(update.weight_norms(held)[0]) == pytest.approx(
        sum(v**2 for v in state.learned_weights("one").values()) ** 0.5
    )
    assert update.features == state.feature_names


def test_a_population_with_no_update_refuses_every_question_about_one() -> None:
    """Asked to learn without a learning configuration, a population says so."""
    state = agent.PopulationState.founded((_birth(),))
    for ask in (
        lambda: state.learned_weights("one"),
        lambda: state.estimator_state("one"),
        lambda: state.standardised("one"),
        lambda: state.observe(ORIGIN + MINUTE, {"one": 1.0}),
    ):
        with pytest.raises(agent.PopulationDoesNotLearn):
            ask()


def test_a_learning_population_whose_genomes_carry_no_rate_refuses() -> None:
    """How fast an agent forgets is inherited, so a genome without it is refused."""
    genes = dict(_birth().genome.genes)
    del genes[agent.FORGETTING_FACTOR]
    rateless = agent.AgentBirth(
        agent_id="one",
        genome=Genome(
            genes=MappingProxyType(genes), schema_version=GENE_SCHEMA_VERSION
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )
    with pytest.raises(KeyError, match="never defaulted"):
        agent.PopulationState.founded((rateless,), learning=config.LearningConfig())


def test_an_outcome_at_the_reading_s_own_instant_is_refused() -> None:
    """The cheapest look-ahead there is, refused rather than absorbed."""
    state = _founded()
    window = _window(0, {"alpha": 1.0, "beta": -1.0})
    state.step(window, INSTRUMENT)
    with pytest.raises(agent.OutcomeFromTheSameBar):
        state.observe(window.as_of, {"one": 1.0})


def test_an_outcome_before_anything_was_shown_is_refused() -> None:
    """There is no reading for it to belong to, and none is invented."""
    state = _founded()
    with pytest.raises(agent.NothingWasShown):
        state.observe(ORIGIN + MINUTE, {"one": 1.0})
    with pytest.raises(agent.NothingWasShown):
        state.standardised("one")


def test_an_outcome_that_leaves_a_living_agent_out_is_refused() -> None:
    """Part of a population updated is a population nobody can describe."""
    state = agent.PopulationState.founded(
        (_birth("one"), _birth("two")), learning=config.LearningConfig()
    )
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    with pytest.raises(KeyError, match="were given no outcome"):
        state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})


def test_a_snapshot_carries_what_the_update_was_turned_by() -> None:
    """A restored population learns under the configuration it was put away with."""
    declared = config.LearningConfig(cold_start_weight=0.3)
    state = _founded(declared)
    put_away = state.snapshot()
    assert put_away.learning == declared
    assert agent.PopulationState.restored(put_away).learning == declared


def test_a_population_that_does_not_learn_snapshots_as_one() -> None:
    """The absence of an update is carried too, rather than left to be inferred."""
    put_away = agent.PopulationState.founded((_birth(),)).snapshot()
    assert put_away.learning is None
    assert agent.PopulationState.restored(put_away).learning is None
