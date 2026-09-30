"""The online update's own behaviour, below the level the facts assert it at.

The fact evidence in `tests/facts/test_learning.py` makes claims about what a
population does. What is checked here is the arithmetic underneath: the columns
the update declares, the cold start it seeds, the two bounds it applies and the
refusals a population raises when it is asked to learn without being able to.
Nothing here is evidence of anything and nothing here is market data — the
inputs are numbers written in this file.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType

import numpy as np
import pytest

from liq.evolution.ecology import agent, config, learning
from liq.evolution.ecology.types import Bar, BarWindow, Genome
from tests.support import level_shift_stream as stream

FEATURES = ("alpha", "beta")
GENE_SCHEMA_VERSION = "unit-genes-1"
FEATURE_SCHEMA_VERSION = "unit-features-1"
MODEL_VERSION = "unit-model-1"
INSTRUMENT = "UNIT"
FORGETTING = 0.95

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def _birth(
    agent_id: str = "one",
    *,
    forgetting: float = FORGETTING,
    prior: float = 0.0,
) -> agent.AgentBirth:
    """One agent carrying both features and a forgetting gene inside the range."""
    genes: dict[str, float] = {agent.ENTRY_THRESHOLD: 0.0}
    for name in FEATURES:
        genes[f"{agent.MASK_PREFIX}{name}"] = 1.0
        genes[f"{agent.WEIGHT_PREFIX}{name}"] = prior
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
    started = learning.cold_start(dict.fromkeys(FEATURES, 0.0), config.LearningConfig())
    assert set(started) == declared


def test_the_cold_start_seeds_the_accumulator_the_weight_is_read_from() -> None:
    """Each agent's prior survives the first observation rather than vanishing.

    FTRL recomputes its weights from an accumulator every bar, so a prior
    written only into the weights would be discarded the first time anything
    arrived. Seeding the accumulator is what makes the genome's value the
    recursion's starting point — per feature, per agent.
    """
    priors = {"alpha": 0.4, "beta": -0.3}
    started = learning.cold_start(priors, config.LearningConfig())
    for name, prior in priors.items():
        assert started[f"{learning.WEIGHT_PREFIX}{name}"] == prior
        assert started[f"{learning.DRIFT_PREFIX}{name}"] != 0.0
        assert started[f"{learning.ENERGY_PREFIX}{name}"] == 0.0
    assert started[learning.SCALE_MASS] == 0.0

    state = agent.PopulationState.founded(
        (_birth("leans", prior=0.4), _birth("level")),
        learning=config.LearningConfig(),
    )
    assert set(state.learned_weights("leans").values()) == {0.4}
    assert set(state.learned_weights("level").values()) == {0.0}
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    state.observe(ORIGIN + 2 * MINUTE, {"leans": 1.0, "level": 1.0})
    survived = state.learned_weights("leans")
    assert survived["alpha"] == pytest.approx(FORGETTING * 0.4)
    assert set(state.learned_weights("level").values()) == {0.0}


def test_a_start_outside_the_weight_bound_is_refused_not_clipped() -> None:
    """A prior or an inherited state the bound would not admit never starts.

    Both starting points are held to it, because both are read by the first
    decision; and a weight that is not a number is refused with them.
    """
    bound = config.LearningConfig().weight_norm_bound
    at_the_bound = bound / np.sqrt(len(FEATURES))
    admitted = agent.PopulationState.founded(
        (_birth(prior=at_the_bound),), learning=config.LearningConfig()
    )
    assert float(np.hypot(*admitted.learned_weights("one").values())) <= bound

    for prior in (np.nextafter(at_the_bound, np.inf) * 1.0001, float("nan")):
        with pytest.raises(agent.StartingStateOutsideItsBound, match="'one'"):
            agent.PopulationState.founded(
                (_birth(prior=prior),), learning=config.LearningConfig()
            )

    inherited = dict(admitted.learned_state("one"))
    inherited[f"{learning.WEIGHT_PREFIX}alpha"] = float("inf")
    with pytest.raises(agent.StartingStateOutsideItsBound):
        agent.PopulationState.founded(
            (replace(_birth(), learned_state=MappingProxyType(inherited)),),
            learning=config.LearningConfig(),
        )


def test_a_restore_is_not_a_birth_and_reads_no_prior() -> None:
    """What was put away comes back as it was, whatever the genes now say.

    The prior is read at a birth; a restore resumes a state, so its genes'
    ``weight.`` values are never re-applied and the starting bound is not
    re-checked against it.
    """
    state = agent.PopulationState.founded(
        (_birth(prior=0.4),), learning=config.LearningConfig()
    )
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    put_away = state.snapshot()
    restored = agent.PopulationState.restored(put_away)
    assert dict(restored.learned_state("one")) == dict(state.learned_state("one"))
    assert set(restored.learned_weights("one").values()) != {0.4}


def test_an_update_bound_to_columns_that_are_not_there_refuses() -> None:
    """A column the update owns and the population never declared is a KeyError."""
    with pytest.raises(KeyError, match="not among its learned names"):
        learning.OnlineUpdate(
            config=config.LearningConfig(),
            features=FEATURES,
            column_of={learning.SCALE_MASS: 0},
        )


def test_the_gain_is_held_under_its_bound_and_the_holding_is_recorded() -> None:
    """A state bound tighter than the arithmetic's own maximum is applied.

    The gain starts at step_scale / step_offset per feature before any evidence
    arrives, so a bound below that is met at founding, held and recorded then,
    and met again at the first observation, which carries no gradient and lets
    the evidence decay.
    """
    tight = config.LearningConfig(state_bound=0.05)
    state = _founded(tight)
    founding = state.bounds_reached()
    assert founding == (
        agent.BoundReached(
            agent_id="one", as_of=None, bound=config.STATE_BOUND_KEY, at_founding=True
        ),
    )
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    reached = state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    assert [record.bound for record in reached] == [config.STATE_BOUND_KEY]
    assert not any(record.at_founding for record in reached)
    assert state.bounds_reached() == founding + reached
    size = sum(value**2 for value in state.estimator_state("one").values()) ** 0.5
    assert size <= tight.state_bound * (1.0 + 1e-9)


def test_a_founding_gain_under_a_tight_bound_is_held_and_the_prior_kept_exact() -> None:
    """Held at founding: the gain lands on the bound and each weight is its gene.

    The evidence is raised to the least value the bound admits, by the same
    hold every outcome applies, and the prior is seeded against that evidence.
    So the weight the accumulator implies is the gene, bit for bit, and not a
    shrunk version of it. A founder with a stated birth instant is recorded at
    that instant, in UTC, and one without is recorded at no instant.
    """
    tight = config.LearningConfig(state_bound=0.05)
    born_at = datetime(2024, 1, 1, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
    state = agent.PopulationState.founded(
        (_birth("leans", prior=0.4), replace(_birth("dated"), born_at=born_at)),
        learning=tight,
    )
    energy, held = learning.founding_energy(tight, FEATURES)
    assert held
    for agent_id, prior in (("leans", 0.4), ("dated", 0.0)):
        stored = state.learned_state(agent_id)
        assert {
            name: stored[f"{learning.ENERGY_PREFIX}{name}"] for name in FEATURES
        } == (dict(energy))
        assert set(state.learned_weights(agent_id).values()) == {prior}
        gain = sum(v**2 for v in state.estimator_state(agent_id).values()) ** 0.5
        assert gain == pytest.approx(tight.state_bound, rel=1e-12)
    step = (
        tight.step_offset + np.sqrt(energy["alpha"])
    ) / tight.step_scale + tight.squared_penalty
    assert state.learned_state("leans")[f"{learning.DRIFT_PREFIX}alpha"] == (
        -0.4 * step
    )
    assert state.bounds_reached() == (
        agent.BoundReached(
            agent_id="leans", as_of=None, bound=config.STATE_BOUND_KEY, at_founding=True
        ),
        agent.BoundReached(
            agent_id="dated",
            as_of=born_at.astimezone(UTC),
            bound=config.STATE_BOUND_KEY,
            at_founding=True,
        ),
    )


def test_under_the_shipped_bound_founding_is_exactly_what_it_was() -> None:
    """A bound at or above the founding gain leaves founding untouched and unrecorded."""
    shipped = config.LearningConfig()
    energy, held = learning.founding_energy(shipped, FEATURES)
    assert (dict(energy), held) == (dict.fromkeys(FEATURES, 0.0), False)
    priors = {"alpha": 0.4, "beta": -0.3}
    assert learning.cold_start(priors, shipped, energy) == learning.cold_start(
        priors, shipped
    )
    assert _founded().bounds_reached() == ()


def test_an_inherited_state_is_admitted_verbatim_or_refused_never_held() -> None:
    """A carried state inside the state bound starts exactly as carried; one outside is refused.

    Holding it would change what crossed the birth boundary. A gain that is not
    a number is outside the bound too.
    """
    parent = agent.PopulationState.founded(
        (_birth(),), learning=config.LearningConfig()
    )
    parent.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    parent.observe(ORIGIN + 2 * MINUTE, {"one": 1.0})
    carried = dict(parent.learned_state("one"))
    carried_gain = sum(v**2 for v in parent.estimator_state("one").values()) ** 0.5

    inside = config.LearningConfig(state_bound=carried_gain)
    child = agent.PopulationState.founded(
        (replace(_birth(), learned_state=MappingProxyType(carried)),), learning=inside
    )
    assert dict(child.learned_state("one")) == carried
    assert child.bounds_reached() == ()

    below = config.LearningConfig(state_bound=carried_gain / 2.0)
    with pytest.raises(agent.StartingStateOutsideItsBound, match="inherited gain"):
        agent.PopulationState.founded(
            (replace(_birth(), learned_state=MappingProxyType(carried)),),
            learning=below,
        )
    not_a_number = {**carried, f"{learning.ENERGY_PREFIX}alpha": float("nan")}
    with pytest.raises(agent.StartingStateOutsideItsBound):
        agent.PopulationState.founded(
            (replace(_birth(), learned_state=MappingProxyType(not_a_number)),),
            learning=config.LearningConfig(),
        )


def test_only_a_founding_hold_may_carry_no_instant() -> None:
    """A bound met at an outcome says when; only a founding hold may not."""
    with pytest.raises(ValueError, match="only a founding hold"):
        agent.BoundReached(agent_id="one", as_of=None, bound=config.STATE_BOUND_KEY)


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_an_outcome_that_is_not_a_number_is_refused_and_nothing_is_learned(
    value: float,
) -> None:
    """One non-finite outcome would poison the agent for good, so it is refused.

    Absorbed, it turns the weights and the gain into not-a-number from then on,
    records no bound, and leaves the agent flat at every later decision point
    with nothing saying why. Refused, the population is exactly as it was: the
    learned state, the records and the reading awaiting its outcome are all
    untouched, so a finite outcome for the same reading is still accepted.
    Nothing is imputed in its place.
    """
    state = agent.PopulationState.founded(
        (_birth("one"), _birth("two")), learning=config.LearningConfig()
    )
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    before = {name: dict(state.learned_state(name)) for name in state.agent_ids()}
    pending = state.standardised("two")

    with pytest.raises(agent.OutcomeNotFinite, match="'two'"):
        state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0, "two": value})

    assert {name: dict(state.learned_state(name)) for name in state.agent_ids()} == (
        before
    )
    assert state.bounds_reached() == ()
    assert state.standardised("two") == pending
    state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0, "two": -1.0})
    assert all(
        np.isfinite(list(state.learned_state(name).values())).all()
        for name in state.agent_ids()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_feature_that_is_not_a_number_is_refused_and_nothing_is_touched(
    value: float,
) -> None:
    """One non-finite feature would poison the standardiser for good, so it is refused.

    Folded in, it turns that feature's moments into not-a-number from then on.
    Every later reading of it is then not-a-number, whatever the view offers.
    Refused, the population is exactly as it was: no moment has moved, no
    reading is put aside and no wish is formed, so the next finite view is
    answered as though the refused one had never been offered. Nothing is
    imputed in its place.
    """
    state = agent.PopulationState.founded(
        (_birth("one"), _birth("two")), learning=config.LearningConfig()
    )
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    state.observe(ORIGIN + 2 * MINUTE, {"one": 1.0, "two": -1.0})
    before = {name: dict(state.learned_state(name)) for name in state.agent_ids()}
    wanted = {name: state.intended(name) for name in state.agent_ids()}

    with pytest.raises(agent.FeatureNotFinite, match="'beta'"):
        state.step(_window(1, {"alpha": 2.0, "beta": value}), INSTRUMENT)

    assert {name: dict(state.learned_state(name)) for name in state.agent_ids()} == (
        before
    )
    assert {name: state.intended(name) for name in state.agent_ids()} == wanted
    with pytest.raises(agent.NothingWasShown):
        state.standardised("one")
    state.step(_window(1, {"alpha": 2.0, "beta": 3.0}), INSTRUMENT)
    assert all(
        np.isfinite(list(state.learned_state(name).values())).all()
        for name in state.agent_ids()
    )


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
    declared = config.LearningConfig(state_bound=500.0)
    state = _founded(declared)
    put_away = state.snapshot()
    assert put_away.learning == declared
    assert agent.PopulationState.restored(put_away).learning == declared


def test_a_pending_reading_that_does_not_fit_is_refused_at_restore() -> None:
    """A reading an outcome could not be paired with is refused, not trimmed."""
    state = agent.PopulationState.founded(
        (_birth("one"), _birth("two")), learning=config.LearningConfig()
    )
    assert state.snapshot().pending is None
    state.step(_window(0, {"alpha": 1.0, "beta": -1.0}), INSTRUMENT)
    put_away = state.snapshot()
    assert put_away.pending is not None
    readings = dict(put_away.pending.readings)
    for spoiled, match in (
        (replace(put_away, learning=None), "does not learn"),
        (
            replace(
                put_away,
                pending=replace(put_away.pending, readings={"one": readings["one"]}),
            ),
            "exactly the agents",
        ),
        (
            replace(
                put_away,
                pending=replace(
                    put_away.pending,
                    readings={**readings, "two": readings["two"][:1]},
                ),
            ),
            "features",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            agent.PopulationState.restored(spoiled)


def test_a_pending_reading_is_stamped_in_utc() -> None:
    """The instant a reading was formed at is held in UTC whatever it came in."""
    elsewhere = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))
    pending = agent.PendingReading(as_of=elsewhere, readings={"one": (0.0,)})
    assert pending.as_of == datetime(2024, 1, 1, tzinfo=UTC)
    assert pending.as_of.tzinfo is UTC


def test_a_population_that_does_not_learn_snapshots_as_one() -> None:
    """The absence of an update is carried too, rather than left to be inferred."""
    put_away = agent.PopulationState.founded((_birth(),)).snapshot()
    assert put_away.learning is None
    assert agent.PopulationState.restored(put_away).learning is None


# ---------------------------------------------------------------------------
# What a birth carries across. A birth either carries no learned state, and is
# seeded from its own prior, or carries the whole of what the update declares,
# and is admitted verbatim. Anything in between is refused, because a carried
# weight without its accumulator is an inconsistent state and a name the update
# does not declare is a column nothing chose.
# ---------------------------------------------------------------------------

#: Where the walked parent hands its state on, and how fast it forgets. The
#: slow end of the declared range, where what is learned outlives the prior
#: longest, so a birth that silently dropped it would be furthest from right.
BORN_AT = 400
PARENT_FORGETS_AT = 0.9999

#: How many post-birth decision points the carrying offspring and its
#: state-less sibling must wish differently at, for the carried state to count
#: as having crossed the birth rather than been swallowed by it. The fast agent
#: of the estimator ADR took 20 bars to turn over.
DIFFERING_AFTER_BIRTH = 20


def _walked_parent() -> dict[str, float]:
    """Everything a parent learned over the level-shift stream up to ``BORN_AT``."""
    parent = agent.PopulationState.founded(
        (stream.birth("parent", forgetting=PARENT_FORGETS_AT),),
        learning=config.LearningConfig(),
    )
    stream.walk(parent, stream.windows()[: BORN_AT + 1])
    return dict(parent.learned_state("parent"))


def test_a_birth_carrying_learned_state_starts_from_exactly_that_state() -> None:
    """A Lamarckian birth through ``founded`` is not silently made Baldwinian."""
    carried = _walked_parent()
    assert carried[f"{learning.WEIGHT_PREFIX}{stream.FEATURE}"] != 0.0
    offspring = agent.PopulationState.founded(
        (
            replace(
                stream.birth("offspring", forgetting=PARENT_FORGETS_AT),
                learned_state=MappingProxyType(carried),
            ),
        ),
        learning=config.LearningConfig(),
    )
    assert dict(offspring.learned_state("offspring")) == carried


def test_a_carried_state_reaches_the_wishes_after_birth() -> None:
    """The offspring that carries its parent's state and a sibling that carries
    nothing, born together, go on to wish differently."""
    carried = _walked_parent()
    born = agent.PopulationState.founded(
        (
            replace(
                stream.birth("carries", forgetting=PARENT_FORGETS_AT),
                learned_state=MappingProxyType(carried),
            ),
            stream.birth("starts-fresh", forgetting=PARENT_FORGETS_AT),
        ),
        learning=config.LearningConfig(),
    )
    after = stream.windows()[BORN_AT:]
    walked = stream.walk(
        born,
        after,
        outcomes=tuple(
            stream.outcome_at(index) for index in range(BORN_AT, stream.DECISION_POINTS)
        ),
    )
    differing = sum(
        mine != theirs
        for mine, theirs in zip(
            walked["carries"].wanted, walked["starts-fresh"].wanted, strict=True
        )
    )
    assert differing >= DIFFERING_AFTER_BIRTH, (
        f"the offspring carrying its parent's state and the sibling carrying "
        f"none wished differently at {differing} of {len(after)} post-birth "
        f"decision points"
    )


def test_a_birth_carrying_part_of_what_the_update_declares_is_refused() -> None:
    """A weight without its accumulator is not a state anything could resume."""
    carried = _walked_parent()
    partial = {
        name: value
        for name, value in carried.items()
        if not name.startswith(learning.DRIFT_PREFIX)
    }
    with pytest.raises(ValueError, match=learning.DRIFT_PREFIX):
        agent.PopulationState.founded(
            (
                replace(
                    stream.birth("partial", forgetting=PARENT_FORGETS_AT),
                    learned_state=MappingProxyType(partial),
                ),
            ),
            learning=config.LearningConfig(),
        )


def test_a_birth_carrying_a_name_the_update_does_not_declare_is_refused() -> None:
    """A misspelled column is refused rather than grown as a new one."""
    carried = _walked_parent()
    misspelled = f"learned.drift.{stream.FEATURE}"
    for learned_state in (
        {**carried, misspelled: 1.0},
        {misspelled: 1.0},
    ):
        with pytest.raises(ValueError, match=misspelled):
            agent.PopulationState.founded(
                (
                    replace(
                        stream.birth("stranger", forgetting=PARENT_FORGETS_AT),
                        learned_state=MappingProxyType(learned_state),
                    ),
                ),
                learning=config.LearningConfig(),
            )
