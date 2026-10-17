"""Births into, and deaths from, a population that is already walking.

A population comes into being whole, from :meth:`PopulationState.founded` or
:meth:`PopulationState.restored`. What is checked here is what happens to it
afterwards: an agent admitted while the others are mid-walk, and an agent
removed from them. Admission goes through the same birth rule founding does —
a birth carries no learned state or exactly the whole of what the update
declares, a birth carrying nothing has its founding gain held and recorded,
and a start outside the bounds is refused and counted.

Nothing here is market data. The walks are over the declared arithmetic stream
in ``tests/support/level_shift_stream.py``, and every bound a check uses is
derived from a pass over that stream under the shipped configuration rather
than written down here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType

import numpy as np
import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent, config, learning
from liq.evolution.ecology.types import BarWindow
from tests.support import level_shift_stream as stream

#: The slow end of the declared forgetting range, where what an agent learned
#: outlives its prior longest, so a birth that silently dropped a carried state
#: would be furthest from right. The same cell the ``founded()`` ruling used.
PARENT_FORGETS_AT = 0.9999

#: Where the walked parent hands its state on.
BORN_AT = 400

#: How many post-birth decision points a carrying offspring and its state-less
#: sibling must wish differently at. The 2026-09-30 ruling's threshold
#: (`plans/oracle-weight-gene-prior-2026-09-30.md`, the ``founded()`` pair,
#: check 2), not a number chosen here.
DIFFERING_AFTER_BIRTH = 20

#: A non-zero prior, so a birth that ignored its own genes would be visible.
PRIOR = 0.4


def _walk(
    state: agent.PopulationState,
    windows: Sequence[BarWindow],
    *,
    first: int,
) -> dict[str, list[float]]:
    """Step and settle ``state`` over ``windows``, which start at stream index ``first``.

    Every decision point is followed by its outcome, stamped at the next
    window's instant, so the walk ends settled: no reading is left awaiting an
    outcome, and an admission or a cull may follow it. The last window is the
    instant the final outcome becomes known and is not itself stepped.
    """
    wanted: dict[str, list[float]] = {agent_id: [] for agent_id in state.agent_ids()}
    for offset, window in enumerate(windows[:-1]):
        step = state.step(window, stream.INSTRUMENT)
        for agent_id, exposure in zip(step.agent_ids, step.wanted, strict=True):
            wanted.setdefault(agent_id, []).append(exposure)
        state.observe(
            windows[offset + 1].as_of,
            dict.fromkeys(state.agent_ids(), stream.outcome_at(first + offset)),
        )
    return wanted


def _parent_walked_to(
    born_at: int, learns: config.LearningConfig
) -> agent.PopulationState:
    """A population of one parent, walked and settled up to decision point ``born_at``."""
    state = agent.PopulationState.founded(
        (stream.birth("parent", forgetting=PARENT_FORGETS_AT),), learning=learns
    )
    _walk(state, stream.windows()[: born_at + 1], first=0)
    return state


def _carrying(
    agent_id: str,
    carried: Mapping[str, float],
    *,
    born_at: datetime,
    parents: tuple[str, ...] = ("parent",),
) -> agent.AgentBirth:
    """An offspring of the stream's genome, carrying ``carried`` across its birth."""
    return replace(
        stream.birth(agent_id, forgetting=PARENT_FORGETS_AT),
        learned_state=MappingProxyType(dict(carried)),
        parents=parents,
        born_at=born_at,
    )


def _fresh(
    agent_id: str,
    *,
    born_at: datetime,
    prior: float = stream.PRIOR_AT,
    parents: tuple[str, ...] = ("parent",),
) -> agent.AgentBirth:
    """An offspring of the stream's genome, carrying nothing across its birth."""
    return replace(
        stream.birth(agent_id, forgetting=PARENT_FORGETS_AT, prior=prior),
        parents=parents,
        born_at=born_at,
    )


def _instant(index: int) -> datetime:
    """The stream's decision instant at ``index``."""
    return stream.windows()[index].as_of


def _norm(values: Mapping[str, float]) -> float:
    return float(np.linalg.norm(np.fromiter(values.values(), dtype=np.float64)))


# ---------------------------------------------------------------------------
# (a) The founded() pair, re-run at the path in-run births take.
# ---------------------------------------------------------------------------


def test_a_carried_state_is_admitted_mid_walk_bit_for_bit() -> None:
    """A Lamarckian birth into a walking population starts from exactly what it carried."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    carried = dict(state.learned_state("parent"))
    assert carried[f"{learning.WEIGHT_PREFIX}{stream.FEATURE}"] != 0.0

    state.admit(_carrying("offspring", carried, born_at=_instant(BORN_AT)))

    assert dict(state.learned_state("offspring")) == carried
    assert state.agent_ids() == ("parent", "offspring")


def test_a_carried_state_and_a_fresh_sibling_admitted_mid_walk_wish_apart() -> None:
    """The carrying offspring and a same-genome sibling carrying nothing wish differently."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    carried = dict(state.learned_state("parent"))
    born_at = _instant(BORN_AT)
    state.admit(_carrying("carries", carried, born_at=born_at))
    state.admit(_fresh("starts-fresh", born_at=born_at))

    after = stream.windows()[BORN_AT:]
    wanted = _walk(state, after, first=BORN_AT)

    differing = sum(
        mine != theirs
        for mine, theirs in zip(wanted["carries"], wanted["starts-fresh"], strict=True)
    )
    assert differing >= DIFFERING_AFTER_BIRTH, (
        f"the offspring carrying its parent's state and the sibling carrying none "
        f"wished differently at {differing} of {len(after) - 1} post-birth "
        "decision points"
    )


def test_a_partial_or_undeclared_carried_state_is_refused_at_admission() -> None:
    """A weight without its accumulator, or a column nothing chose, is not admitted."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    carried = dict(state.learned_state("parent"))
    born_at = _instant(BORN_AT)
    partial = {
        name: value
        for name, value in carried.items()
        if not name.startswith(learning.DRIFT_PREFIX)
    }
    misspelled = f"learned.drift.{stream.FEATURE}"
    before = state.storage()

    with pytest.raises(ValueError, match=learning.DRIFT_PREFIX):
        state.admit(_carrying("partial", partial, born_at=born_at))
    with pytest.raises(ValueError, match=misspelled):
        state.admit(
            _carrying("stranger", {**carried, misspelled: 1.0}, born_at=born_at)
        )

    assert state.agent_ids() == ("parent",)
    assert state.storage() == before
    assert state.admission_refusals() == 0, (
        "an incomplete state is refused as malformed, not as outside its bound"
    )


# ---------------------------------------------------------------------------
# (b) A birth carrying nothing: its own prior, the founding hold, the record.
# ---------------------------------------------------------------------------


def _held_at_founding() -> config.LearningConfig:
    """A state bound below the founding gain, derived from a founded agent's own gain."""
    shipped = agent.PopulationState.founded(
        (stream.birth("probe", forgetting=PARENT_FORGETS_AT),),
        learning=config.LearningConfig(),
    )
    founding_gain = _norm(shipped.estimator_state("probe"))
    return config.LearningConfig(state_bound=founding_gain / 2.0)


def test_a_fresh_birth_mid_walk_starts_exactly_as_founding_would_start_it() -> None:
    """Its prior, held at founding and recorded, as ``founded()`` does it."""
    tight = _held_at_founding()
    state = _parent_walked_to(BORN_AT, tight)
    born_at = _instant(BORN_AT).astimezone(timezone(timedelta(hours=-5)))
    birth = _fresh("newborn", born_at=born_at, prior=PRIOR)
    recorded_before = state.bounds_reached()

    state.admit(birth)

    as_founded = agent.PopulationState.founded((birth,), learning=tight)
    assert dict(state.learned_state("newborn")) == dict(
        as_founded.learned_state("newborn")
    )
    assert state.learned_weights("newborn") == {stream.FEATURE: PRIOR}
    energy, held = learning.founding_energy(tight, (stream.FEATURE,))
    assert held
    assert (
        state.learned_state("newborn")[f"{learning.ENERGY_PREFIX}{stream.FEATURE}"]
        == energy[stream.FEATURE]
    )
    assert state.bounds_reached() == (
        *recorded_before,
        agent.BoundReached(
            agent_id="newborn",
            as_of=born_at.astimezone(UTC),
            bound=config.STATE_BOUND_KEY,
            at_founding=True,
        ),
    )
    assert as_founded.bounds_reached() == state.bounds_reached()[-1:]


def test_a_fresh_birth_under_the_shipped_bound_is_not_held_or_recorded() -> None:
    """A bound at or above the founding gain leaves a newborn's start untouched."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())

    state.admit(_fresh("newborn", born_at=_instant(BORN_AT), prior=PRIOR))

    assert state.bounds_reached() == ()
    assert dict(state.learned_state("newborn")) == dict(
        learning.cold_start({stream.FEATURE: PRIOR}, config.LearningConfig())
    )


# ---------------------------------------------------------------------------
# (c) ULP: a state held exactly on the weight-norm bound is admissible.
# ---------------------------------------------------------------------------


def _held_on_the_weight_bound() -> tuple[config.LearningConfig, list[dict[str, float]]]:
    """Every state the weight-norm hold produced over a walk, under a derived bound.

    The bound is half the smaller peak weight norm of a generous-bound pass,
    as EV-E70's weight half derives it, so this file chooses no number the
    stream then happens to cross. A state is collected after every outcome at
    which the hold acted on that agent, so each is a weight norm the hold has
    just landed on the bound.
    """
    generous = agent.PopulationState.founded(
        stream.two_agents(quick=0.9, slow=PARENT_FORGETS_AT),
        learning=config.LearningConfig(),
    )
    walked = stream.walk(generous, stream.windows())
    peak = min(max(trajectory.weight_norms) for trajectory in walked.values())
    tight = config.LearningConfig(weight_norm_bound=peak / 2.0)

    state = agent.PopulationState.founded(
        stream.two_agents(quick=0.9, slow=PARENT_FORGETS_AT), learning=tight
    )
    windows = stream.windows()
    held: list[dict[str, float]] = []
    for index, window in enumerate(windows[:-1]):
        state.step(window, stream.INSTRUMENT)
        for record in state.observe(
            windows[index + 1].as_of,
            dict.fromkeys(state.agent_ids(), stream.outcome_at(index)),
        ):
            if record.bound == config.WEIGHT_NORM_BOUND_KEY:
                held.append(dict(state.learned_state(record.agent_id)))
    return tight, held


@pytest.mark.parametrize("path", ["founded", "admit"])
def test_a_state_held_exactly_on_the_weight_bound_is_admitted(path: str) -> None:
    """A Lamarckian offspring of a parent the hold just put on the bound is admitted.

    The hold scales the accumulator so the weight norm lands on the bound, and
    the refusal compares the starting norm with that bound by ``>``. If the two
    computed the norm differently, a parent held one unit in the last place
    over would have its offspring refused for the hold's own rounding. Run at
    both birth paths, so the founding path observes the behaviour before the
    admission path exists.
    """
    tight, held = _held_on_the_weight_bound()
    bound = tight.weight_norm_bound
    assert held, f"no hold acted under the derived bound {bound}"
    norms = [
        _norm({k: v for k, v in s.items() if k.startswith(learning.WEIGHT_PREFIX)})
        for s in held
    ]
    assert all(norm == pytest.approx(bound, rel=1e-12) for norm in norms)
    over = [norm for norm in norms if norm > bound]

    born_at = _instant(stream.DECISION_POINTS - 1)
    births = tuple(
        _carrying(f"offspring-{index}", carried, born_at=born_at)
        for index, carried in enumerate(held)
    )
    try:
        if path == "founded":
            admitted = agent.PopulationState.founded(births, learning=tight)
        else:
            admitted = agent.PopulationState.founded(
                (stream.birth("parent", forgetting=PARENT_FORGETS_AT),),
                learning=tight,
            )
            for birth in births:
                admitted.admit(birth)
    except agent.StartingStateOutsideItsBound as refused:
        pytest.fail(
            f"{len(over)} of {len(held)} held states sit above the bound {bound!r} "
            f"(largest {max(over, default=bound)!r}, "
            f"{max(over, default=bound) - bound!r} over); refused: {refused}"
        )
    for birth, carried in zip(births, held, strict=True):
        assert dict(admitted.learned_state(birth.agent_id)) == carried


def _held_on_the_state_bound() -> tuple[config.LearningConfig, list[dict[str, float]]]:
    """Every state the gain hold produced, at founding and over a walk.

    The bound is half the founding gain of an agent founded under the shipped
    configuration, so it is met at founding and again while the cold-start
    evidence is still thin. A state is collected wherever the hold acted.
    """
    tight = _held_at_founding()
    state = agent.PopulationState.founded(
        stream.two_agents(quick=0.9, slow=PARENT_FORGETS_AT), learning=tight
    )
    held = [
        dict(state.learned_state(record.agent_id)) for record in state.bounds_reached()
    ]
    windows = stream.windows()
    for index, window in enumerate(windows[:-1]):
        state.step(window, stream.INSTRUMENT)
        for record in state.observe(
            windows[index + 1].as_of,
            dict.fromkeys(state.agent_ids(), stream.outcome_at(index)),
        ):
            if record.bound == config.STATE_BOUND_KEY:
                held.append(dict(state.learned_state(record.agent_id)))
    return tight, held


@pytest.mark.parametrize("path", ["founded", "admit"])
def test_a_state_whose_gain_was_held_on_the_bound_is_admitted(path: str) -> None:
    """The gain half of the same risk: an inherited gain the hold put on the
    state bound is not refused as outside it."""
    tight, held = _held_on_the_state_bound()
    bound = tight.state_bound
    assert held, f"no hold acted under the derived state bound {bound}"
    born_at = _instant(stream.DECISION_POINTS - 1)
    births = tuple(
        _carrying(f"offspring-{index}", carried, born_at=born_at)
        for index, carried in enumerate(held)
    )
    try:
        if path == "founded":
            admitted = agent.PopulationState.founded(births, learning=tight)
        else:
            admitted = agent.PopulationState.founded(
                (stream.birth("parent", forgetting=PARENT_FORGETS_AT),),
                learning=tight,
            )
            for birth in births:
                admitted.admit(birth)
    except agent.StartingStateOutsideItsBound as refused:
        pytest.fail(f"{len(held)} gain-held states, bound {bound!r}; {refused}")
    for birth, carried in zip(births, held, strict=True):
        assert dict(admitted.learned_state(birth.agent_id)) == carried


# ---------------------------------------------------------------------------
# (d) Every StartingStateOutsideItsBound refusal at admission is counted.
# ---------------------------------------------------------------------------


def test_every_start_refused_as_outside_its_bound_is_counted() -> None:
    """A prior at twice the weight bound and an inherited gain outside the state
    bound are each refused, counted, and leave the population as it was."""
    shipped = config.LearningConfig()
    state = _parent_walked_to(BORN_AT, shipped)
    born_at = _instant(BORN_AT)
    before = state.storage()
    assert state.admission_refusals() == 0

    with pytest.raises(agent.StartingStateOutsideItsBound):
        state.admit(
            _fresh("too-sure", born_at=born_at, prior=2.0 * shipped.weight_norm_bound)
        )
    assert state.admission_refusals() == 1

    carried = dict(state.learned_state("parent"))
    lost = {**carried, f"{learning.ENERGY_PREFIX}{stream.FEATURE}": float("nan")}
    with pytest.raises(agent.StartingStateOutsideItsBound):
        state.admit(_carrying("lost", lost, born_at=born_at))
    assert state.admission_refusals() == 2

    assert state.agent_ids() == ("parent",)
    assert state.storage() == before
    state.admit(_carrying("fits", carried, born_at=born_at))
    assert state.admission_refusals() == 2


# ---------------------------------------------------------------------------
# (e) A cull: out of the living set, the store not reallocated, lineage kept.
# ---------------------------------------------------------------------------


def test_a_cull_removes_the_agent_without_reallocating_and_keeps_its_lineage() -> None:
    """The dead are gone from the living set and still answer for where they came from."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    born_at = _instant(BORN_AT)
    state.admit(_fresh("first", born_at=born_at))
    state.admit(_fresh("second", born_at=born_at, parents=("parent", "first")))
    survivors = {
        agent_id: (
            state.genome(agent_id),
            dict(state.learned_state(agent_id)),
            state.history(agent_id),
            state.lineage(agent_id),
        )
        for agent_id in ("parent", "second")
    }
    before = state.storage()
    died_at = _instant(BORN_AT + 1)

    closed = state.cull("first", died_at=died_at)

    after = state.storage()
    assert after.agents == before.agents - 1
    assert after.capacity == before.capacity
    assert after.reallocations == before.reallocations
    assert after.contiguous
    assert state.agent_ids() == ("parent", "second")
    with pytest.raises(KeyError, match="is not alive"):
        state.genome("first")
    assert closed == state.lineage("first")
    assert closed.parents == ("parent",)
    assert closed.born_at == born_at
    assert closed.died_at == died_at
    assert state.departed() == (closed,)
    for agent_id, (genome, learned, history, lineage) in survivors.items():
        assert state.genome(agent_id) == genome
        assert dict(state.learned_state(agent_id)) == learned
        assert state.history(agent_id) == history
        assert state.lineage(agent_id) == lineage


def test_a_birth_after_a_cull_takes_the_freed_room_rather_than_a_new_store() -> None:
    """Admission grows the store only when there is no room; a cull makes room."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    born_at = _instant(BORN_AT)
    state.admit(_fresh("first", born_at=born_at))
    grown = state.storage()
    state.cull("first", died_at=_instant(BORN_AT + 1))

    state.admit(_fresh("second", born_at=_instant(BORN_AT + 1)))

    assert state.storage().capacity == grown.capacity
    assert state.storage().reallocations == grown.reallocations


def test_a_cull_or_a_birth_is_refused_where_it_could_not_be_one() -> None:
    """Who is alive, and when, has to make sense before anything moves."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    born_at = _instant(BORN_AT)
    state.admit(_fresh("child", born_at=born_at))

    with pytest.raises(ValueError, match="already"):
        state.admit(_fresh("child", born_at=born_at))
    with pytest.raises(ValueError, match="instant"):
        state.admit(replace(_fresh("undated", born_at=born_at), born_at=None))
    with pytest.raises(ValueError, match="before"):
        state.cull("child", died_at=_instant(BORN_AT - 1))
    with pytest.raises(KeyError, match="is not alive"):
        state.cull("nobody", died_at=born_at)

    state.cull("child", died_at=born_at)
    with pytest.raises(KeyError, match="is not alive"):
        state.cull("child", died_at=born_at)
    with pytest.raises(ValueError, match="already"):
        state.admit(_fresh("child", born_at=born_at))
    with pytest.raises(ValueError, match="last"):
        state.cull("parent", died_at=born_at)


def test_no_birth_or_death_lands_between_a_reading_and_its_outcome() -> None:
    """The outcome is owed to exactly the agents that formed the reading."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    state.admit(_fresh("child", born_at=_instant(BORN_AT)))
    state.step(stream.windows()[BORN_AT], stream.INSTRUMENT)

    with pytest.raises(agent.ReadingAwaitsItsOutcome):
        state.admit(_fresh("late", born_at=_instant(BORN_AT)))
    with pytest.raises(agent.ReadingAwaitsItsOutcome):
        state.cull("child", died_at=_instant(BORN_AT))
    assert state.agent_ids() == ("parent", "child")


def test_lineage_of_someone_who_never_lived_is_refused_by_name() -> None:
    """A lookup that missed says who is alive and who has died."""
    state = _parent_walked_to(1, config.LearningConfig())
    with pytest.raises(KeyError, match="never lived"):
        state.lineage("nobody")


def test_an_admitted_genome_that_does_not_fit_is_refused_before_anything_moves() -> (
    None
):
    """Another gene vocabulary, or a rate outside the range, is not admitted."""
    learns = config.LearningConfig()
    state = _parent_walked_to(1, learns)
    born_at = _instant(1)
    before = state.storage()
    fresh = _fresh("other", born_at=born_at)
    other_schema = replace(
        fresh, genome=replace(fresh.genome, schema_version="another-genes-1")
    )
    too_slow = replace(
        fresh,
        genome=replace(
            fresh.genome,
            genes=MappingProxyType(
                {**fresh.genome.genes, agent.FORGETTING_FACTOR: 1.0}
            ),
        ),
    )

    with pytest.raises(ValueError, match="gene schema"):
        state.admit(other_schema)
    with pytest.raises(agent.ForgettingFactorOutsideItsRange):
        state.admit(too_slow)

    assert state.storage() == before
    assert state.agent_ids() == ("parent",)


def test_a_birth_carrying_the_books_identity_is_refused_and_nothing_moves() -> None:
    """The aggregate book's account name is reserved, so no birth mid-walk takes it."""
    state = _parent_walked_to(1, config.LearningConfig())
    ids_before = state.agent_ids()
    refusals_before = state.admission_refusals()

    with pytest.raises(ValueError, match="reserved for the aggregate book"):
        state.admit(_fresh(ecology.BOOK_ACCOUNT_ID, born_at=_instant(1)))

    assert state.agent_ids() == ids_before
    assert state.admission_refusals() == refusals_before


def test_a_population_that_does_not_learn_admits_and_culls_too() -> None:
    """Holding agents is a capability apart from learning; turnover is part of it."""
    state = agent.PopulationState.founded(
        (stream.birth("parent", forgetting=PARENT_FORGETS_AT),)
    )
    born_at = _instant(0)

    state.admit(_fresh("child", born_at=born_at))
    assert state.lineage("child").learned_state_disposition is (
        agent.LearnedStateDisposition.NOTHING_CARRIED
    )
    state.cull("parent", died_at=_instant(1))
    assert state.agent_ids() == ("child",)


# ---------------------------------------------------------------------------
# What a birth records about how it was made.
# ---------------------------------------------------------------------------


def test_a_birth_records_its_variation_its_seed_and_what_it_carried() -> None:
    """Lineage says how the agent was made, not only from whom."""
    state = _parent_walked_to(BORN_AT, config.LearningConfig())
    carried = dict(state.learned_state("parent"))
    born_at = _instant(BORN_AT)

    state.admit(
        replace(
            _carrying("lamarckian", carried, born_at=born_at),
            mutation_scale=0.125,
            spawn_seed=17,
        )
    )
    state.admit(replace(_fresh("baldwinian", born_at=born_at), spawn_seed=18))

    lamarckian = state.lineage("lamarckian")
    assert lamarckian.mutation_scale == 0.125
    assert lamarckian.spawn_seed == 17
    assert lamarckian.learned_state_disposition is agent.LearnedStateDisposition.CARRIED
    baldwinian = state.lineage("baldwinian")
    assert baldwinian.mutation_scale is None
    assert baldwinian.spawn_seed == 18
    assert (
        baldwinian.learned_state_disposition
        is agent.LearnedStateDisposition.NOTHING_CARRIED
    )
    assert state.lineage("parent").learned_state_disposition is (
        agent.LearnedStateDisposition.NOTHING_CARRIED
    )
    assert state.lineage("parent").died_at is None


def test_a_death_instant_states_its_zone() -> None:
    """A death instant is held in UTC, and one without a zone is refused."""
    elsewhere = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))
    lineage = agent.Lineage(agent_id="one", parents=(), born_at=None, died_at=elsewhere)
    assert lineage.died_at == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="died_at"):
        agent.Lineage(
            agent_id="one",
            parents=(),
            born_at=None,
            died_at=datetime(2024, 1, 1),  # noqa: DTZ001
        )


# ---------------------------------------------------------------------------
# (f) A snapshot after admissions and culls restores the same population.
# ---------------------------------------------------------------------------


def _turned_over() -> agent.PopulationState:
    """A walking population that has admitted, refused and culled."""
    learns = _held_at_founding()
    state = _parent_walked_to(BORN_AT, learns)
    carried = dict(state.learned_state("parent"))
    born_at = _instant(BORN_AT)
    state.admit(
        replace(
            _carrying("carries", carried, born_at=born_at),
            mutation_scale=0.25,
            spawn_seed=3,
        )
    )
    state.admit(_fresh("fresh", born_at=born_at, prior=PRIOR))
    state.admit(_fresh("doomed", born_at=born_at, prior=-PRIOR))
    with pytest.raises(agent.StartingStateOutsideItsBound):
        state.admit(
            _fresh("too-sure", born_at=born_at, prior=2.0 * learns.weight_norm_bound)
        )
    _walk(state, stream.windows()[BORN_AT : BORN_AT + 6], first=BORN_AT)
    state.cull("doomed", died_at=_instant(BORN_AT + 5))
    return state


def test_a_snapshot_after_turnover_restores_the_same_population() -> None:
    """The living, the dead, the refusals and the holds all come back."""
    live = _turned_over()
    put_away = live.snapshot()

    restored = agent.PopulationState.restored(put_away)

    assert restored.agent_ids() == live.agent_ids()
    for agent_id in live.agent_ids():
        assert restored.genome(agent_id) == live.genome(agent_id)
        assert dict(restored.learned_state(agent_id)) == dict(
            live.learned_state(agent_id)
        )
        assert restored.history(agent_id) == live.history(agent_id)
        assert restored.lineage(agent_id) == live.lineage(agent_id)
    assert restored.departed() == live.departed()
    assert restored.lineage("doomed") == live.lineage("doomed")
    assert restored.admission_refusals() == live.admission_refusals() == 1
    assert restored.bounds_reached() == live.bounds_reached()

    rest = stream.windows()[BORN_AT + 5 : BORN_AT + 60]
    assert _walk(restored, rest, first=BORN_AT + 5) == _walk(
        live, rest, first=BORN_AT + 5
    )


def test_a_snapshot_whose_dead_are_not_dead_is_refused() -> None:
    """A departed lineage must be of someone dead, once, and not of the living."""
    put_away = _turned_over().snapshot()
    (departed,) = put_away.departed
    living = put_away.agents[0]

    for spoiled in (
        replace(put_away, departed=(departed, departed)),
        replace(put_away, departed=(replace(departed, agent_id=living.agent_id),)),
        replace(put_away, departed=(replace(departed, died_at=None),)),
        replace(put_away, departed=(replace(departed, death_kind=None),)),
        replace(put_away, admission_refusals=-1),
    ):
        with pytest.raises(ValueError):
            agent.PopulationState.restored(spoiled)
