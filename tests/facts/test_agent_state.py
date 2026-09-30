"""Evidence that an agent's two parts come apart, and survive being put away.

Two claims, and they are the same claim seen from two sides. An agent is not one
blob of state: what it inherited and what it learned are different things with
different lifetimes, and the experiment this platform exists to run — which of
the two crosses a birth boundary — cannot even be *stated* against a
representation that holds them as one. So the first claim is that the two are
separately addressable: reachable apart, writable apart, and carried apart by a
birth.

The second is that putting an agent away and taking it out again changes
nothing about what it goes on to do. That is deliberately a claim about
*subsequent behaviour* rather than about fields: a snapshot whose numbers all
compare equal and whose restored agent then diverges on the next decision point
has preserved a record of an agent, not the agent. So the check steps the
continuing population and the restored one over the same decision points and
compares what each one did, not what each one holds.

**What the behavioural comparison can and cannot catch, stated at its true
width.** A decision point is answered from the heritable genes, *what the agent
has learned*, the position it was holding when it formed the wish, and the
feature vocabulary it was born under. A snapshot that lost or garbled any of
the first three is caught behaviourally, and the proof that each is caught is
run below rather than argued; the fourth is not a divergence but a refusal, and
the refusing is checked where the refusal lives rather than here. What is left
outside the trajectory is the outcome history and the lineage: nothing steps on
either, so a snapshot that dropped one would produce an identical trajectory
and is caught here only by the round-trip assertions, which are the weaker kind
of evidence.

The learned half was owed until the decision rule read what an agent had
learned, and it is owed no longer. The wish is now formed from the learned
weights, so a restore that rebuilt them from the cold start diverges on the
next decision point rather than quietly agreeing — and that loss is seeded
below and required to show.

Nothing here is market data. The decision points are the declared arithmetic
ramp the null bar source generates; every number computed from them is a number
about the ramp and says nothing about any instrument. The claims are structural
and about the representation, which is why they live in this repository at all —
the ones that need real tape live beside it, in the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from types import MappingProxyType

from liq.evolution.ecology import Genome, adapters, agent, config, learning

#: Vocabulary the genes below belong to.
GENE_SCHEMA_VERSION = "state-genes-1"

#: Vocabulary the feature names below belong to. The decision points are
#: stamped with it too, so no agent here is evaluated under a vocabulary it was
#: not born under — which is a separate claim, checked elsewhere.
FEATURE_SCHEMA_VERSION = "state-features-1"

#: Version of the decision model the agents below were born under.
MODEL_VERSION = "state-model-1"

#: Vocabulary the behaviour descriptors below belong to.
DESCRIPTOR_SCHEMA_VERSION = "state-descriptors-1"

#: The two features the declared ramp broadcasts: a level that climbs by one a
#: bar, and the count of decision points so far.
LEVEL = "level"
STEP = "step"

#: One column of what an agent here has learned — the weight its update holds
#: for the switched-on feature. Named so a learned value could never be
#: mistaken for a gene, which is the point of holding the two apart, and taken
#: from the update's own columns because a learning population is born
#: declaring exactly those and nothing else.
OBSERVED = f"{learning.WEIGHT_PREFIX}{STEP}"

#: Who is alive in every population built here.
FIRST = "first"
SECOND = "second"

#: The prior each of them starts its learned weight for the switched-on
#: feature from — its ``weight.`` gene. Non-zero and different, so a restore
#: that re-applied the prior instead of reading what was learned would put
#: each agent somewhere it was not; that each differs from what the agent has
#: learned by the snapshot point is asserted below rather than assumed.
FIRST_PRIOR = -0.1
SECOND_PRIOR = 0.25

#: How fast each of them discards what it learned. Both inside the range the
#: learning configuration declares, and different from each other, so the two
#: agents' learned trajectories are not the same sequence by construction.
FIRST_FORGETS_AT = 0.95
SECOND_FORGETS_AT = 0.99

#: The features the genomes above switch on and weight, in the order the
#: population lays them out. Named so the cold start a lossy restore is built
#: from is each agent's own rather than one this module guesses at.
_FEATURES = (LEVEL, STEP)

#: How long after a decision point the outcome of holding what was wanted
#: there becomes known. One bar of the declared ramp: the wish is acted on over
#: the bar that follows it, so what it earned cannot be known before that bar
#: has finished.
OUTCOME_KNOWN_AFTER = timedelta(minutes=1)

#: How many decision points the ramp is walked over, and how many of them share
#: a walk-forward segment. Long enough that the second half of the walk — the
#: part a restore has to reproduce — carries several turns of the relationship
#: below rather than one.
DECISION_POINTS = 40
SEGMENT_LENGTH = 14

#: The declared outcome stream both agents are shown, and how often it turns
#: over. What holding a reading earned is ±1 in blocks, so an agent that learns
#: is pulled one way and then the other and its wish has to follow; a constant
#: outcome would leave a trajectory an identity could be asserted over without
#: saying anything.
OUTCOME_SWING = 1.0
OUTCOME_TURNS_EVERY = 5

#: How many decision points the walk covers before the population is put away.
#: Roughly half, and deliberately not exactly half: the snapshot has to be
#: taken at a point where the agents are *holding* something, or the seeded
#: loss that zeroes the position below would be zeroing a position nobody held
#: and would prove nothing. That the position is non-zero there is asserted
#: rather than assumed, so a later change to this stream cannot quietly empty
#: the check.
SNAPSHOT_AFTER = 18


def _birth(
    agent_id: str,
    *,
    step_weight: float,
    entry: float,
    forgetting: float,
) -> agent.AgentBirth:
    """One agent entering the population, with both its parts stated apart."""
    return agent.AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=MappingProxyType(
                {
                    f"{agent.MASK_PREFIX}{LEVEL}": 0.0,
                    f"{agent.WEIGHT_PREFIX}{LEVEL}": 0.0,
                    f"{agent.MASK_PREFIX}{STEP}": 1.0,
                    f"{agent.WEIGHT_PREFIX}{STEP}": step_weight,
                    agent.ENTRY_THRESHOLD: entry,
                    agent.FORGETTING_FACTOR: forgetting,
                }
            ),
            schema_version=GENE_SCHEMA_VERSION,
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def _founded(learns: config.LearningConfig | None = None) -> agent.PopulationState:
    """Two agents whose wishes differ, and differ over the ramp rather than once.

    The first wants exposure while the ramp is young and gives it up as the
    count climbs; the second does the opposite. Neither trajectory is constant,
    so an identity asserted over them is a claim about a sequence rather than
    about one value repeated.
    """
    return agent.PopulationState.founded(
        (
            _birth(
                FIRST,
                step_weight=FIRST_PRIOR,
                entry=0.0,
                forgetting=FIRST_FORGETS_AT,
            ),
            _birth(
                SECOND,
                step_weight=SECOND_PRIOR,
                entry=0.0,
                forgetting=SECOND_FORGETS_AT,
            ),
        ),
        learning=config.LearningConfig() if learns is None else learns,
    )


def _population(state: agent.PopulationState) -> adapters.ArrayGenomePopulation:
    """The population, reached through the capability the ecology keeps it by."""
    return adapters.ArrayGenomePopulation(
        state=state, descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION
    )


def _decision_points() -> tuple[agent.BarWindow, ...]:
    """The declared ramp's decision points, held so both walks get the same ones."""
    return tuple(
        adapters.NullBarSource(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            window_count=DECISION_POINTS,
            segment_length=SEGMENT_LENGTH,
        ).windows()
    )


def _earned(index: int) -> float:
    """What holding a reading at decision point ``index`` turned out to earn.

    Declared here rather than derived from what the agents did, and it turns
    over: an outcome that never changed sign would let an agent converge once
    and then wish the same thing forever, which is exactly the trajectory an
    identity check cannot say anything with.
    """
    turning = (index // OUTCOME_TURNS_EVERY) % 2 == 0
    return OUTCOME_SWING if turning else -OUTCOME_SWING


@dataclass(frozen=True)
class Did:
    """What the population did at one decision point, on both of its outputs.

    Two things, because the population now has two. ``step`` is the wish each
    agent formed and the position it held when it formed it; ``learned`` is
    what each agent's online update had arrived at once the outcome of that
    wish was known. Comparing only the first would leave the learned half
    unwatched, which is exactly the gap this pair closes.

    Attributes:
        step: The wishes and the positions behind them.
        learned: Each agent's learned weights, in the population's order.
    """

    step: agent.PopulationStep
    learned: tuple[tuple[float, ...], ...]


def _walk(
    state: agent.PopulationState,
    windows: tuple[agent.BarWindow, ...],
    instrument: str,
    *,
    from_index: int = 0,
) -> tuple[Did, ...]:
    """Step ``state`` over ``windows``, settling, learning and being scored.

    What is handed back after each decision point is deliberately not nothing:
    an agent reaches the exposure it wanted, records the outcome, and — a bar
    later, which is when it could be known — is shown what holding that
    exposure earned, which moves every column of what it has learned. A walk that left every mutable part
    untouched would make the restore below trivially faithful.

    ``from_index`` is where these windows sit in the whole ramp, so that a walk
    resumed at the midpoint is shown the same outcomes the walk it is being
    compared with was shown. Without it the two halves would be fed the stream
    from its beginning twice and the comparison would be of two different
    experiments.
    """
    trajectory = []
    for offset, window in enumerate(windows):
        step = state.step(window, instrument)
        for agent_id, wanted in zip(step.agent_ids, step.wanted, strict=True):
            state.hold(agent_id, wanted)
            state.record_outcome(agent_id, wanted)
        earned = dict.fromkeys(step.agent_ids, _earned(from_index + offset))
        state.observe(window.as_of + OUTCOME_KNOWN_AFTER, earned)
        trajectory.append(
            Did(
                step=step,
                learned=tuple(
                    tuple(state.learned_weights(agent_id).values())
                    for agent_id in step.agent_ids
                ),
            )
        )
    return tuple(trajectory)


def test_genome_and_learned_state_separable() -> None:
    """The heritable part and the learned part are reached, written and born apart.

    Four things have to hold, and the last two are what stop the claim from
    being satisfied by two getters over one combined structure: the names do not
    overlap, learning leaves every gene as it was, one agent's learned state can
    be dropped onto another without its genes following, and what a birth hands
    back is the heritable part alone.
    """
    state = _founded()
    population = _population(state)
    first, second = population.agent_ids()

    inherited = dict(population.genome(first).genes)
    learned = dict(population.learned_state(first))
    assert inherited
    assert learned
    assert set(inherited) & set(learned) == set()

    state.learn(first, {OBSERVED: 42.0})
    assert dict(population.genome(first).genes) == inherited
    assert dict(population.learned_state(first)) == {**learned, OBSERVED: 42.0}

    state.learn(first, population.learned_state(second))
    assert dict(population.learned_state(first)) == dict(
        population.learned_state(second)
    )
    assert dict(population.genome(first).genes) == inherited
    assert dict(population.genome(first).genes) != dict(population.genome(second).genes)

    offspring = population.spawn(parents=(first, second), seed=17)
    assert set(offspring.genes) == set(inherited)
    assert OBSERVED not in offspring.genes


def test_a_learned_state_crosses_a_birth_whole_or_not_at_all() -> None:
    """What a birth carries of the learned part is what the agent starts with.

    The separability claim is given its reason by the birth boundary, so the
    fate of a learned state at a birth is pinned here rather than left to the
    representation. A birth carrying the whole of what its parent learned is
    founded holding exactly that; a sibling of the same genome carrying nothing
    is not; and a birth carrying part of it, or part of it and a name nobody
    declared, is refused rather than resolved by whichever side of a merge wins.
    """
    windows = _decision_points()
    instrument = next(iter(windows[0].bars))
    parent = _founded()
    _walk(parent, windows[:SNAPSHOT_AFTER], instrument)
    carried = dict(parent.learned_state(FIRST))
    genome = parent.genome(FIRST)

    def born(agent_id: str, learned_state: dict[str, float]) -> agent.AgentBirth:
        return agent.AgentBirth(
            agent_id=agent_id,
            genome=genome,
            learned_state=MappingProxyType(learned_state),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_version=MODEL_VERSION,
            parents=(FIRST,),
        )

    offspring = agent.PopulationState.founded(
        (born("carries", carried), born("carries-nothing", {})),
        learning=config.LearningConfig(),
    )
    assert dict(offspring.learned_state("carries")) == carried
    assert dict(offspring.learned_state("carries-nothing")) != carried

    partial = {
        name: value
        for name, value in carried.items()
        if not name.startswith(learning.DRIFT_PREFIX)
    }
    conflicting = {**carried, "learned.drift.step": 0.0}
    for refused in (partial, conflicting):
        try:
            admitted = agent.PopulationState.founded(
                (born("refused", refused),), learning=config.LearningConfig()
            )
        except ValueError:
            continue
        raise AssertionError(
            f"a birth carrying {sorted(refused)} was admitted holding "
            f"{dict(admitted.learned_state('refused'))}"
        )


def test_snapshot_restore_preserves_behaviour() -> None:
    """A population put away mid-walk and taken out again does the same next.

    The comparison is over what the two populations *did* across the rest of the
    walk — the wish each agent formed at each decision point, the position it
    was holding when it formed it, and what its online update had arrived at
    once the outcome of that wish was known — not over the bytes a snapshot
    serialises to and not over a single value repeated: both halves change
    partway through, so an identity over them is a claim about a sequence.

    The round-trip assertions afterwards are the weaker, second kind of
    evidence. They now cover less than they used to, which is the point: the
    learned state has moved out from behind them into the trajectory above, and
    what is left behind them is the outcome history and the lineage. See the
    module docstring.
    """
    windows = _decision_points()
    instrument = next(iter(windows[0].bars))

    live = _founded()
    _walk(live, windows[:SNAPSHOT_AFTER], instrument)
    put_away = live.snapshot()

    continued = _walk(
        live, windows[SNAPSHOT_AFTER:], instrument, from_index=SNAPSHOT_AFTER
    )
    restored = agent.PopulationState.restored(put_away)
    resumed = _walk(
        restored, windows[SNAPSHOT_AFTER:], instrument, from_index=SNAPSHOT_AFTER
    )

    assert resumed == continued
    assert len({did.step.wanted for did in continued}) > 1
    assert len({did.learned for did in continued}) > 1

    for agent_id in live.agent_ids():
        assert restored.genome(agent_id) == live.genome(agent_id)
        assert dict(restored.learned_state(agent_id)) == dict(
            live.learned_state(agent_id)
        )
        assert restored.history(agent_id) == live.history(agent_id)
        assert restored.lineage(agent_id) == live.lineage(agent_id)
        assert restored.versions(agent_id) == live.versions(agent_id)
        assert restored.realised(agent_id) == live.realised(agent_id)

    # The intent is carried too, and compared on its own: the next decision
    # point overwrites it, so no trajectory could see it lost, and the
    # distinction between what an agent wanted and what it holds is the thing
    # a restore must not flatten.
    restored_at_the_snapshot = agent.PopulationState.restored(put_away)
    wanted_there = {held.agent_id: held.intended for held in put_away.agents}
    assert set(wanted_there.values()) != {0.0}, (
        "every agent wanted nothing where the population was put away, so a "
        "snapshot that dropped the intent would drop nothing"
    )
    for agent_id, intended in wanted_there.items():
        assert restored_at_the_snapshot.intended(agent_id) == intended


def _stepped(
    state: agent.PopulationState, window: agent.BarWindow, instrument: str
) -> agent.PopulationStep:
    """Answer one decision point the way `_walk` does, and stop before its outcome."""
    step = state.step(window, instrument)
    for agent_id, wanted in zip(step.agent_ids, step.wanted, strict=True):
        state.hold(agent_id, wanted)
        state.record_outcome(agent_id, wanted)
    return step


def test_a_snapshot_between_a_decision_and_its_outcome_resumes_the_same() -> None:
    """Put away with an outcome still owed, a population takes that outcome and goes on.

    The unsettled case, which the check above does not reach: the population
    has answered a decision point and the outcome of what it wanted there is
    not known yet. That is the one instant a resume cannot rebuild from the
    agents alone, because the outcome has to be paired with the reading it
    belongs to and that reading was formed *before* the snapshot. So the
    restored population has to accept the same outcome the live one does,
    learn the same thing from it, and go on to do the same over the rest of
    the walk; and what each wanted and holds has to have come back too.
    """
    windows = _decision_points()
    instrument = next(iter(windows[0].bars))
    at = SNAPSHOT_AFTER

    live = _founded()
    _walk(live, windows[:at], instrument)
    _stepped(live, windows[at], instrument)
    put_away = live.snapshot()
    restored = agent.PopulationState.restored(put_away)

    for agent_id in live.agent_ids():
        assert restored.standardised(agent_id) == live.standardised(agent_id)
        assert restored.intended(agent_id) == live.intended(agent_id)
        assert restored.realised(agent_id) == live.realised(agent_id)

    owed = dict.fromkeys(live.agent_ids(), _earned(at))
    settled_at = windows[at].as_of + OUTCOME_KNOWN_AFTER
    for state in (live, restored):
        state.observe(settled_at, owed)
    assert {
        agent_id: dict(restored.learned_state(agent_id))
        for agent_id in restored.agent_ids()
    } == {agent_id: dict(live.learned_state(agent_id)) for agent_id in live.agent_ids()}

    continued = _walk(live, windows[at + 1 :], instrument, from_index=at + 1)
    resumed = _walk(restored, windows[at + 1 :], instrument, from_index=at + 1)
    assert resumed == continued
    assert len({did.step.wanted for did in continued}) > 1


def test_a_restore_keeps_the_record_of_every_bound_that_acted() -> None:
    """What a population recorded about its bounds is part of what it was.

    A run held under a bound says so, and a resume that forgot saying so would
    describe the same run as one that was never held. So a population whose
    bound acted before it was put away comes back carrying every record, in
    order, and goes on appending to them exactly as the one that was never put
    away does.
    """
    windows = _decision_points()
    instrument = next(iter(windows[0].bars))
    held_tightly = config.LearningConfig(state_bound=0.05)

    live = _founded(held_tightly)
    _walk(live, windows[:SNAPSHOT_AFTER], instrument)
    recorded = live.bounds_reached()
    assert recorded, (
        f"a state bound of {held_tightly.state_bound}, below the gain an agent "
        "starts with, was never reached, so there is nothing to keep"
    )
    restored = agent.PopulationState.restored(live.snapshot())
    assert restored.bounds_reached() == recorded

    later = _walk(live, windows[SNAPSHOT_AFTER:], instrument, from_index=SNAPSHOT_AFTER)
    again = _walk(
        restored, windows[SNAPSHOT_AFTER:], instrument, from_index=SNAPSHOT_AFTER
    )
    assert again == later
    assert restored.bounds_reached() == live.bounds_reached()


def test_a_snapshot_that_lost_something_would_be_caught() -> None:
    """The identity above is asserted over something that can fail.

    Three parts a snapshot could plausibly drop are dropped on purpose here —
    one heritable, one the position the agent was holding, one the whole of
    what it had learned — and each has to change what the restored population
    goes on to do. Without this, the check above would read exactly the same if
    the trajectory were a constant, if the restore silently reused the live
    population, or if stepping and learning consulted nothing the snapshot
    carries.

    The third is the one that was owed. Until an online update existed nothing
    stepped on the learned state, so a snapshot that dropped it produced an
    identical trajectory and was caught only by the round-trip comparison —
    the weaker kind of evidence. It is now caught behaviourally, and the proof
    is run here rather than argued.
    """
    windows = _decision_points()
    instrument = next(iter(windows[0].bars))

    live = _founded()
    _walk(live, windows[:SNAPSHOT_AFTER], instrument)
    put_away = live.snapshot()
    assert {held.realised for held in put_away.agents} != {0.0}, (
        "the population is flat where it is put away, so restoring it with "
        "every position zeroed would drop nothing and the seeded loss below "
        "would pass against a resume that read no position at all"
    )
    assert {tuple(held.learned_state.items()) for held in put_away.agents}, (
        "the snapshot carries no learned state, so the third seeded loss would "
        "be a loss of nothing"
    )
    for held in put_away.agents:
        prior = held.genome.genes[f"{agent.WEIGHT_PREFIX}{STEP}"]
        assert prior != 0.0 and held.learned_state[OBSERVED] != prior, (
            f"{held.agent_id!r} holds {held.learned_state[OBSERVED]} where its "
            f"prior is {prior}, so a restore that re-applied the prior would "
            "put it nowhere new and the third seeded loss would prove nothing"
        )
    continued = _walk(
        live, windows[SNAPSHOT_AFTER:], instrument, from_index=SNAPSHOT_AFTER
    )

    faithful = _walk(
        agent.PopulationState.restored(put_away),
        windows[SNAPSHOT_AFTER:],
        instrument,
        from_index=SNAPSHOT_AFTER,
    )
    assert faithful == continued

    for lossy in (
        _with_a_gene_lost(put_away),
        _with_the_position_lost(put_away),
        _with_the_learned_state_lost(put_away),
    ):
        diverged = _walk(
            agent.PopulationState.restored(lossy),
            windows[SNAPSHOT_AFTER:],
            instrument,
            from_index=SNAPSHOT_AFTER,
        )
        assert diverged != continued


def _with_the_learned_state_lost(
    put_away: agent.PopulationSnapshot,
) -> agent.PopulationSnapshot:
    """The same snapshot with every agent restored as though it had learned nothing.

    Each learned value goes back to where that agent's own genome starts it —
    its ``weight.`` genes as its prior — which is the most plausible way this
    loss would actually happen: a resume that re-applied the prior, rebuilding
    the learned columns from the cold start instead of reading them.
    """
    configured = put_away.learning or config.LearningConfig()
    return replace(
        put_away,
        agents=tuple(
            replace(
                held,
                learned_state=learning.cold_start(
                    {
                        name: held.genome.genes[f"{agent.WEIGHT_PREFIX}{name}"]
                        for name in _FEATURES
                    },
                    configured,
                ),
            )
            for held in put_away.agents
        ),
    )


def _with_a_gene_lost(put_away: agent.PopulationSnapshot) -> agent.PopulationSnapshot:
    """The same snapshot with one agent's entry gene no longer what it was."""
    first, *rest = put_away.agents
    genes = dict(first.genome.genes)
    genes[agent.ENTRY_THRESHOLD] = genes[agent.ENTRY_THRESHOLD] + 1_000.0
    return replace(
        put_away,
        agents=(
            replace(first, genome=replace(first.genome, genes=genes)),
            *rest,
        ),
    )


def _with_the_position_lost(
    put_away: agent.PopulationSnapshot,
) -> agent.PopulationSnapshot:
    """The same snapshot with every agent restored flat, as if nothing was held."""
    return replace(
        put_away,
        agents=tuple(replace(held, realised=0.0) for held in put_away.agents),
    )
