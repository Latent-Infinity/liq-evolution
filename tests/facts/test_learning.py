"""Evidence about the online update: the gene that drives it, and its limits.

Four claims, and they divide in two. The first pair is about what the update
*is*: the rate at which an agent forgets is inherited, so two agents that differ
in nothing else adapt at measurably different speeds; and an agent born without
an inherited learned state starts from the weights its own genome's ``weight.``
genes declare, not from one number every agent shares and not from whatever the
arithmetic happened to start at. The second pair is about what the update
may not do: its internal state stays inside a declared bound across a whole
pass, including on an input carrying no information at all, and a run that
reaches that bound says so rather than carrying on with a state nobody is
watching.

**None of this is market data.** The decision points are the declared
arithmetic ramp, and the step in the relationship is declared here rather than
located in a tape — the reason, and what is lost by it, is written at the top of
`tests/support/level_shift_stream.py`. Every number below is a number about
that stream. The claims are about the estimator's response to what it was
shown, which is why they live in this repository at all; a claim about an
instrument would need the tape and would live beside it, in the consumer.

**What the bound checks can and cannot catch.** The bound is read from the
configuration the population was built with, never written down here, so a
check that passed because somebody widened a number in a test is not reachable.
The pass is sampled at every decision point rather than at the end, and the
report names the bar a bound would first be reached at, so "it was fine
afterwards" cannot stand in for "it was fine throughout". What none of it can
catch is a bound so generous that no candidate could ever exceed it — so the
companion below runs the same check against an estimator built without the
regularisation that holds the bound, and requires it to go red.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from liq.evolution.ecology import agent, config
from tests.support import level_shift_stream as stream
from tests.support import state_norm_probe as probe

#: A forgetting factor near the fast end of the declared range, and one near the
#: slow end. Both are inside the range the configuration declares; the check
#: asserts that rather than assuming it.
QUICK = 0.95
SLOW = 0.995

#: Priors a population carries in its ``weight.`` genes: distinct and non-zero,
#: so a population that started everybody at one value — any value, zero
#: included — is told apart from one that started each agent at its own.
PRIORS = {"leans-short": -0.5, "leans-a-little": 0.25, "leans-long": 0.75}

#: The pair a prior's reach into the wish is exhibited over: one gene apart,
#: either side of nothing, so whichever sign the standardised reading carries,
#: one of them clears the entry gene and the other does not.
LEANS = 0.5

#: The first decision point at which a prior can reach a wish. The standardiser
#: scales by prior-bar moments: with none behind the first decision point and
#: one bar behind the second there is no spread, so the scaled reading — and
#: with it every learned weight's contribution — is exactly zero at both.
PRIOR_REACHES_A_WISH_AT = 2

#: The weight gene of the stream's one feature, read by name.
WEIGHT_GENE = f"{agent.WEIGHT_PREFIX}{stream.FEATURE}"

#: Identity the configurations below are named under. A run's configuration is
#: what provenance records, so it is the configuration these checks read their
#: numbers out of — a quantity that is not one of its keys is not a quantity a
#: later reader could recover.
RUN_ID = "ecology-learning-evidence"


def _default_run() -> config.EcologyConfig:
    """A run named and otherwise left at what the schema itself declares.

    The bounds the checks below read come from here rather than from a
    configuration a check assembled, because the claim is that the quantity is
    *declared* — a number a test had to supply would be a number the run did
    not have to.
    """
    return config.EcologyConfig(run_id=RUN_ID)


def _run(**overrides: object) -> config.EcologyConfig:
    """The same run with one or more learning quantities set to a named value."""
    return config.EcologyConfig(
        run_id=RUN_ID,
        learning=config.LearningConfig(**overrides),  # ty: ignore[invalid-argument-type]
    )


def _declared(run: config.EcologyConfig | None = None) -> Mapping[str, object]:
    """Every value a run configuration declares, by the name provenance carries."""
    return probe.declared_configuration(_default_run() if run is None else run)


def _founded(
    *,
    quick: float = QUICK,
    slow: float = SLOW,
    run: config.EcologyConfig | None = None,
) -> agent.PopulationState:
    """Two agents differing only in their forgetting gene, ready to be shown a stream."""
    return agent.PopulationState.founded(
        stream.two_agents(quick=quick, slow=slow),
        learning=(_default_run() if run is None else run).learning,
    )


def test_forgetting_factor_changes_convergence() -> None:
    """Two agents differing only in how fast they forget adapt at different speeds.

    The direction is asserted, not merely the difference: the agent whose gene
    says to forget faster reaches the reversed relationship *sooner*, and is
    closer to it at the end of the pass. A check that only asked whether the two
    trajectories differed would pass on an implementation that had the gene
    backwards.

    The claim is bounded to what was run. It is about the configured update rule
    and this one declared step in the relationship — not about learning rates in
    general, not about any other estimator family, and not about any market.
    """
    state = _founded()
    walked = stream.walk(state, stream.windows())

    quick = walked[stream.QUICK_TO_FORGET]
    slow = walked[stream.SLOW_TO_FORGET]

    quick_bars = stream.bars_until_the_sign_flips(quick)
    slow_bars = stream.bars_until_the_sign_flips(slow)
    assert quick_bars is not None, (
        "the quick-to-forget agent never adapted to the reversed relationship, "
        "so there is no rate to compare"
    )
    assert slow_bars is not None, (
        "the slow-to-forget agent never adapted within the pass, so the "
        "comparison would be between a rate and an absence of one"
    )
    assert quick_bars < slow_bars, (
        f"the agent whose forgetting gene is {QUICK} took {quick_bars} bars to "
        f"follow the break and the one at {SLOW} took {slow_bars}; a smaller "
        "forgetting factor discards older evidence faster, so it has to be the "
        "one that adapts sooner"
    )

    quick_settled = stream.bars_until_it_settles(quick)
    slow_settled = stream.bars_until_it_settles(slow)
    assert quick_settled is not None and slow_settled is not None
    assert quick_settled < slow_settled, (
        f"the quick agent reached its own new level after {quick_settled} bars "
        f"and the slow one after {slow_settled}; the gene has to change how "
        "long adapting takes, not only whether the sign eventually turns over"
    )


def test_the_forgetting_gene_is_what_makes_the_difference() -> None:
    """The comparison above is a comparison of the gene and of nothing else.

    Two agents with the *same* forgetting gene, walked the same way, have to
    produce the same trajectory. Without this, the check above would read the
    same if the difference came from the identities, the row order, or anything
    else that differs between two agents.
    """
    walked = stream.walk(_founded(quick=QUICK, slow=QUICK), stream.windows())
    assert (
        walked[stream.QUICK_TO_FORGET].learned == walked[stream.SLOW_TO_FORGET].learned
    )


def test_the_forgetting_gene_is_held_inside_its_declared_range() -> None:
    """A forgetting factor outside the declared range is refused, not clipped.

    The range is the configuration's, read by name. A population founded with a
    gene outside it would be an agent learning at a rate nobody declared, and
    silently pulling it to the edge would make the declaration decorative.
    """
    declared = _declared()
    minimum = float(declared[probe.FORGETTING_MINIMUM_KEY])  # ty: ignore[invalid-argument-type]
    maximum = float(declared[probe.FORGETTING_MAXIMUM_KEY])  # ty: ignore[invalid-argument-type]
    assert 0.0 < minimum <= maximum <= 1.0

    with pytest.raises(agent.ForgettingFactorOutsideItsRange):
        _founded(quick=minimum / 2.0)

    # Both edges, each probed one representable step outside it, and each
    # edge itself admitted: a refusal that checked only one side, or that
    # moved an edge inward, would otherwise read the same as this one.
    assert maximum < 1.0, (
        "the declared maximum is 1.0, so nothing representable above it is a "
        "forgetting factor and the upper edge cannot be probed"
    )
    for outside in (
        {"quick": math.nextafter(minimum, 0.0)},
        {"slow": math.nextafter(maximum, 1.0)},
        {"slow": (maximum + 1.0) / 2.0},
    ):
        with pytest.raises(agent.ForgettingFactorOutsideItsRange):
            _founded(**outside)
    at_the_edges = _founded(quick=minimum, slow=maximum)
    assert {
        at_the_edges.genome(agent_id).genes[agent.FORGETTING_FACTOR]
        for agent_id in at_the_edges.agent_ids()
    } == {minimum, maximum}


def _with_priors(
    priors: Mapping[str, float], *, run: config.EcologyConfig | None = None
) -> agent.PopulationState:
    """A population whose agents differ only in their ``weight.`` gene."""
    return agent.PopulationState.founded(
        tuple(
            stream.birth(agent_id, forgetting=SLOW, prior=prior)
            for agent_id, prior in priors.items()
        ),
        learning=(_default_run() if run is None else run).learning,
    )


def _below_the_founding_gain() -> config.EcologyConfig:
    """A run whose state bound is half the largest gain a generous pass reached.

    Derived, never written down: the generous pass is the declared stream under
    the declared bound, and for the selected family the largest gain on it is
    the one an agent is founded with. So half of it is a bound every newly
    founded agent starts above.
    """
    generous = stream.walk(_founded(), stream.windows())
    reached = max(generous[stream.QUICK_TO_FORGET].state_norms)
    return _run(state_bound=reached / 2.0)


def test_cold_start_weights_declared() -> None:
    """An agent born with nothing learned starts from its own genome's priors.

    Three things are asserted and all are needed. Each agent's learned weight
    equals the ``weight.`` gene its genome carries for that feature, read by
    name and compared bit for bit, over agents whose priors are distinct and
    non-zero — so an implementation that started everybody at one number,
    zero or the configuration's, is caught. Building the same population again
    gives the same state, so "declared" means determined rather than merely
    plausible. And the value is the genome's, not a number written here.
    """
    first = _with_priors(PRIORS)
    again = _with_priors(PRIORS)
    starts = set()
    for agent_id in first.agent_ids():
        declared = first.genome(agent_id).genes[WEIGHT_GENE]
        weights = first.learned_weights(agent_id)
        assert weights, "an agent that can learn nothing has no cold start to declare"
        assert dict(weights) == {stream.FEATURE: declared}, (
            f"{agent_id!r} starts at {dict(weights)} where its genome's "
            f"{WEIGHT_GENE!r} declares {declared}"
        )
        assert dict(again.learned_state(agent_id)) == dict(
            first.learned_state(agent_id)
        )
        starts.add(weights[stream.FEATURE])
    assert len(starts) == len(PRIORS), (
        f"agents carrying {len(PRIORS)} distinct priors started at {sorted(starts)}"
    )

    windows = stream.windows(count=1)
    assert first.step(windows[0], stream.INSTRUMENT) == again.step(
        windows[0], stream.INSTRUMENT
    )


def test_a_different_weight_gene_is_a_different_cold_start() -> None:
    """Two agents one ``weight.`` gene apart start apart, and go on to wish apart.

    The starting weights are compared in the feature the gene names. The wish is
    all-or-nothing, so its half is an exhibited pair at a stated decision point
    rather than a quantified claim: the first decision point at which the
    standardised reading is non-zero, which is where a prior can first reach a
    wish at all. The whole pass is counted too, so a cold start that the first
    update silently discarded — which would leave the two agents alike from
    the first outcome on — is caught rather than exhibited once and missed.
    """
    pair = _with_priors({"leans-long": LEANS, "leans-short": -LEANS})
    genes = {
        agent_id: dict(pair.genome(agent_id).genes) for agent_id in pair.agent_ids()
    }
    apart = {
        name
        for name in genes["leans-long"]
        if genes["leans-long"][name] != genes["leans-short"][name]
    }
    assert apart == {WEIGHT_GENE}
    assert (
        pair.learned_weights("leans-long")[stream.FEATURE]
        != pair.learned_weights("leans-short")[stream.FEATURE]
    )

    walked = stream.walk(pair, stream.windows())
    long, short = walked["leans-long"], walked["leans-short"]
    at = PRIOR_REACHES_A_WISH_AT
    assert long.standardised[:at] == short.standardised[:at] == (0.0,) * at
    assert long.standardised[at] != 0.0
    assert long.wanted[at] != short.wanted[at], (
        f"two agents whose priors are {LEANS} and {-LEANS} wished "
        f"{long.wanted[at]} and {short.wanted[at]} at decision point {at}, "
        f"on a standardised reading of {long.standardised[at]}"
    )
    differing = sum(
        mine != theirs for mine, theirs in zip(long.wanted, short.wanted, strict=True)
    )
    later = sum(
        mine != theirs
        for mine, theirs in zip(
            long.wanted[at + 1 :], short.wanted[at + 1 :], strict=True
        )
    )
    assert later > 0, (
        f"the pair wished differently at {differing} of {len(long.wanted)} decision "
        "points and at none after the first outcome could reach a wish, so the "
        "prior did not survive the update"
    )


def test_update_state_bounded_over_full_pass() -> None:
    """The estimator's state stays inside its declared bound for the whole pass.

    Twice over, on two streams, because the failure this guards is defined by
    the *absence* of information rather than by its content: a recursive
    estimator with exponential forgetting divides by a quantity that decays when
    nothing new arrives. The declared stream exercises the ordinary case; the
    constant-column mutation of it removes the excitation entirely, which is the
    case where wind-up happens if it is going to.

    Both bounds are read from the configuration by name, the pass is sampled at
    every decision point rather than at its end, and the report names the bar a
    bound would first be reached at.
    """
    declared = _declared()
    state_bound = float(declared[probe.STATE_BOUND_KEY])  # ty: ignore[invalid-argument-type]
    weight_bound = float(declared[probe.WEIGHT_NORM_BOUND_KEY])  # ty: ignore[invalid-argument-type]

    windows = stream.windows()
    unexciting = probe.constant_column(
        windows, feature=stream.FEATURE, instrument=stream.INSTRUMENT
    )
    _decision_zero_is_within_bounds(windows, state_bound, weight_bound)
    tight = _below_the_founding_gain()
    _decision_zero_is_within_bounds(
        windows, tight.learning.state_bound, weight_bound, run=tight
    )
    _a_prior_survives_a_gradient_free_first_outcome(windows, run=tight)
    _a_prior_beyond_the_bound_is_refused(weight_bound)

    for label, pass_over in (
        ("declared stream", windows),
        ("constant column", unexciting),
    ):
        walked = stream.walk(_founded(), pass_over)
        for agent_id, trajectory in walked.items():
            report = probe.judge(
                stamps=tuple(window.as_of for window in pass_over[1:]),
                state_norms=trajectory.state_norms,
                weight_norms=trajectory.weight_norms,
                state_bound=state_bound,
                weight_norm_bound=weight_bound,
            )
            assert report.bars == len(pass_over) - 1, (
                f"the {label} was sampled at {report.bars} of "
                f"{len(pass_over) - 1} decision points, so a bound could have "
                "been exceeded outside what was looked at"
            )
            assert report.within_bounds, (
                f"{agent_id!r} exceeded {report.first_breach_of} on the {label} "
                f"at bar {report.first_breach_index} "
                f"({report.first_breach_at.isoformat() if report.first_breach_at else '—'}); "
                f"largest state norm {report.largest_state_norm}, largest weight "
                f"norm {report.largest_weight_norm}"
            )


def _decision_zero_is_within_bounds(
    windows: tuple[agent.BarWindow, ...],
    state_bound: float,
    weight_bound: float,
    *,
    run: config.EcologyConfig | None = None,
) -> None:
    """Sample a population carrying non-zero priors before it has seen anything.

    The pass below is sampled after each outcome, so the state a population
    was founded with would be read by the first decision without ever being
    looked at. It is looked at here. The gain is recomputed from the stored
    accumulators, so a founding hold applied only to the reported gain would
    be seen. Under a bound below the founding gain, this is the sample that
    sees an agent founded outside it.
    """
    carrying = _with_priors(PRIORS, run=run)
    carrying.step(windows[0], stream.INSTRUMENT)
    sampled = {
        agent_id: probe.sampled(carrying, agent_id) for agent_id in carrying.agent_ids()
    }
    assert {weight_norm for _, weight_norm in sampled.values()} != {0.0}, (
        "every agent sampled at decision point 0 carries a zero weight, so the "
        "sample could not see a prior outside its bound"
    )
    for agent_id, (state_norm, weight_norm) in sampled.items():
        report = probe.judge(
            stamps=(windows[0].as_of,),
            state_norms=(state_norm,),
            weight_norms=(weight_norm,),
            state_bound=state_bound,
            weight_norm_bound=weight_bound,
        )
        assert report.within_bounds, (
            f"{agent_id!r} exceeded {report.first_breach_of} at decision point 0; "
            f"state norm {state_norm}, weight norm {weight_norm}"
        )


def _a_prior_survives_a_gradient_free_first_outcome(
    windows: tuple[agent.BarWindow, ...], *, run: config.EcologyConfig
) -> None:
    """At a first outcome that carries no gradient, a prior changes by forgetting alone.

    The standardised reading at decision point 0 is exactly zero, because there
    are no moments behind it. So the first outcome moves no weight through the
    gradient, and each learned weight has to be its prior times the agent's
    forgetting factor. A state bound that was met at founding and then held at
    the first outcome, by raising the evidence under a drift seeded against
    none, would shrink every prior here instead. That is a gene-declared value
    rescaled with nothing to say so.
    """
    carrying = _with_priors(PRIORS, run=run)
    carrying.step(windows[0], stream.INSTRUMENT)
    for agent_id in carrying.agent_ids():
        assert carrying.standardised(agent_id) == (0.0,)
    carrying.observe(
        windows[1].as_of,
        dict.fromkeys(carrying.agent_ids(), stream.outcome_at(0)),
    )
    for agent_id, prior in PRIORS.items():
        survived = carrying.learned_weights(agent_id)[stream.FEATURE]
        assert survived == pytest.approx(SLOW * prior, rel=1e-12), (
            f"{agent_id!r}'s prior {prior} is {survived} after a first outcome "
            f"that carried no gradient, where forgetting alone gives "
            f"{SLOW * prior}; under a state bound of {run.learning.state_bound}"
        )


def _a_prior_beyond_the_bound_is_refused(weight_bound: float) -> None:
    """A prior at twice the declared weight bound never reaches a decision."""
    beyond = 2.0 * weight_bound
    try:
        admitted = _with_priors({"beyond": beyond})
    except agent.StartingStateOutsideItsBound:
        return
    weight_norm = probe.sampled(admitted, "beyond")[1]
    pytest.fail(
        f"a prior of {beyond} was admitted and would be read at decision point 0 "
        f"with weight norm {weight_norm} against a declared bound of {weight_bound}"
    )


def test_the_bounded_state_check_can_see_a_bound_being_exceeded() -> None:
    """The check above is asserted over something that can fail.

    The same probe, the same stream and the same sampling, judged against a
    bound tight enough that the pass reaches it. Without this the check above
    would read exactly the same against a bound no estimator could exceed, which
    is how a guardrail becomes decoration.
    """
    windows = stream.windows()
    walked = stream.walk(_founded(), windows)
    trajectory = walked[stream.QUICK_TO_FORGET]
    report = probe.judge(
        stamps=tuple(window.as_of for window in windows[1:]),
        state_norms=trajectory.state_norms,
        weight_norms=trajectory.weight_norms,
        state_bound=min(trajectory.state_norms) / 2.0,
        weight_norm_bound=float("inf"),
    )
    assert not report.within_bounds
    assert report.first_breach_of == probe.STATE_BOUND_KEY
    assert report.first_breach_at is not None


def test_reaching_the_state_bound_is_recorded() -> None:
    """A run that reaches its state bound says so, and does not carry on unbounded.

    The bound is deliberately set tight enough that this pass reaches it. Two
    things then have to be true, and the second is the one that matters: the
    population records reaching it, naming the agent and the instant; and the
    state afterwards is held at the bound rather than continuing past it. A
    guardrail that recorded a breach and then let the state run would be a log
    line, not a bound.
    """
    tight = _below_the_founding_gain()

    state = _founded(run=tight)
    founding = state.bounds_reached()
    assert {record.agent_id for record in founding} == set(state.agent_ids()), (
        f"every agent was founded with a gain above the declared bound "
        f"{tight.learning.state_bound}, and the founding records name "
        f"{sorted(record.agent_id for record in founding)}"
    )
    assert all(
        record.at_founding
        and record.as_of is None
        and record.bound == probe.STATE_BOUND_KEY
        for record in founding
    ), f"a founding hold is not recorded as one: {founding}"
    walked = stream.walk(state, stream.windows())

    records = state.bounds_reached()
    assert records[: len(founding)] == founding
    held_later = records[len(founding) :]
    assert held_later, (
        f"the pass went on to reach a declared bound of "
        f"{tight.learning.state_bound} and nothing after founding was recorded"
    )
    assert {record.bound for record in records} == {probe.STATE_BOUND_KEY}
    assert {record.agent_id for record in records} <= set(state.agent_ids())
    assert all(
        record.as_of is not None and not record.at_founding for record in held_later
    )

    for trajectory in walked.values():
        assert max(trajectory.state_norms) <= tight.learning.state_bound * (
            1.0 + 1e-9
        ), (
            "the state carried on past the bound after it was recorded, so the "
            "record describes a run that continued unbounded"
        )


#: How many outcomes the cold-start transient is taken to last: two full periods
#: of the declared feature. Until the feature has cycled through its values the
#: standardiser's spread, and so the gradient's size, is still settling. It is a
#: property of the stream, not a number fitted to where any bound is reached.
COLD_START_TRANSIENT = 2 * stream.CYCLE

#: The forgetting factor of the one agent the lost-excitation reach is shown on.
LOST_EXCITATION_FORGETTING = 0.99


def _held_from_the_break(
    windows: tuple[agent.BarWindow, ...],
) -> tuple[agent.BarWindow, ...]:
    """The declared stream with its feature held from the break onward.

    INVALID MUTATION of the declared stream. From the break on, the feature
    keeps the value it takes at the break and stops varying. Every instant,
    bar and outcome is the stream's own. It exists so the loss of excitation
    after a stretch of learning can be shown, and it is representative of
    nothing.
    """
    tail = probe.constant_column(
        windows[stream.BREAK_AT :], feature=stream.FEATURE, instrument=stream.INSTRUMENT
    )
    return (*windows[: stream.BREAK_AT], *tail)


def test_the_state_bound_is_reached_again_when_excitation_is_lost() -> None:
    """A gain climbing back as information vanishes is held and recorded, after the start.

    Without this the only reach EV-E70 shows is the cold-start one: the founded
    gain is the family's ceiling, so any bound below it is met at founding.
    Here the feature stops varying at the break. The standardised reading then
    decays towards zero, the evidence decays with it, and the gain climbs back
    towards its founding value. That is the bounded form of the wind-up the
    bound exists for.

    The bound is derived from a generous pass over the same mutated stream. It
    is the midpoint between the largest gain after the cold-start transient and
    before the break, and the largest gain after the break. So it sits above
    everything the exciting stretch reached, below what the lost excitation
    reaches, and below the founding gain. Records must then resume after the
    break, and none may fire between the end of the transient and the break.
    """
    windows = _held_from_the_break(stream.windows())

    def founded(run: config.EcologyConfig) -> agent.PopulationState:
        return agent.PopulationState.founded(
            (stream.birth("loses-excitation", forgetting=LOST_EXCITATION_FORGETTING),),
            learning=run.learning,
        )

    generous = stream.walk(founded(_default_run()), windows)["loses-excitation"]
    exciting = max(generous.state_norms[COLD_START_TRANSIENT : stream.BREAK_AT])
    unexcited = max(generous.state_norms[stream.BREAK_AT :])
    founding_gain = probe.sampled(founded(_default_run()), "loses-excitation")[0]
    assert exciting < unexcited < founding_gain, (
        f"the gain after the transient peaked at {exciting} while the feature "
        f"varied and at {unexcited} after it stopped, against a founding gain "
        f"of {founding_gain}; there is no bound that only lost excitation reaches"
    )
    tight = _run(state_bound=(exciting + unexcited) / 2.0)

    state = founded(tight)
    stream.walk(state, windows)
    outcome_at = {window.as_of: index - 1 for index, window in enumerate(windows)}
    held_at = sorted(
        outcome_at[record.as_of]
        for record in state.bounds_reached()
        if not record.at_founding
    )
    between = [
        index for index in held_at if COLD_START_TRANSIENT <= index < stream.BREAK_AT
    ]
    assert not between, (
        f"the bound {tight.learning.state_bound} was held at outcomes {between}, "
        "between the cold-start transient and the break, where the feature varied"
    )
    resumed = [index for index in held_at if index >= stream.BREAK_AT]
    assert resumed, (
        f"the feature stopped varying at outcome {stream.BREAK_AT} and the gain "
        f"never reached the bound {tight.learning.state_bound} after it, where "
        f"the generous pass climbed to {unexcited}"
    )


def test_reaching_the_weight_norm_bound_is_recorded_and_held() -> None:
    """The weight half of the bound, reached, recorded and held at Tier 1.

    Derived much as the state bound above is: half the smaller of the two
    agents' largest weight norms over a generous-bound pass, so this file
    chooses no number the stream then happens to cross, and both agents cross
    it. The records have to name the weight bound by
    its dotted key and nothing else — the state bound is left at its declared
    value, which this pass does not reach — and from each agent's first record
    on, the weights have to sit inside the bound twice over: the weight the
    rule reads, and the weight the stored accumulators imply. The second is the
    one a trimmed-but-not-held weight fails, because the next outcome
    recomputes the weight from the accumulator.
    """
    generous = stream.walk(_founded(), stream.windows())
    reached = min(max(trajectory.weight_norms) for trajectory in generous.values())
    tight = _run(weight_norm_bound=reached / 2.0)
    bound = tight.learning.weight_norm_bound

    state = _founded(run=tight)
    windows = stream.windows()
    records: list[agent.BoundReached] = []
    held_from: dict[str, int] = {}
    for index, window in enumerate(windows[:-1]):
        state.step(window, stream.INSTRUMENT)
        records.extend(
            state.observe(
                windows[index + 1].as_of,
                dict.fromkeys(state.agent_ids(), stream.outcome_at(index)),
            )
        )
        for record in records:
            held_from.setdefault(record.agent_id, index)
        for agent_id, since in held_from.items():
            reads = probe.sampled(state, agent_id)[1]
            implied = probe.weights_from_accumulators(state, agent_id)
            assert reads <= bound * (1.0 + 1e-9), (
                f"{agent_id!r}'s weight norm is {reads} at outcome {index}, "
                f"after the bound {bound} was recorded at outcome {since}"
            )
            assert sum(value**2 for value in implied) ** 0.5 <= bound * (1.0 + 1e-9), (
                f"{agent_id!r}'s accumulators imply weights {implied} at outcome "
                f"{index}, outside the bound {bound} the record says was held; "
                "the next outcome recomputes the weight from them"
            )
            assert implied == pytest.approx(
                tuple(state.learned_weights(agent_id).values()), rel=1e-12
            ), "the weight the rule reads is not the weight the accumulators hold"

    assert records, (
        f"every agent of the generous pass reached a weight norm of at least "
        f"{reached}, and a bound of {bound} was never recorded"
    )
    assert {record.bound for record in records} == {probe.WEIGHT_NORM_BOUND_KEY}
    assert set(held_from) == set(state.agent_ids()), (
        f"only {sorted(held_from)} reached a bound set at half the smaller peak "
        "norm, so the other agent's hold was never looked at"
    )
    assert state.bounds_reached() == tuple(records)


def test_the_bounded_state_check_reads_a_non_finite_norm_as_a_breach() -> None:
    """A norm that is not a number is outside its bound, not inside it.

    Every comparison with a not-a-number is false, so a judge that asked
    "is it above the bound?" would report an agent whose state the arithmetic
    had already lost as one the bound was holding. Asked of each bound in turn,
    at a stated bar, with finite norms either side.
    """
    stamps = tuple(window.as_of for window in stream.windows(count=3))
    finite = (0.1, 0.1, 0.1)
    for state_norms, weight_norms, breached in (
        ((0.1, math.nan, 0.1), finite, probe.STATE_BOUND_KEY),
        (finite, (0.1, math.nan, 0.1), probe.WEIGHT_NORM_BOUND_KEY),
        ((0.1, math.inf, 0.1), finite, probe.STATE_BOUND_KEY),
    ):
        report = probe.judge(
            stamps=stamps,
            state_norms=state_norms,
            weight_norms=weight_norms,
            state_bound=1.0e3,
            weight_norm_bound=1.0e2,
        )
        assert not report.within_bounds, (
            f"norms {state_norms} and {weight_norms} were judged inside bounds "
            "of 1000 and 100"
        )
        assert (report.first_breach_index, report.first_breach_of) == (1, breached)
        assert report.first_breach_at == stamps[1]
    nan_report = probe.judge(
        stamps=stamps,
        state_norms=(0.1, math.nan, 0.1),
        weight_norms=finite,
        state_bound=1.0e3,
        weight_norm_bound=1.0e2,
    )
    assert math.isnan(nan_report.largest_state_norm), (
        f"the largest state norm over a pass holding a not-a-number was "
        f"reported as {nan_report.largest_state_norm}"
    )


def test_a_run_inside_its_bound_records_nothing() -> None:
    """Reaching the bound is recorded because it happened, not on every run.

    Without this the check above would pass against an implementation that
    recorded a breach unconditionally, which would make the record say nothing.
    """
    state = _founded()
    stream.walk(state, stream.windows())
    assert state.bounds_reached() == ()
