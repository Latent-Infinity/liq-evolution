"""Evidence about the online update: the gene that drives it, and its limits.

Four claims, and they divide in two. The first pair is about what the update
*is*: the rate at which an agent forgets is inherited, so two agents that differ
in nothing else adapt at measurably different speeds; and an agent that has been
shown no outcome carries the weights its configuration declares, not whatever
the arithmetic happened to start at. The second pair is about what the update
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

#: A cold start that is not zero, so an agent carrying the declared value can be
#: told from one carrying whatever an uninitialised array holds.
DECLARED_COLD_START = 0.25

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


def test_cold_start_weights_declared() -> None:
    """An agent shown no outcome carries the weights its configuration declares.

    Two things are asserted and both are needed. The value is the configured
    one, read by name rather than written here, so an implementation that
    started everybody at zero would be caught even though zero is a perfectly
    reasonable number. And building the same population again gives the same
    thing, so "declared" means determined rather than merely plausible.
    """
    run = _run(cold_start_weight=DECLARED_COLD_START)
    declared = _declared(run)
    cold_start = float(declared[probe.COLD_START_WEIGHT_KEY])  # ty: ignore[invalid-argument-type]

    first = _founded(run=run)
    again = _founded(run=run)
    for agent_id in first.agent_ids():
        weights = first.learned_weights(agent_id)
        assert weights, "an agent that can learn nothing has no cold start to declare"
        assert set(weights.values()) == {cold_start}, (
            f"{agent_id!r} starts at {sorted(set(weights.values()))} where its "
            f"configuration declares {cold_start}"
        )
        assert dict(again.learned_weights(agent_id)) == dict(weights)

    windows = stream.windows(count=1)
    assert first.step(windows[0], stream.INSTRUMENT) == again.step(
        windows[0], stream.INSTRUMENT
    )


def test_a_different_declared_cold_start_is_a_different_cold_start() -> None:
    """The check above reads the configuration rather than a coincidence.

    A population built under a different declared value has to start somewhere
    different. Without this the check above would pass against an
    implementation that ignored the key and happened to agree with it.
    """
    elsewhere = agent.PopulationState.founded(
        stream.two_agents(quick=QUICK, slow=SLOW),
        learning=config.LearningConfig(cold_start_weight=DECLARED_COLD_START * 3.0),
    )
    started = _founded()
    for agent_id in started.agent_ids():
        assert dict(elsewhere.learned_weights(agent_id)) != dict(
            started.learned_weights(agent_id)
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
    generous = stream.walk(_founded(), stream.windows())
    reached = max(generous[stream.QUICK_TO_FORGET].state_norms)
    tight = _run(state_bound=reached / 2.0)

    state = _founded(run=tight)
    walked = stream.walk(state, stream.windows())

    records = state.bounds_reached()
    assert records, (
        f"the pass reached a state norm of {reached} against a declared bound "
        f"of {tight.learning.state_bound} and nothing was recorded"
    )
    assert {record.bound for record in records} == {probe.STATE_BOUND_KEY}
    assert {record.agent_id for record in records} <= set(state.agent_ids())
    assert all(record.as_of is not None for record in records)

    for trajectory in walked.values():
        assert max(trajectory.state_norms) <= tight.learning.state_bound * (
            1.0 + 1e-9
        ), (
            "the state carried on past the bound after it was recorded, so the "
            "record describes a run that continued unbounded"
        )


def test_a_run_inside_its_bound_records_nothing() -> None:
    """Reaching the bound is recorded because it happened, not on every run.

    Without this the check above would pass against an implementation that
    recorded a breach unconditionally, which would make the record say nothing.
    """
    state = _founded()
    stream.walk(state, stream.windows())
    assert state.bounds_reached() == ()
