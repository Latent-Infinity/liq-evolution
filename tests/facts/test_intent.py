"""Evidence that what an agent has learned reaches the wish it forms.

One claim, and it is the half of the intent fact that the genome and the view
cannot carry. An agent's wish is a function of three things — the genome it
inherited, the state it has learned and the view it was broadcast — and the
first two are asserted in different repositories for the reason each needs: the
genome clause and the reproducibility clause are about a rule applied to a tape
and live beside the tape, in the consumer; this one is about whether the learned
state is *consulted at all*, which is a property of the representation and needs
no market.

**Why this check exists as its own module.** A system can learn perfectly,
converge at gene-controlled rates, bound its state, standardise causally and
report all of it — and still never act on any of it. That system passes every
other learning claim in this repository and the first two clauses of the intent
claim unchanged. It was the shipped system: before this check, replacing an
agent's entire learned state with arbitrary values before every one of the
declared stream's decision points left the wish sequence bit-identical. The
clause below is the falsifier for exactly that, and it is quantified two ways
because the two catch different regressions — an exhibited pair at one stated
decision point, which is what an all-or-nothing wish can actually establish, and
a count over a whole pass, which is what stops a rule that consults the learned
state at one bar and ignores it at the rest.

**What the exhibited pair can and cannot establish.** The wish is all-or-nothing,
so "a differing learned weight produces a differing intent" is a claim about a
pair that is exhibited, not a claim quantified over every perturbation: a change
too small to carry the reading across the entry gene changes nothing and is not
a counterexample. The pair below is chosen so the reading crosses the gene, and
the crossing is asserted rather than assumed.

**Nothing here is market data.** The decision points are the declared arithmetic
ramp; `tests/support/level_shift_stream.py` records what that costs and what the
approved tape would have added. Every number here is a number about that stream.
"""

from __future__ import annotations

import numpy as np

from liq.evolution.ecology import agent, config, learning
from tests.support import level_shift_stream as stream

#: How many decision points are walked before the pair is exhibited. Enough
#: that the online standardiser has prior-bar statistics to scale by — at the
#: first decision point there are none and the standardised reading is zero, so
#: no learned weight could move a wish there and a pair exhibited at it would
#: prove nothing.
WARMED_UP_AFTER = 40

#: The two learned weights the exhibited pair differs in. Equal and opposite, so
#: whichever sign the standardised reading carries at the stated decision point,
#: one of the two clears the entry gene and the other does not.
LEANS_LONG = 1.0
LEANS_SHORT = -1.0

#: The spread of the arbitrary values the pass below is perturbed with. Large
#: relative to anything the update itself produces, so a wish that does not move
#: has not merely failed to cross the gene.
ARBITRARY_SPREAD = 1.0e3

#: Seed the arbitrary values are drawn under, so the count this reports is the
#: same count on every machine.
SEED = 7

#: The forgetting factor both agents here carry. One value, because nothing
#: below is a claim about the gene.
FORGETTING = 0.99

#: The learned-state name the rule consults for the stream's one feature.
LEARNED_WEIGHT = f"{learning.WEIGHT_PREFIX}{stream.FEATURE}"


def _founded() -> agent.PopulationState:
    """One agent over the declared stream, at its configured cold start."""
    return agent.PopulationState.founded(
        (stream.birth(stream.QUICK_TO_FORGET, forgetting=FORGETTING),),
        learning=config.LearningConfig(),
    )


def _shown(
    state: agent.PopulationState, windows: tuple[object, ...], *, until: int
) -> None:
    """Step ``state`` over the first ``until`` of ``windows``, showing each outcome.

    The pairing is the one the stream declares: a decision point is answered
    from the view at that instant, and what the reading earned is shown at the
    instant after it — which is why the walk stops one short of the decision
    point the pair is exhibited at rather than at it.
    """
    for index in range(until):
        state.step(windows[index], stream.INSTRUMENT)  # ty: ignore[invalid-argument-type]
        state.observe(
            windows[index + 1].as_of,  # ty: ignore[unresolved-attribute]
            {stream.QUICK_TO_FORGET: stream.outcome_at(index)},
        )


def test_a_learned_weight_the_rule_consults_moves_the_wish() -> None:
    """Two agents alike in genome and history, apart in one learned weight, differ.

    The pair is exhibited at one stated decision point and every other input is
    held identical: the same genome, the same stream, the same decision point,
    the same prior-bar statistics. What differs is one learned weight, and the
    wish has to differ with it — otherwise the agent learns and cannot act.

    Three things make the pair mean something and each is asserted rather than
    assumed: the genomes are equal, so nothing heritable is doing the work; the
    standardised reading at the stated point is not zero, so the weight has
    something to multiply; and the two weights sit either side of nothing, so
    the reading crosses the entry gene rather than merely changing size.
    """
    windows = stream.windows(count=WARMED_UP_AFTER + 2)
    exhibited_at = windows[WARMED_UP_AFTER]

    leans_long = _founded()
    leans_short = _founded()
    for state in (leans_long, leans_short):
        _shown(state, windows, until=WARMED_UP_AFTER)

    assert leans_long.genome(stream.QUICK_TO_FORGET) == leans_short.genome(
        stream.QUICK_TO_FORGET
    ), (
        "the pair differs in a gene, so a differing wish would say nothing about learning"
    )

    leans_long.learn(stream.QUICK_TO_FORGET, {LEARNED_WEIGHT: LEANS_LONG})
    leans_short.learn(stream.QUICK_TO_FORGET, {LEARNED_WEIGHT: LEANS_SHORT})

    long_step = leans_long.step(exhibited_at, stream.INSTRUMENT)
    short_step = leans_short.step(exhibited_at, stream.INSTRUMENT)

    reading = leans_long.standardised(stream.QUICK_TO_FORGET)[0]
    assert reading != 0.0, (
        f"the standardised reading at {exhibited_at.as_of.isoformat()} is zero, "
        "so no learned weight could move the wish there and the pair below "
        "would pass against a rule that consults nothing"
    )
    assert long_step.wanted != short_step.wanted, (
        f"two agents carrying the same genome and the same history, differing "
        f"only in the learned weight for {stream.FEATURE!r} "
        f"({LEANS_LONG} against {LEANS_SHORT}) on a standardised reading of "
        f"{reading}, formed the same wish {long_step.wanted} at "
        f"{exhibited_at.as_of.isoformat()}; the agent learns and cannot act on it"
    )


def test_an_arbitrary_learned_state_moves_the_wish_sequence() -> None:
    """Over a whole pass, replacing what was learned changes what is wanted.

    The companion to the pair above, and it catches what a pair cannot: a rule
    that consults the learned state at one decision point and ignores it at the
    rest. Two passes over the same stream, identical but for the learned weight
    being overwritten with an arbitrary value before every decision point, and
    the wish sequences have to part.

    The count is reported rather than merely asserted non-zero, because the
    number is the evidence: this check was written against an implementation
    whose wish sequence was bit-identical under the same perturbation at every
    one of the stream's decision points, and a regression to that implementation
    is a count of zero rather than an error.
    """
    windows = stream.windows()
    untouched = _wishes(windows, perturbed=False)
    arbitrary = _wishes(windows, perturbed=True)

    differing = sum(
        1 for left, right in zip(untouched, arbitrary, strict=True) if left != right
    )
    assert differing > 0, (
        f"the learned weight was replaced with an arbitrary value before each "
        f"of {len(windows)} decision points and 0 wishes moved; the wish is "
        "independent of everything the agent has learned"
    )


def _wishes(windows: tuple[object, ...], *, perturbed: bool) -> tuple[float, ...]:
    """Every wish one agent forms over ``windows``, optionally with learning erased.

    Where ``perturbed``, the learned weight the rule consults is overwritten
    with an arbitrary value immediately before each decision point is answered.
    The update still runs and still writes its own weight afterwards, so the
    perturbation is of what the decision reads and not of what the estimator
    does.
    """
    draws = np.random.default_rng(SEED)
    state = _founded()
    formed: list[float] = []
    for index, window in enumerate(windows):
        if perturbed:
            state.learn(
                stream.QUICK_TO_FORGET,
                {LEARNED_WEIGHT: float(draws.normal(0.0, ARBITRARY_SPREAD))},
            )
        step = state.step(window, stream.INSTRUMENT)  # ty: ignore[invalid-argument-type]
        formed.append(step.wanted[0])
        if index + 1 < len(windows):
            state.observe(
                windows[index + 1].as_of,  # ty: ignore[unresolved-attribute]
                {stream.QUICK_TO_FORGET: stream.outcome_at(index)},
            )
    return tuple(formed)
