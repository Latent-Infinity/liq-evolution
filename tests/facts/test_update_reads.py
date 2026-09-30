"""Evidence that the update learns from exactly the reading the decision was formed from.

A decision reads a feature only where the genome switches it on and the view
offers it. The update that learns from the decision's outcome has to read the
same thing. If it learned from every feature instead, the agent would act on
weights fitted to a prediction it never makes: a switched-off feature's prior
and values would steer, through the shared prediction error, the weights of the
features the agent does read. A withheld feature would also reach the update as
a standardised raw zero, a value nobody broadcast. Three checks, one for each
way in:

- a switched-off feature's ``weight.`` gene, varied, moves no wish;
- a switched-off feature's values, replaced, move no switched-on learned weight
  by so much as a bit. The check is on the weights and not only on the wishes,
  because at an entry gene of zero a wish is a sign test and a moved weight can
  hide inside it;
- at a decision point where the view withholds a feature, that feature's
  accumulators move by forgetting alone, and every other feature's update is
  the one an independent reimplementation computes with the withheld
  contribution set to zero.

Each check has a companion that turns the feature on, and there the same
difference has to show. Without it, a check that could see nothing would read
the same.

**Nothing here is market data.** The decision points are the declared
arithmetic stream in `tests/support/level_shift_stream.py`, with a second
feature added by declared arithmetic: an INVALID MUTATION of that stream,
labelled as one, and evidence of nothing about any market.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

import pytest

from liq.evolution.ecology import Genome, agent, config, learning
from tests.support import level_shift_stream as stream
from tests.support import state_norm_probe as probe

#: The second feature, added to the declared stream by arithmetic.
SLOW = "slow"

#: The two series the second feature can carry. They are distinct and not
#: proportional, so replacing one with the other changes the standardised
#: reading and does not merely rescale it.
SLOW_PERIOD = 11
OTHER_PERIOD = 13

#: How much of the second feature the outcome carries. The outcome is always
#: computed from the *first* series. Swapping the series shown is then a
#: history that differs only in the values of that feature, which is the
#: fact's antecedent.
SLOW_SENSITIVITY = 0.3

#: The level the withheld feature sits at, and the decision points it is
#: withheld at. The level keeps a raw zero far from where the feature lives,
#: so a standardised zero-that-nobody-broadcast is large and easy to see. The
#: points are well past warm-up and before the stream's break.
SLOW_LEVEL = 100.0
WITHHELD = range(300, 311)

#: One forgetting factor for every agent here, because nothing below is a
#: claim about that gene.
FORGETTING = 0.99

#: The prior a comparison varies a ``weight.`` gene to, inside the declared
#: weight bound, away from the 0.0 every other genome here carries.
VARIED_PRIOR = 1.0


def _slow_at(index: int) -> float:
    """The second feature's first series."""
    return float(index % SLOW_PERIOD) - float(SLOW_PERIOD // 2)


def _other_at(index: int) -> float:
    """The second feature's replacement series."""
    return float((3 * index) % OTHER_PERIOD) - float(OTHER_PERIOD // 2)


def _outcome_at(index: int) -> float:
    """What holding the reading at ``index`` earned, from the first series only."""
    return stream.outcome_at(index) + SLOW_SENSITIVITY * _slow_at(index)


def _outcomes() -> tuple[float, ...]:
    return tuple(_outcome_at(index) for index in range(stream.DECISION_POINTS))


def _windows(
    slow: Callable[[int], float] = _slow_at,
    *,
    withheld: Sequence[int] = (),
) -> tuple[agent.BarWindow, ...]:
    """The declared stream with the second feature added, and withheld where named.

    INVALID MUTATION of the declared stream: one feature is added, and at the
    named decision points it is left out of the view. Nothing else changes.
    """
    return tuple(
        replace(
            window,
            features=MappingProxyType(
                {
                    stream.INSTRUMENT: MappingProxyType(
                        {stream.FEATURE: stream.feature_at(index)}
                        if index in withheld
                        else {
                            stream.FEATURE: stream.feature_at(index),
                            SLOW: slow(index),
                        }
                    )
                }
            ),
        )
        for index, window in enumerate(stream.windows())
    )


def _birth(
    agent_id: str, *, slow_on: bool, slow_prior: float = 0.0
) -> agent.AgentBirth:
    """An agent reading the declared feature, with the second one on or off."""
    genes = {
        f"{agent.MASK_PREFIX}{stream.FEATURE}": 1.0,
        f"{agent.WEIGHT_PREFIX}{stream.FEATURE}": stream.PRIOR_AT,
        f"{agent.MASK_PREFIX}{SLOW}": 1.0 if slow_on else 0.0,
        f"{agent.WEIGHT_PREFIX}{SLOW}": slow_prior,
        agent.ENTRY_THRESHOLD: stream.ENTRY_AT,
        agent.FORGETTING_FACTOR: FORGETTING,
    }
    return agent.AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=MappingProxyType(genes), schema_version=stream.GENE_SCHEMA_VERSION
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=stream.FEATURE_SCHEMA_VERSION,
        model_version=stream.MODEL_VERSION,
    )


def _founded(*births: agent.AgentBirth) -> agent.PopulationState:
    state = agent.PopulationState.founded(births, learning=config.LearningConfig())
    assert state.feature_names == (stream.FEATURE, SLOW)
    return state


def _count_apart(left: Sequence[float], right: Sequence[float]) -> int:
    return sum(mine != theirs for mine, theirs in zip(left, right, strict=True))


def _walk_prior_pair(*, slow_on: bool) -> Mapping[str, stream.Trajectory]:
    """Two agents a switched-off (or on) feature's ``weight.`` gene apart."""
    return stream.walk(
        _founded(
            _birth("prior-zero", slow_on=slow_on),
            _birth("prior-varied", slow_on=slow_on, slow_prior=VARIED_PRIOR),
        ),
        _windows(),
        outcomes=_outcomes(),
    )


def test_a_switched_off_feature_s_prior_moves_no_wish() -> None:
    """(a) Varying a switched-off feature's ``weight.`` gene moves no wish anywhere.

    The companion switches the same feature on. There the prior is where its
    learned weight starts, so the wishes part by design, which shows that this
    walk can see a prior at all.
    """
    walked = _walk_prior_pair(slow_on=False)
    reference = walked["prior-zero"].wanted
    assert len(set(reference)) > 1, "the reference never changed its wish"
    moved = _count_apart(walked["prior-varied"].wanted, reference)
    assert moved == 0, (
        f"a switched-off feature's prior of {VARIED_PRIOR} against 0.0 moved "
        f"{moved} of {len(reference)} wishes"
    )

    switched_on = _walk_prior_pair(slow_on=True)
    assert (
        _count_apart(
            switched_on["prior-varied"].wanted, switched_on["prior-zero"].wanted
        )
        > 0
    ), "switched on, the same prior moved no wish, so this walk cannot see one"


def _walk_series_pair(*, slow_on: bool) -> tuple[stream.Trajectory, stream.Trajectory]:
    """One agent shown each series of the second feature, with the same outcomes."""
    return tuple(  # ty: ignore[invalid-return-type]
        stream.walk(
            _founded(_birth("shown", slow_on=slow_on)),
            _windows(series),
            outcomes=_outcomes(),
        )["shown"]
        for series in (_slow_at, _other_at)
    )


def test_a_switched_off_feature_s_values_move_no_switched_on_weight() -> None:
    """(b) Replacing a switched-off feature's series leaves every switched-on weight exact.

    The weight the rule read at each decision point and the weight after each
    outcome are both compared bit for bit, not only the wish. The companion
    switches the feature on, and there the same replacement has to move the
    declared feature's weight.
    """
    first, second = _walk_series_pair(slow_on=False)
    apart = max(
        abs(mine - theirs)
        for mine, theirs in zip(
            first.in_force + first.learned,
            second.in_force + second.learned,
            strict=True,
        )
    )
    assert first.in_force == second.in_force and first.learned == second.learned, (
        f"replacing a switched-off feature's values moved the switched-on "
        f"{stream.FEATURE!r} weight by up to {apart}"
    )
    assert first.wanted == second.wanted

    on_first, on_second = _walk_series_pair(slow_on=True)
    assert on_first.learned != on_second.learned, (
        "switched on, replacing the feature's values moved no weight, so this "
        "comparison cannot see one"
    )


def _reference_update(
    before: Mapping[str, float],
    readings: Mapping[str, float],
    outcome: float,
    configured: config.LearningConfig,
) -> dict[str, float]:
    """One FTRL-Proximal step, restated here, from the reading the decision used.

    Written out rather than imported, so the oracle does not agree with the
    implementation whatever either does. ``readings`` holds the value each
    feature contributed to the decision, with a withheld feature at zero.
    """
    surprise = (
        sum(
            readings[name] * before[f"{learning.WEIGHT_PREFIX}{name}"]
            for name in readings
        )
        - outcome
    )
    after: dict[str, float] = {}
    for name, value in readings.items():
        weight = before[f"{learning.WEIGHT_PREFIX}{name}"]
        drift = before[f"{learning.DRIFT_PREFIX}{name}"]
        energy = before[f"{learning.ENERGY_PREFIX}{name}"]
        gradient = surprise * value
        curvature = (
            math.sqrt(energy + gradient * gradient) - math.sqrt(energy)
        ) / configured.step_scale
        drift = FORGETTING * drift + gradient - curvature * weight
        energy = FORGETTING * energy + gradient * gradient
        step = (
            configured.step_offset + math.sqrt(energy)
        ) / configured.step_scale + configured.squared_penalty
        after[f"{learning.DRIFT_PREFIX}{name}"] = drift
        after[f"{learning.ENERGY_PREFIX}{name}"] = energy
        after[f"{learning.WEIGHT_PREFIX}{name}"] = -drift / step
    return after


def test_a_withheld_feature_moves_by_forgetting_alone_and_steers_nothing() -> None:
    """(c) Across a withheld decision point, the update consumes nothing of that feature.

    The withheld feature's drift and energy change by the forgetting factor
    alone. The declared feature's drift, energy and weight equal an
    independent FTRL step computed from the reading the decision was formed
    from, with the withheld feature's contribution set to zero. That reading
    is recomputed by the probe's own causal standardisation, not read back
    from the population. The companion runs the same series with nothing
    withheld, and there the second feature's accumulators move by more than
    forgetting.
    """
    configured = config.LearningConfig()
    assert configured.absolute_penalty == 0.0

    def levelled(index: int) -> float:
        return SLOW_LEVEL + _slow_at(index)

    windows = _windows(levelled, withheld=WITHHELD)
    cycle = probe.reference_standardised(
        tuple(stream.feature_at(index) for index in range(len(windows))),
        forgetting=FORGETTING,
        peek=False,
    )
    state = _founded(_birth("reads-both", slow_on=True))
    outcomes = _outcomes()
    consumed_while_withheld: dict[int, float] = {}
    drift_of_slow = f"{learning.DRIFT_PREFIX}{SLOW}"
    energy_of_slow = f"{learning.ENERGY_PREFIX}{SLOW}"
    for index, window in enumerate(windows[: WITHHELD[-1] + 1]):
        state.step(window, stream.INSTRUMENT)
        before = dict(state.learned_state("reads-both"))
        if index in WITHHELD:
            consumed_while_withheld[index] = state.standardised("reads-both")[1]
        state.observe(windows[index + 1].as_of, {"reads-both": outcomes[index]})
        if index not in WITHHELD:
            continue
        after = state.learned_state("reads-both")
        assert (after[drift_of_slow], after[energy_of_slow]) == (
            FORGETTING * before[drift_of_slow],
            FORGETTING * before[energy_of_slow],
        ), (
            f"at withheld decision point {index} the withheld feature's "
            f"accumulators moved by more than forgetting; the update consumed "
            f"{consumed_while_withheld[index]} for a value nobody broadcast"
        )
        expected = _reference_update(
            before, {stream.FEATURE: cycle[index]}, outcomes[index], configured
        )
        for name, value in expected.items():
            assert after[name] == pytest.approx(value, rel=1e-9, abs=1e-15), (
                f"at withheld decision point {index}, {name} is {after[name]} "
                f"where the update computed without the withheld feature gives "
                f"{value}; the update consumed {consumed_while_withheld[index]} "
                "for it"
            )

    offered = _founded(_birth("reads-both", slow_on=True))
    nothing_withheld = _windows(levelled)
    for index, window in enumerate(nothing_withheld[: WITHHELD[0] + 1]):
        offered.step(window, stream.INSTRUMENT)
        before = dict(offered.learned_state("reads-both"))
        offered.observe(
            nothing_withheld[index + 1].as_of, {"reads-both": outcomes[index]}
        )
    after = offered.learned_state("reads-both")
    assert after[drift_of_slow] != FORGETTING * before[drift_of_slow], (
        "offered, the second feature moved by forgetting alone, so this check "
        "cannot tell a feature the update consumed from one it did not"
    )
