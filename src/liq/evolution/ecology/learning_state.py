"""Names and initial values for the state an agent learns from outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from liq.evolution.ecology.config import LearningConfig

#: The family the V2.6.2 ablation selected, named so a run's record says which
#: one produced it rather than leaving a reader to infer it from the code.
ESTIMATOR_FAMILY = "ftrl_proximal"

#: What one agent's update cost per bar when it was measured, in seconds. See
#: the module docstring; recorded rather than asserted, because a timing
#: assertion on a shared machine is a flaky test rather than evidence.
MEASURED_SECONDS_PER_AGENT_PER_BAR = 8.6e-07

#: Learned-state name: the weight the update has arrived at for one feature.
#: Distinct from the genome's ``weight.`` genes, and deliberately so — the
#: heritable weight and the learned one are different quantities with different
#: lifetimes, and the inheritance arm turns on being able to carry one without
#: the other.
WEIGHT_PREFIX = "learned.weight."

#: Learned-state name: the forgetting-weighted accumulated adjusted gradient.
DRIFT_PREFIX = "update.drift."

#: Learned-state name: the forgetting-weighted accumulated squared gradient.
#: The gain is a falling function of this, which is why holding the gain under
#: a bound is done by raising this rather than by editing the gain.
ENERGY_PREFIX = "update.energy."

#: Learned-state names: the online standardiser's forgetting-weighted moments,
#: and the mass they are divided by. Separately addressable, because a
#: checkpoint that restored the weights and not these would resume an agent
#: whose inputs are scaled differently from the ones it learned on.
SCALE_TOTAL_PREFIX = "scale.total."
SCALE_SQUARES_PREFIX = "scale.squares."
SCALE_MASS = "scale.mass"

#: Read-only name: the multiplier the estimator would apply to the next
#: surprise. Derived from the energy rather than stored, so there is one place
#: the gain is defined and no second copy to fall out of step.
GAIN_PREFIX = "gain."


def weight_names(features: Sequence[str]) -> tuple[str, ...]:
    """The learned-state names holding the weights, in feature order."""
    return tuple(f"{WEIGHT_PREFIX}{name}" for name in features)


def gain_names(features: Sequence[str]) -> tuple[str, ...]:
    """The names the per-feature gains are reported under, in feature order."""
    return tuple(f"{GAIN_PREFIX}{name}" for name in features)


def declared_columns(features: Sequence[str]) -> tuple[str, ...]:
    """Every learned-state name the update owns, for ``features``.

    A population that learns declares these at every agent's birth, so what an
    agent can learn is part of what it was born as rather than something that
    appeared when an estimator was attached.
    """
    return (
        *weight_names(features),
        *(f"{DRIFT_PREFIX}{name}" for name in features),
        *(f"{ENERGY_PREFIX}{name}" for name in features),
        *(f"{SCALE_TOTAL_PREFIX}{name}" for name in features),
        *(f"{SCALE_SQUARES_PREFIX}{name}" for name in features),
        SCALE_MASS,
    )


def cold_start(features: Sequence[str], config: LearningConfig) -> Mapping[str, float]:
    """What every column above holds before the agent has been shown anything.

    The declared cold-start weight is seeded into the accumulator the weight is
    computed *from*, not only into the weight, so it is the starting point of
    the recursion rather than a value the first observation silently discards.
    """
    seed = -config.cold_start_weight * (
        config.step_offset / config.step_scale + config.squared_penalty
    )
    values: dict[str, float] = {SCALE_MASS: 0.0}
    for name in features:
        values[f"{WEIGHT_PREFIX}{name}"] = config.cold_start_weight
        values[f"{DRIFT_PREFIX}{name}"] = seed
        values[f"{ENERGY_PREFIX}{name}"] = 0.0
        values[f"{SCALE_TOTAL_PREFIX}{name}"] = 0.0
        values[f"{SCALE_SQUARES_PREFIX}{name}"] = 0.0
    return values


@dataclass(frozen=True)
class BoundsApplied:
    """Which agents had a declared bound act on them during one observation.

    Two flags rather than one, because the two bounds guard different failures
    and a report that merged them could not say which guardrail held.

    Attributes:
        state: One entry per agent: whether the gain had to be held down.
        weights: One entry per agent: whether the weight vector had to be.
    """

    state: NDArray[np.bool_]
    weights: NDArray[np.bool_]

    def any_applied(self) -> bool:
        """Whether either bound acted on anybody."""
        return bool(self.state.any() or self.weights.any())
