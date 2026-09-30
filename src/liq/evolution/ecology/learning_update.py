"""Array update mechanics for the learned state of an ecology population."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.learning_state import (
    DRIFT_PREFIX,
    ENERGY_PREFIX,
    SCALE_MASS,
    SCALE_SQUARES_PREFIX,
    SCALE_TOTAL_PREFIX,
    BoundsApplied,
    declared_columns,
    weight_names,
)

#: Smallest divisor the arithmetic below will use, so a mass or a spread of
#: zero produces zero rather than a warning and a not-a-number.
_TINY = 1.0e-300


class OnlineUpdate:
    """FTRL-Proximal over a whole population at once, in the columns it is given.

    Holds no state of its own beyond the column indices and the configuration:
    every number it reads and writes lives in the population's learned-state
    array, so there is one place an agent's learning is stored and a checkpoint
    of that array is a checkpoint of the learner.
    """

    def __init__(
        self,
        *,
        config: LearningConfig,
        features: Sequence[str],
        column_of: Mapping[str, int],
    ) -> None:
        """Bind the update to the columns ``column_of`` says its names live in.

        Args:
            config: Every quantity the update is turned by.
            features: The feature names, in the order the population holds them.
            column_of: Learned-state name to its column in the population's
                array.

        Raises:
            KeyError: If a column the update owns was not declared.
        """
        self._config = config
        self._features = tuple(features)
        missing = tuple(
            name for name in declared_columns(self._features) if name not in column_of
        )
        if missing:
            raise KeyError(
                f"a population that learns must declare what the update writes; "
                f"{list(missing)} were not among its learned names"
            )
        self._weight_at = _columns(weight_names(self._features), column_of)
        self._drift_at = _columns(
            tuple(f"{DRIFT_PREFIX}{name}" for name in self._features), column_of
        )
        self._energy_at = _columns(
            tuple(f"{ENERGY_PREFIX}{name}" for name in self._features), column_of
        )
        self._total_at = _columns(
            tuple(f"{SCALE_TOTAL_PREFIX}{name}" for name in self._features), column_of
        )
        self._squares_at = _columns(
            tuple(f"{SCALE_SQUARES_PREFIX}{name}" for name in self._features),
            column_of,
        )
        self._mass_at = column_of[SCALE_MASS]

    @property
    def features(self) -> tuple[str, ...]:
        """The feature names this update reads, in column order."""
        return self._features

    def standardise(
        self,
        learned: NDArray[np.float64],
        raw: NDArray[np.float64],
        forgetting: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Scale ``raw`` by each agent's prior-bar statistics, then fold it in.

        The order is the whole of the causality claim. A value is centred and
        scaled by what came before it and only then joins the statistics, so
        the standardised value at a bar cannot depend on that bar.

        Args:
            learned: The population's learned-state array, written in place.
            raw: One row per agent of the feature values at this decision
                point.
            forgetting: Each agent's forgetting factor.

        Returns:
            The standardised values, one row per agent.
        """
        mass = learned[:, self._mass_at][:, None]
        total = learned[:, self._total_at]
        squares = learned[:, self._squares_at]
        seen = mass > 0.0
        mean = np.where(seen, total / np.maximum(mass, _TINY), 0.0)
        second = np.where(seen, squares / np.maximum(mass, _TINY), 0.0)
        spread = np.sqrt(np.maximum(second - mean * mean, 0.0))
        scaled = np.where(
            spread > 0.0, (raw - mean) / np.where(spread > 0.0, spread, 1.0), 0.0
        )
        decay = forgetting[:, None]
        learned[:, self._mass_at] = forgetting * learned[:, self._mass_at] + 1.0
        learned[:, self._total_at] = decay * total + raw
        learned[:, self._squares_at] = decay * squares + raw * raw
        return scaled

    def observe(
        self,
        learned: NDArray[np.float64],
        standardised: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        forgetting: NDArray[np.float64],
    ) -> BoundsApplied:
        """Move every agent's weights towards ``outcomes``, and hold the bounds.

        Args:
            learned: The population's learned-state array, written in place.
            standardised: What each agent consumed at the decision point the
                outcome belongs to.
            outcomes: What each agent realised over the bar that followed it.
            forgetting: Each agent's forgetting factor.

        Returns:
            BoundsApplied: Who had a declared bound act on them.
        """
        weights = learned[:, self._weight_at]
        drift = learned[:, self._drift_at]
        energy = learned[:, self._energy_at]

        surprise = np.einsum("af,af->a", standardised, weights) - outcomes
        gradient = surprise[:, None] * standardised
        curvature = (
            np.sqrt(energy + gradient * gradient) - np.sqrt(energy)
        ) / self._config.step_scale
        decay = forgetting[:, None]
        drift = decay * drift + gradient - curvature * weights
        energy = decay * energy + gradient * gradient

        energy, held_state = self._hold_the_gain(energy)
        weights = self._weights_from(drift, energy)
        drift, weights, held_weights = self._hold_the_weights(drift, energy, weights)

        learned[:, self._drift_at] = drift
        learned[:, self._energy_at] = energy
        learned[:, self._weight_at] = weights
        return BoundsApplied(state=held_state, weights=held_weights)

    def gains(self, learned: NDArray[np.float64]) -> NDArray[np.float64]:
        """The multiplier each agent would apply to its next surprise."""
        return self._gain_from(learned[:, self._energy_at])

    def weights(self, learned: NDArray[np.float64]) -> NDArray[np.float64]:
        """What each agent has learned, one row per agent, in feature order."""
        return learned[:, self._weight_at]

    def unfit_to_start(
        self, learned: NDArray[np.float64], carried: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """Which agents would start outside what their bounds admit.

        The norms are the ones :meth:`observe` holds, computed the same way, so
        an agent admitted here is one the bound would not have acted on. A
        weight that is not finite is unfit whatever its norm compares as. The
        gain is checked only where ``carried`` says the start was inherited,
        and it is unfit unless it is inside the state bound, so a gain that is
        not a number is unfit as well. A start built from a prior has its
        founding gain held by :func:`founding_energy`, never refused.
        """
        weights = self.weights(learned)
        return (
            ~np.isfinite(weights).all(axis=1)
            | (self.weight_norms(learned) > self._config.weight_norm_bound)
            | (carried & ~(self.state_norms(learned) <= self._config.state_bound))
        )

    def state_norms(self, learned: NDArray[np.float64]) -> NDArray[np.float64]:
        """How big each agent's gain is, as one number per agent."""
        return np.linalg.norm(self.gains(learned), axis=1)

    def weight_norms(self, learned: NDArray[np.float64]) -> NDArray[np.float64]:
        """How big each agent's learned weight vector is."""
        return np.linalg.norm(self.weights(learned), axis=1)

    def _gain_from(self, energy: NDArray[np.float64]) -> NDArray[np.float64]:
        """The gain implied by an accumulated squared gradient."""
        return _gain_from(self._config, energy)

    def _weights_from(
        self, drift: NDArray[np.float64], energy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The weights FTRL-Proximal's closed form gives for these accumulators."""
        penalty = self._config.absolute_penalty
        step = (
            self._config.step_offset + np.sqrt(energy)
        ) / self._config.step_scale + self._config.squared_penalty
        shrunk = np.sign(drift) * np.maximum(np.abs(drift) - penalty, 0.0)
        return np.where(np.abs(drift) <= penalty, 0.0, -shrunk / step)

    def _hold_the_gain(
        self, energy: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Raise accumulated evidence until the gain fits under its bound."""
        return _hold_the_gain(self._config, energy)

    def _hold_the_weights(
        self,
        drift: NDArray[np.float64],
        energy: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Scale back the accumulator until the weights fit under their bound.

        The accumulator is what the weights are computed from, so scaling it
        makes the reduction survive the next bar; scaling the weights alone is
        undone by the closed form as soon as anything else arrives. One pass is
        enough, a property of the closed form rather than an assumption about it.
        Scaling the accumulator by ``c`` scales each coordinate's numerator by
        at most ``c`` — exactly ``c`` with no absolute penalty, less where one
        penalty is subtracted after scaling — while the denominator, a function
        of accumulated evidence, does not move. So the new norm is at most ``c``
        times the old, and ``c`` is chosen to land it on the bound.
        """
        bound = self._config.weight_norm_bound
        size = np.linalg.norm(weights, axis=1)
        over = size > bound
        if not over.any():
            return drift, weights, over
        held_drift = drift.copy()
        held_drift[over] *= (bound / size[over])[:, None]
        return held_drift, self._weights_from(held_drift, energy), over


def founding_energy(
    config: LearningConfig, features: Sequence[str]
) -> tuple[Mapping[str, float], bool]:
    """The evidence an agent that has learned nothing is founded with, per feature.

    Zero, held under the state bound by the same hold every outcome applies,
    and there is no second formula. A newly founded agent's gain is the
    family's ceiling, ``step_scale / step_offset`` per feature. So under a
    bound at or above that ceiling the hold returns zero unchanged, and under a
    bound below it the founding evidence is the least the bound admits.

    Returns:
        The founding evidence by feature, and whether the bound had to act.
    """
    held, over = _hold_the_gain(config, np.zeros((1, len(features))))
    return dict(zip(features, held[0].tolist(), strict=True)), bool(over[0])


def _gain_from(
    config: LearningConfig, energy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The gain implied by an accumulated squared gradient."""
    return config.step_scale / (config.step_offset + np.sqrt(energy))


def _hold_the_gain(
    config: LearningConfig, energy: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Raise accumulated evidence until the gain fits under its bound.

    Acting on the gain rather than recording it is the point: an estimator
    told to behave as though it had seen more evidence than it has is an
    estimator whose next step is small, which is what the bound is *for*.
    Reaching the bound is reported by the flag, so a run does not go quiet
    about having been held.
    """
    bound = config.state_bound
    size = np.linalg.norm(_gain_from(config, energy), axis=1)
    over = size > bound
    if not over.any():
        return energy, over
    held = energy.copy()
    shrink = (bound / size[over])[:, None]
    root = (config.step_offset + np.sqrt(held[over])) / shrink
    held[over] = np.maximum(root - config.step_offset, 0.0) ** 2
    return held, over


def _columns(names: Sequence[str], column_of: Mapping[str, int]) -> NDArray[np.intp]:
    """The columns ``names`` occupy, for index-array arithmetic to slice."""
    return np.asarray([column_of[name] for name in names], dtype=np.intp)
