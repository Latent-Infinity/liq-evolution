"""What a run's configuration refuses to be.

The schema's interesting content is what it will not accept. A run with no
identity cannot be reproduced or withdrawn, and a run configured for a
discontinuity nothing implements would record a policy in its provenance that it
did not follow — which is worse than having no setting at all, because the
record would look deliberate.

That the schema exposes no epoch or replay key is asserted where the fact lives,
against the tape, in the consumer repository. It is not restated here: a second
copy of a claim is a second place for it to drift.
"""

from __future__ import annotations

import dataclasses

import pytest

from liq.evolution.ecology.config import (
    DEFAULT_HISTORY_CAPACITY,
    STATE_BOUND_KEY,
    WEIGHT_NORM_BOUND_KEY,
    EcologyConfig,
    LearningConfig,
    PopulationConfig,
)


def test_a_run_carries_an_identity() -> None:
    """A pass that cannot be named is refused where it is configured."""
    with pytest.raises(ValueError, match="must carry an identity"):
        EcologyConfig(run_id="")


def test_history_carries_across_a_split_boundary_unless_told_otherwise() -> None:
    """Continuity is the default, and it is the reading that is implemented."""
    assert EcologyConfig(run_id="named").window_carries_across_segments is True


def test_a_discontinuity_nothing_implements_is_refused_rather_than_ignored() -> None:
    """Asking for a restart at a fold fails; it is never accepted and disregarded."""
    with pytest.raises(ValueError, match="nothing implements that"):
        EcologyConfig(run_id="named", window_carries_across_segments=False)


def test_a_configuration_cannot_be_changed_once_a_run_is_driven_under_it() -> None:
    """One reader cannot re-point a run's configuration under another."""
    config = EcologyConfig(run_id="named")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.run_id = "renamed"  # type: ignore[misc]


def test_the_declared_losses_and_bounds_are_keys_of_the_run() -> None:
    """The ceiling and both bounds are configuration a run's digest covers.

    An outcome history's ceiling is a declaration that the oldest outcome will
    be dropped, and a state bound is a declaration about what a run would be
    stopped from doing. Neither is recoverable from a finished run if it lives
    as a default at a call site, so both are keys here and this is the check
    that says so.
    """
    declared = dataclasses.asdict(EcologyConfig(run_id="named"))
    assert declared["population"]["history_capacity"] == DEFAULT_HISTORY_CAPACITY
    assert set(declared["learning"]) == {
        "cold_start_weight",
        "forgetting_minimum",
        "forgetting_maximum",
        "state_bound",
        "weight_norm_bound",
        "step_scale",
        "step_offset",
        "absolute_penalty",
        "squared_penalty",
    }
    assert STATE_BOUND_KEY == "learning.state_bound"
    assert WEIGHT_NORM_BOUND_KEY == "learning.weight_norm_bound"


def test_an_outcome_history_that_retains_nothing_is_refused() -> None:
    """A ceiling of zero is not a small history, it is the absence of one."""
    with pytest.raises(ValueError, match="retains nothing"):
        PopulationConfig(history_capacity=0)


@pytest.mark.parametrize(
    ("changes", "refused"),
    [
        ({"forgetting_minimum": 0.0}, "strictly inside"),
        ({"forgetting_minimum": 0.99, "forgetting_maximum": 0.5}, "strictly inside"),
        ({"forgetting_maximum": 1.5}, "strictly inside"),
        ({"state_bound": 0.0}, "admits no state at all"),
        ({"weight_norm_bound": -1.0}, "admits no state at all"),
        ({"step_scale": 0.0}, "no step to take"),
        ({"step_offset": -1.0}, "no step to take"),
        ({"absolute_penalty": -1.0}, "rewards a larger weight"),
        ({"squared_penalty": -0.5}, "rewards a larger weight"),
    ],
)
def test_a_learning_configuration_that_means_nothing_is_refused(
    changes: dict[str, float], refused: str
) -> None:
    """Each declared quantity is checked where it is declared, not where it is used.

    A forgetting factor outside (0, 1] is not slow or fast learning, it is
    neither; a bound at or below zero admits no state, so nothing could run
    inside it; a step that divides by zero is not a step; and a negative
    penalty rewards exactly what regularisation exists to discourage.
    """
    with pytest.raises(ValueError, match=refused):
        LearningConfig(**changes)
