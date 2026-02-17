"""Objective function wiring for the fitness pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from liq.evolution.config import FitnessConfig, FitnessStageConfig


def wire_objectives(
    config: FitnessConfig | FitnessStageConfig,
) -> Any:
    """Wire up objective functions based on fitness stage configuration.

    Args:
        config: Fitness stage configuration.

    Returns:
        Configured objective pipeline.
    """
    raise NotImplementedError
