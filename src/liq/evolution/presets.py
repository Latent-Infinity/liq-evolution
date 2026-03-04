"""Operational presets for regime-aware evolution workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from liq.evolution.config import (
    EvolutionConfig,
    GPConfig,
    RegimeGateConfig,
    build_gp_config,
)
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.config import SeedInjectionConfig


@dataclass(frozen=True)
class RegimeOperationalPreset:
    """Named operational preset with deterministic regime-aware settings."""

    name: str
    description: str
    seed_templates: tuple[str, ...]
    operator_mode: Literal["standard", "module_preserving"]
    stability_controls: dict[str, int | float | bool]
    evolution_config: EvolutionConfig
    gp_runtime_config: LiqGPConfig


def _baseline_preset() -> RegimeOperationalPreset:
    seed_templates = (
        "ema_crossover",
        "rsi_oversold",
        "carry_spread_expansion",
    )
    evolution_config = EvolutionConfig(
        population_size=120,
        max_depth=6,
        generations=20,
        seed=17,
        gp=GPConfig(
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_count=3,
            tournament_size=4,
            constant_opt_enabled=True,
            seed_injection=SeedInjectionConfig(interval=5, count=2, method="variation"),
        ),
        regime=RegimeGateConfig(
            regime_confidence_threshold=0.55,
            regime_occupancy_threshold=0.10,
            regime_hysteresis_margin=0.05,
            regime_min_persistence=2,
        ),
    )
    gp_runtime = build_gp_config(evolution_config).model_copy(
        update={"crossover_mode": "standard"}
    )
    stability_controls = {
        "walkforward_embargo_bars": 1,
        "regime_hysteresis_margin": evolution_config.regime.regime_hysteresis_margin,
        "regime_min_persistence": evolution_config.regime.regime_min_persistence,
        "objective_stability_weight": 0.2,
    }
    return RegimeOperationalPreset(
        name="baseline",
        description="Balanced exploration/exploitation baseline regime preset.",
        seed_templates=seed_templates,
        operator_mode="standard",
        stability_controls=stability_controls,
        evolution_config=evolution_config,
        gp_runtime_config=gp_runtime,
    )


def _stability_first_preset() -> RegimeOperationalPreset:
    seed_templates = (
        "regime_switching_trend_mean_reversion",
        "carry_spread_contraction",
        "atr_volatility_spike",
    )
    evolution_config = EvolutionConfig(
        population_size=140,
        max_depth=6,
        generations=28,
        seed=23,
        gp=GPConfig(
            mutation_rate=0.15,
            crossover_rate=0.85,
            elitism_count=5,
            tournament_size=5,
            constant_opt_enabled=True,
            constant_opt_max_time_seconds=2.0,
            seed_injection=SeedInjectionConfig(interval=3, count=3, method="variation"),
        ),
        regime=RegimeGateConfig(
            regime_confidence_threshold=0.65,
            regime_occupancy_threshold=0.15,
            regime_hysteresis_margin=0.08,
            regime_min_persistence=3,
        ),
    )
    gp_runtime = build_gp_config(evolution_config).model_copy(
        update={"crossover_mode": "module_preserving"}
    )
    stability_controls = {
        "walkforward_embargo_bars": 2,
        "regime_hysteresis_margin": evolution_config.regime.regime_hysteresis_margin,
        "regime_min_persistence": evolution_config.regime.regime_min_persistence,
        "objective_stability_weight": 0.35,
    }
    return RegimeOperationalPreset(
        name="stability_first",
        description="Lower-churn preset prioritizing regime stability and safety.",
        seed_templates=seed_templates,
        operator_mode="module_preserving",
        stability_controls=stability_controls,
        evolution_config=evolution_config,
        gp_runtime_config=gp_runtime,
    )


_PRESET_BUILDERS = {
    "baseline": _baseline_preset,
    "stability_first": _stability_first_preset,
}


def list_regime_presets() -> tuple[str, ...]:
    """Return available preset names in deterministic order."""
    return tuple(sorted(_PRESET_BUILDERS.keys()))


def get_regime_preset(name: str) -> RegimeOperationalPreset:
    """Build and return a named regime operational preset."""
    normalized = name.strip().lower()
    try:
        builder = _PRESET_BUILDERS[normalized]
    except KeyError as exc:
        available = ", ".join(list_regime_presets())
        raise KeyError(f"unknown preset {name!r}; available: {available}") from exc
    return builder()


__all__ = ["RegimeOperationalPreset", "get_regime_preset", "list_regime_presets"]
