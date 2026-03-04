"""Stage-17 tests for regime operational presets."""

from __future__ import annotations

import pytest

from liq.evolution.presets import get_regime_preset, list_regime_presets


def test_list_regime_presets_is_deterministic() -> None:
    assert list_regime_presets() == ("baseline", "stability_first")


def test_baseline_preset_exposes_seed_templates_operator_mode_and_controls() -> None:
    preset = get_regime_preset("baseline")

    assert preset.name == "baseline"
    assert preset.seed_templates
    assert preset.operator_mode == "standard"
    assert preset.gp_runtime_config.crossover_mode == "standard"
    assert preset.stability_controls["walkforward_embargo_bars"] == 1
    assert preset.stability_controls["regime_min_persistence"] >= 1


def test_stability_first_preset_uses_module_preserving_mode() -> None:
    preset = get_regime_preset("stability_first")

    assert preset.name == "stability_first"
    assert preset.seed_templates
    assert preset.operator_mode == "module_preserving"
    assert preset.gp_runtime_config.crossover_mode == "module_preserving"
    assert preset.stability_controls["objective_stability_weight"] > 0.2
    assert preset.evolution_config.regime.regime_confidence_threshold >= 0.6


def test_get_regime_preset_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get_regime_preset("not-a-preset")


def test_get_regime_preset_is_repeatable() -> None:
    first = get_regime_preset("baseline")
    second = get_regime_preset("baseline")

    assert first.seed_templates == second.seed_templates
    assert first.operator_mode == second.operator_mode
    assert first.stability_controls == second.stability_controls
    assert first.gp_runtime_config == second.gp_runtime_config
