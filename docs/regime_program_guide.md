# Regime Program Guide

This guide documents regime-aware operational presets.

## Preset API

Use `liq.evolution.presets` to retrieve deterministic preset bundles:

```python
from liq.evolution.presets import get_regime_preset, list_regime_presets

names = list_regime_presets()
baseline = get_regime_preset("baseline")
stability_first = get_regime_preset("stability_first")
```

Each `RegimeOperationalPreset` exposes:

- `seed_templates`: default seeded template set
- `operator_mode`: `"standard"` or `"module_preserving"`
- `stability_controls`: operational stability knobs
- `evolution_config`: canonical `EvolutionConfig`
- `gp_runtime_config`: mapped runtime `liq.gp.GPConfig`

## Runnable examples

Run both examples from the repository root:

```bash
python liq-evolution/examples/baseline_regime_preset_demo.py
python liq-evolution/examples/stability_first_regime_preset_demo.py
```

These scripts print preset metadata, seed templates, stability controls, and fully
materialized config payloads for review.

## Operational guidance

- Use `baseline` for balanced search with standard crossover.
- Use `stability_first` for lower-churn evolution with module-preserving crossover.
- Pin a deterministic seed in `EvolutionConfig` for reproducible replay.
- Treat this preset layer as the operational entrypoint for regime-aware runs.
