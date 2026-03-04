#!/usr/bin/env python3
"""Show stability-first regime preset settings."""

from __future__ import annotations

from pprint import pprint

from liq.evolution.presets import get_regime_preset


def main() -> None:
    preset = get_regime_preset("stability_first")
    print(f"preset={preset.name}")
    print(f"description={preset.description}")
    print(f"operator_mode={preset.operator_mode}")
    print("seed_templates:")
    for name in preset.seed_templates:
        print(f"  - {name}")
    print("stability_controls:")
    pprint(preset.stability_controls)
    print("evolution_config:")
    pprint(preset.evolution_config.model_dump(mode="json"))
    print("gp_runtime_config:")
    pprint(preset.gp_runtime_config.model_dump(mode="json"))


if __name__ == "__main__":
    main()
