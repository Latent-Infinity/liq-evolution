"""CLI entry point for liq-evolution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from liq.evolution.config import EvolutionConfig
from liq.evolution.errors import ConfigurationError

_SAFE_STAGE_A_BUDGET_LIMIT = 25_000


class _CliConfigError(Exception):
    """Structured CLI validation error."""

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        remediation: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.remediation = remediation
        self.context = dict(context or {})


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that never writes raw usage/errors to stderr."""

    def error(self, message: str) -> NoReturn:
        raise argparse.ArgumentError(None, message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="liquid evolution configuration and diagnostic CLI",
        exit_on_error=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON or YAML evolution config payload",
    )
    parser.add_argument("--population-size", type=int, help="Top-level population_size")
    parser.add_argument("--max-depth", type=int, help="Top-level max_depth")
    parser.add_argument("--generations", type=int, help="Top-level generations")
    parser.add_argument("--seed", type=int, help="Top-level RNG seed")
    parser.add_argument("--batch-size", type=int, help="Override batch_size")
    parser.add_argument(
        "--full-eval-interval",
        type=int,
        help="Override full-eval interval",
    )

    parser.add_argument(
        "--stage-a-threshold",
        type=float,
        help="Override run.stage_a_threshold",
    )
    parser.add_argument(
        "--stage-a-candidate-budget",
        type=int,
        help="Override run.stage_a_candidate_budget",
    )
    parser.add_argument(
        "--stage-b-candidate-budget",
        type=int,
        help="Override run.stage_b_candidate_budget",
    )
    parser.add_argument(
        "--stage-a-min-candidates",
        type=int,
        help="Override run.stage_a_min_candidates",
    )
    parser.add_argument(
        "--stage-b-min-candidates",
        type=int,
        help="Override run.stage_b_min_candidates",
    )

    parser.add_argument(
        "--label-metric",
        choices=("f1", "precision_at_k", "accuracy"),
        help="Override fitness_stages.label_metric",
    )
    parser.add_argument(
        "--label-top-k",
        type=float,
        help="Override fitness_stages.label_top_k",
    )
    parser.add_argument(
        "--use-backtest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable stage-B backtest gate",
    )
    parser.add_argument(
        "--backtest-top-n",
        type=int,
        help="Override fitness_stages.backtest_top_n",
    )
    parser.add_argument(
        "--backtest-metric",
        choices=("sharpe_ratio", "sortino_ratio", "total_return"),
        help="Override fitness_stages.backtest_metric",
    )

    parser.add_argument(
        "--regime-confidence-threshold",
        type=float,
        help="Override regime.regime_confidence_threshold",
    )
    parser.add_argument(
        "--regime-occupancy-threshold",
        type=float,
        help="Override regime.regime_occupancy_threshold",
    )
    parser.add_argument(
        "--regime-hysteresis-margin",
        type=float,
        help="Override regime.regime_hysteresis_margin",
    )
    parser.add_argument(
        "--regime-min-persistence",
        type=int,
        help="Override regime.regime_min_persistence",
    )
    parser.add_argument(
        "--allow-expensive",
        action="store_true",
        help="Allow expensive budget combinations",
    )
    return parser


def _merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if value is None:
            continue

        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dicts(
                cast_mapping_to_dict(merged[key]),
                cast_mapping_to_dict(value),
            )
        else:
            merged[key] = value
    return merged


def cast_mapping_to_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    return {str(k): v for k, v in value.items()}


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise _CliConfigError(
            error_code="evo.cli.config_missing",
            message=f"Configuration file not found: {path!s}",
            remediation="Provide an existing --config path, then rerun.",
            context={"config_path": str(path)},
        )

    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise _CliConfigError(
            error_code="evo.cli.config_type_unsupported",
            message=(
                f"Unsupported config format '{path.suffix}'. Use .json, .yaml, or .yml."
            ),
            remediation=(
                "Convert the file to JSON or YAML and retry, "
                "or remove --config for defaults."
            ),
            context={"config_path": str(path)},
        )

    try:
        payload_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _CliConfigError(
            error_code="evo.cli.config_read_failed",
            message=f"Failed to read config file: {exc}",
            remediation="Check file permissions and retry.",
            context={"config_path": str(path)},
        ) from exc

    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise _CliConfigError(
                error_code="evo.cli.config_parse_error",
                message=f"Invalid JSON config payload: {exc}",
                remediation="Fix JSON syntax and rerun.",
                context={"config_path": str(path)},
            ) from exc
        if isinstance(raw, Mapping):
            return cast_mapping_to_dict(raw)
        raise _CliConfigError(
            error_code="evo.cli.config_schema_violation",
            message="JSON config must be an object.",
            remediation="Provide a top-level object with liq-evolution config keys.",
            context={"config_path": str(path)},
        )

    try:
        import yaml
    except Exception as exc:
        raise _CliConfigError(
            error_code="evo.cli.config_yaml_dependency",
            message="YAML config selected but PyYAML is not installed.",
            remediation="Install pyyaml or use a .json config file.",
            context={"config_path": str(path)},
        ) from exc

    try:
        raw = yaml.safe_load(payload_text)
    except Exception as exc:
        raise _CliConfigError(
            error_code="evo.cli.config_parse_error",
            message=f"Invalid YAML config payload: {exc}",
            remediation="Fix YAML syntax and rerun.",
            context={"config_path": str(path)},
        ) from exc

    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return cast_mapping_to_dict(raw)
    raise _CliConfigError(
        error_code="evo.cli.config_schema_violation",
        message="YAML config must be an object.",
        remediation="Provide a top-level object with liq-evolution config keys.",
        context={"config_path": str(path)},
    )


def _collect_overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if namespace.population_size is not None:
        overrides["population_size"] = namespace.population_size
    if namespace.max_depth is not None:
        overrides["max_depth"] = namespace.max_depth
    if namespace.generations is not None:
        overrides["generations"] = namespace.generations
    if namespace.seed is not None:
        overrides["seed"] = namespace.seed
    if namespace.batch_size is not None:
        overrides["batch_size"] = namespace.batch_size
    if namespace.full_eval_interval is not None:
        overrides["full_eval_interval"] = namespace.full_eval_interval

    run_overrides: dict[str, Any] = {}
    for key, value in (
        ("stage_a_threshold", namespace.stage_a_threshold),
        ("stage_a_candidate_budget", namespace.stage_a_candidate_budget),
        ("stage_b_candidate_budget", namespace.stage_b_candidate_budget),
        ("stage_a_min_candidates", namespace.stage_a_min_candidates),
        ("stage_b_min_candidates", namespace.stage_b_min_candidates),
    ):
        if value is not None:
            run_overrides[key] = value
    if run_overrides:
        overrides["run"] = run_overrides

    fitness_stage_overrides: dict[str, Any] = {}
    if namespace.label_metric is not None:
        fitness_stage_overrides["label_metric"] = namespace.label_metric
    if namespace.label_top_k is not None:
        fitness_stage_overrides["label_top_k"] = namespace.label_top_k
    if namespace.backtest_top_n is not None:
        fitness_stage_overrides["backtest_top_n"] = namespace.backtest_top_n
    if namespace.backtest_metric is not None:
        fitness_stage_overrides["backtest_metric"] = namespace.backtest_metric
    if namespace.use_backtest is not None:
        fitness_stage_overrides["use_backtest"] = namespace.use_backtest

    if fitness_stage_overrides:
        overrides["fitness_stages"] = fitness_stage_overrides
        if namespace.use_backtest is not None:
            overrides.setdefault("fitness", {})["stage_b_enabled"] = namespace.use_backtest

    regime_overrides: dict[str, Any] = {}
    for key, value in (
        ("regime_confidence_threshold", namespace.regime_confidence_threshold),
        ("regime_occupancy_threshold", namespace.regime_occupancy_threshold),
        ("regime_hysteresis_margin", namespace.regime_hysteresis_margin),
        ("regime_min_persistence", namespace.regime_min_persistence),
    ):
        if value is not None:
            regime_overrides[key] = value
    if regime_overrides:
        overrides["regime"] = regime_overrides

    return overrides


def _validate_runtime_safety(
    config: EvolutionConfig,
    *,
    allow_expensive: bool,
) -> None:
    if config.fitness_stages.use_backtest != config.fitness.stage_b_enabled:
        raise _CliConfigError(
            error_code="evo.cli.stage_b_mismatch",
            message="Stage-B flags are inconsistent between legacy and live fields.",
            remediation=(
                "Set both fitness_stages.use_backtest and fitness.stage_b_enabled "
                "to the same value."
            ),
            context={
                "fitness_stages.use_backtest": config.fitness_stages.use_backtest,
                "fitness.stage_b_enabled": config.fitness.stage_b_enabled,
            },
        )

    if allow_expensive:
        return

    if config.run.stage_a_candidate_budget >= _SAFE_STAGE_A_BUDGET_LIMIT:
        raise _CliConfigError(
            error_code="evo.cli.unsafe_budget",
            message=(
                "Stage-A candidate budget is above the safe guardrail "
                f"({_SAFE_STAGE_A_BUDGET_LIMIT-1})."
            ),
            remediation=(
                "Use a lower stage-a-candidate-budget or pass --allow-expensive "
                "if this run is intentionally long."
            ),
            context={"stage_a_candidate_budget": config.run.stage_a_candidate_budget},
        )


def _build_config(namespace: argparse.Namespace) -> EvolutionConfig:
    merged = _load_config_file(namespace.config) if namespace.config else {}
    merged = _merge_dicts(merged, _collect_overrides(namespace))

    try:
        config = EvolutionConfig(**merged)
    except ConfigurationError as exc:
        raise _CliConfigError(
            error_code="evo.cli.config_validation_error",
            message=f"EvolutionConfig validation failed: {exc}",
            remediation=(
                "Fix config values and rerun. See the error context for each field."
            ),
            context={"validation_errors": [{"msg": str(exc), "type": "configuration_error"}]},
        ) from exc
    except ValidationError as exc:
        raise _CliConfigError(
            error_code="evo.cli.config_validation_error",
            message=f"EvolutionConfig validation failed: {exc}",
            remediation=(
                "Fix config values and rerun. See the error context for each field."
            ),
            context={"validation_errors": exc.errors()},
        ) from exc

    _validate_runtime_safety(config, allow_expensive=namespace.allow_expensive)
    return config


def _render_failure(exc: _CliConfigError) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": exc.error_code,
        "error_message": exc.message,
        "remediation": exc.remediation,
        "context": exc.context,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the liq-evolution CLI."""
    try:
        try:
            namespace = _parse_args(argv)
        except argparse.ArgumentError as exc:
            raise _CliConfigError(
                error_code="evo.cli.argparse_error",
                message=f"Invalid CLI argument: {exc}",
                remediation="Use --help to inspect allowed arguments and value formats.",
                context={"argument": str(exc)},
            ) from exc
        except SystemExit as exc:
            if exc.code == 0:
                return 0
            raise _CliConfigError(
                error_code="evo.cli.argparse_exit",
                message="CLI parser exited with an unexpected non-zero code.",
                remediation=(
                    "Check CLI usage and values. Use --help for supported flags."
                ),
                context={"parser_exit_code": exc.code},
            ) from exc

        config = _build_config(namespace)
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": config.model_dump(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except _CliConfigError as exc:
        print(json.dumps(_render_failure(exc), indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        failure = _render_failure(
            _CliConfigError(
                error_code="evo.cli.unexpected_error",
                message=f"Unexpected error: {exc}",
                remediation="Retry with corrected inputs and report a regression bug.",
                context={"exception": exc.__class__.__name__},
            ),
        )
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
