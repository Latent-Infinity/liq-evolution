"""CLI entry point for liq-evolution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="liquid evolution experiment and artifact CLI",
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
        "--detector-source",
        choices=("evolved", "trained"),
        default="evolved",
        help="Regime detector source. 'trained' requires a trusted Phase 1 artifact path.",
    )
    parser.add_argument(
        "--regime-classifier-path",
        type=Path,
        help="Trusted SVMRegimeClassifier artifact produced by this workspace.",
    )
    parser.add_argument(
        "--handoff-classifier-path",
        type=Path,
        help="Trusted Phase 1 hand-off classifier artifact for save/load parity checks.",
    )
    parser.add_argument(
        "--phase4-evidence-path",
        type=Path,
        help="BTC-derived detector_swap.json evidence path for Phase 4 paired evolution.",
    )
    parser.add_argument(
        "--phase4-output-dir",
        type=Path,
        help="Output directory for Phase 4 detector-source evolution artifacts.",
    )
    parser.add_argument(
        "--phase4-paired-comparison",
        action="store_true",
        help=(
            "Run both evolved and trained Phase 4 branches and write paired comparison artifacts. "
            "Without this flag, Phase 4 runs only --detector-source."
        ),
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
            overrides.setdefault("fitness", {})["stage_b_enabled"] = (
                namespace.use_backtest
            )

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


def _validate_detector_source(namespace: argparse.Namespace) -> None:
    if namespace.detector_source != "trained":
        return
    if (
        namespace.regime_classifier_path is None
        and namespace.handoff_classifier_path is None
    ):
        raise _CliConfigError(
            error_code="evo.cli.trained_detector_missing_artifact",
            message="--detector-source=trained requires a trusted classifier artifact path.",
            remediation=(
                "Pass --regime-classifier-path or --handoff-classifier-path pointing to a trusted "
                "SVMRegimeClassifier artifact produced by Phase 1 in this workspace. "
                "Do not load user-uploaded or unverified joblib files."
            ),
            context={"detector_source": namespace.detector_source},
        )


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
                f"({_SAFE_STAGE_A_BUDGET_LIMIT - 1})."
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
            context={
                "validation_errors": [{"msg": str(exc), "type": "configuration_error"}]
            },
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

        _validate_detector_source(namespace)
        config = _build_config(namespace)
        phase4_selected_result = None
        phase4_paired_result = None
        if (
            namespace.phase4_evidence_path is not None
            or namespace.phase4_output_dir is not None
        ):
            if (
                namespace.phase4_evidence_path is None
                or namespace.phase4_output_dir is None
            ):
                raise _CliConfigError(
                    error_code="evo.cli.phase4_paths_missing",
                    message="Phase 4 evolution requires both --phase4-evidence-path and --phase4-output-dir.",
                    remediation="Pass both paths or omit both Phase 4 flags.",
                    context={
                        "phase4_evidence_path": str(namespace.phase4_evidence_path)
                        if namespace.phase4_evidence_path
                        else None,
                        "phase4_output_dir": str(namespace.phase4_output_dir)
                        if namespace.phase4_output_dir
                        else None,
                    },
                )
            from liq.evolution.phase4_detector_swap import (
                run_detector_source,
                run_paired_detector_swap,
            )

            trusted_classifier_path = (
                namespace.handoff_classifier_path or namespace.regime_classifier_path
            )
            if namespace.phase4_paired_comparison:
                if trusted_classifier_path is None:
                    raise _CliConfigError(
                        error_code="evo.cli.phase4_trusted_artifact_missing",
                        message="Phase 4 paired comparison requires a trusted classifier artifact path.",
                        remediation=(
                            "Pass --handoff-classifier-path or --regime-classifier-path so paired "
                            "comparison verifies the classifier SHA-256 against Phase 4 evidence."
                        ),
                        context={"phase4_paired_comparison": True},
                    )
                phase4_paired_result = run_paired_detector_swap(
                    seed=config.seed,
                    evidence_path=namespace.phase4_evidence_path,
                    output_root=namespace.phase4_output_dir,
                    population_size=config.population_size,
                    generations=config.generations,
                    trusted_classifier_path=trusted_classifier_path,
                )
            else:
                selected_summary = run_detector_source(
                    seed=config.seed,
                    detector_source=namespace.detector_source,
                    evidence_path=namespace.phase4_evidence_path,
                    output_dir=namespace.phase4_output_dir
                    / namespace.detector_source
                    / f"seed-{config.seed}",
                    population_size=config.population_size,
                    generations=config.generations,
                    trusted_classifier_path=trusted_classifier_path,
                )
                phase4_selected_result = selected_summary.to_dict()
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": config.model_dump(),
                    "phase4_detector_source_evolution": phase4_selected_result,
                    "phase4_paired_evolution": phase4_paired_result,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except _CliConfigError as exc:
        print(
            json.dumps(_render_failure(exc), indent=2, sort_keys=True), file=sys.stderr
        )
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
