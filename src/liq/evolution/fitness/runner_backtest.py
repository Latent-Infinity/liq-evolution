"""Backtest-based fitness evaluation via liq-runner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any, Literal

import numpy as np
import polars as pl

from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.evaluation_schema import (
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
)
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate as gp_evaluate
from liq.gp.types import FitnessResult


class _ProgramStrategy:
    """Wraps a GP Program as a duck-type liq-runner Strategy."""

    def __init__(self, program: Program) -> None:
        self._program = program

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        pass  # No-op: program is already evolved

    def predict(self, features: pl.DataFrame) -> GPSignalOutput:
        context = {col: features[col].to_numpy() for col in features.columns}
        scores_array = gp_evaluate(self._program, context)
        return GPSignalOutput(scores=pl.Series("scores", scores_array))


class BacktestFitnessEvaluator:
    """Evaluates GP programs using backtested trading performance.

    Uses dependency injection for the backtest runner function.
    """

    OBJECTIVE_VECTOR_VERSION = "stage_b_v2"
    OBJECTIVE_NAMES = (
        "return",
        "drawdown_risk",
        "turnover",
        "regime_stability",
        "walk_forward_stability",
        "regime_coverage",
        "complexity_penalty",
    )
    OBJECTIVE_DIRECTIONS = (
        "maximize",
        "minimize",
        "minimize",
        "maximize",
        "maximize",
        "maximize",
        "minimize",
    )

    _TINY_REGIME_FRACTION = 0.2
    _MIN_TINY_REGIME_COUNT = 2
    _MAX_REGIME_CHAOS_RATIO = 0.75

    def __init__(
        self,
        backtest_runner: Callable[[Any], Sequence[dict[str, Any]]],
        metric: str = "sharpe_ratio",
        *,
        objective_mode: Literal["scalar", "vector"] = "scalar",
        simulator: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        risk_model: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        max_folds: int | None = None,
        max_runtime_seconds: float | None = None,
        memory_budget_mb: float | None = None,
    ) -> None:
        self._backtest_runner = backtest_runner
        self._metric = metric
        if objective_mode not in {"scalar", "vector"}:
            raise ValueError("objective_mode must be 'scalar' or 'vector'")
        if max_folds is not None and max_folds < 1:
            raise ValueError("max_folds must be >= 1 when provided")
        if max_runtime_seconds is not None and (
            max_runtime_seconds <= 0.0 or not np.isfinite(max_runtime_seconds)
        ):
            raise ValueError("max_runtime_seconds must be finite and > 0 when provided")
        if memory_budget_mb is not None and (
            memory_budget_mb <= 0.0 or not np.isfinite(memory_budget_mb)
        ):
            raise ValueError("memory_budget_mb must be finite and > 0 when provided")
        self._objective_mode = objective_mode
        self._simulator = simulator
        self._risk_model = risk_model
        self._max_folds = max_folds
        self._max_runtime_seconds = (
            float(max_runtime_seconds) if max_runtime_seconds is not None else None
        )
        self._memory_budget_mb = (
            float(memory_budget_mb) if memory_budget_mb is not None else None
        )

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],  # noqa: ARG002
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for program in programs:
            result = self._evaluate_single(program)
            results.append(result)
        return results

    def evaluate_fitness(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        return self.evaluate(programs, context)

    def _make_metadata(
        self,
        *,
        metric_value: float,
        n_folds: int,
        fold_values: list[float],
        objective_vector: tuple[float, ...],
        fold_objective_vectors: list[tuple[float, ...]],
        runtime_seconds: float,
        folds_truncated: bool,
        runtime_budget_exceeded: bool,
        reason: str | None = None,
        reason_code: str | None = None,
        regime_reason_codes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build metadata payload matching stage-0 contract."""
        fold_metrics = {
            f"fold:{idx}": {"metric": value} for idx, value in enumerate(fold_values)
        }
        metadata: dict[str, Any] = {
            "metric": self._metric,
            "n_folds": n_folds,
            "fold_values": fold_values,
            METADATA_KEY_PER_SPLIT_METRICS: {
                "all": {"metric": metric_value},
                **fold_metrics,
            },
            METADATA_KEY_RAW_OBJECTIVES: (metric_value,),
            "objective_vector_version": self.OBJECTIVE_VECTOR_VERSION,
            "objective_names": self.OBJECTIVE_NAMES,
            "objective_directions": self.OBJECTIVE_DIRECTIONS,
            "objective_vector": objective_vector,
            "fold_objective_vectors": fold_objective_vectors,
            "runtime_metrics": {
                "fold_count": n_folds,
                "metric_fold_count": len(fold_values),
                "objective_vector_count": len(fold_objective_vectors),
                "objective_mode": self._objective_mode,
                "evaluation_seconds": runtime_seconds,
                "max_folds": self._max_folds,
                "max_runtime_seconds": self._max_runtime_seconds,
                "memory_budget_mb": self._memory_budget_mb,
                "folds_truncated": folds_truncated,
                "runtime_budget_exceeded": runtime_budget_exceeded,
            },
            METADATA_KEY_BEHAVIOR_DESCRIPTORS: {},
            METADATA_KEY_CONSTRAINT_VIOLATIONS: {},
            METADATA_KEY_SLICE_SCORES: {},
        }
        if regime_reason_codes is not None:
            metadata["regime_objective_reason_codes"] = tuple(regime_reason_codes)
        if len(objective_vector) > 2:
            metadata[METADATA_KEY_BEHAVIOR_DESCRIPTORS] = {
                "turnover": float(objective_vector[2]),
            }
        if reason is not None:
            metadata["reason"] = reason
        if reason_code is not None:
            metadata["reason_code"] = reason_code
        return metadata

    @staticmethod
    def _metric_value(
        metrics: Mapping[str, Any],
        *keys: str,
        default: float = 0.0,
    ) -> float:
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                return parsed
        return float(default)

    @staticmethod
    def _metric_value_optional(
        metrics: Mapping[str, Any],
        *keys: str,
    ) -> float | None:
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                return parsed
        return None

    @staticmethod
    def _safe_ratio(value: float, total: float) -> float:
        if total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, value / total))

    @staticmethod
    def _extract_regime_trace(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, pl.Series):
            return value.to_list()
        if isinstance(value, np.ndarray):
            return list(value.tolist())
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    @staticmethod
    def _trace_stability(trace: list[Any]) -> float:
        n = len(trace)
        if n <= 1:
            return 0.0
        transitions = sum(
            1 for left, right in zip(trace[:-1], trace[1:], strict=False) if left != right
        )
        return 1.0 - transitions / max(1.0, float(n - 1))

    @staticmethod
    def _trace_churn_ratio(trace: list[Any]) -> float:
        n = len(trace)
        if n <= 1:
            return 0.0
        transitions = sum(
            1 for left, right in zip(trace[:-1], trace[1:], strict=False) if left != right
        )
        return transitions / max(1.0, float(n - 1))

    @staticmethod
    def _trace_coverage(trace: list[Any]) -> float:
        n = len(trace)
        if n <= 0:
            return 0.0
        segments = 1
        for left, right in zip(trace[:-1], trace[1:], strict=False):
            if left != right:
                segments += 1
        return segments / float(n)

    @classmethod
    def _score_regime_coverage_from_trace(cls, trace: list[Any]) -> float:
        return cls._trace_coverage(trace)

    def _extract_fold_regime_fields(
        self,
        metrics: Mapping[str, Any],
        *,
        reason_codes: list[str],
    ) -> dict[str, Any]:
        trace = self._extract_regime_trace(metrics.get("regime_trace"))

        explicit_stability = self._metric_value_optional(
            metrics, "regime_stability"
        )
        regime_penalty = self._metric_value_optional(
            metrics, "regime_penalty", "regime_robustness_penalty"
        )
        explicit_coverage = self._metric_value_optional(
            metrics, "regime_coverage", "regime_coverage_score"
        )
        explicit_regime_stability = explicit_stability is not None
        explicit_regime_stability = (
            explicit_regime_stability or regime_penalty is not None
        )
        explicit_regime_coverage = explicit_coverage is not None
        explicit_walk_forward_stability = self._metric_value_optional(
            metrics, "walk_forward_stability"
        )

        regime_stability = explicit_stability
        if regime_stability is None:
            if regime_penalty is not None:
                regime_stability = max(0.0, 1.0 - regime_penalty)
            elif trace is not None and trace:
                regime_stability = self._trace_stability(trace)
            else:
                reason_codes.append("missing_regime_evidence")
                regime_stability = 0.0

        if explicit_coverage is None:
            if trace is not None and trace:
                explicit_coverage = self._score_regime_coverage_from_trace(trace)
            else:
                explicit_coverage = 0.0

        if explicit_walk_forward_stability is None:
            explicit_walk_forward_stability = float("nan")

        return {
            "trace": trace,
            "regime_stability": float(regime_stability),
            "regime_coverage": float(explicit_coverage),
            "walk_forward_stability": float(explicit_walk_forward_stability),
            "regime_stability_explicit": explicit_regime_stability,
            "regime_coverage_explicit": explicit_regime_coverage,
        }

    def _compose_fold_payload(
        self,
        strategy: Any,
        fold: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(fold)
        base_metrics_raw = fold.get("metrics", {})
        base_metrics = dict(base_metrics_raw) if isinstance(base_metrics_raw, Mapping) else {}

        def _merge_component(component: Mapping[str, Any] | None) -> None:
            if component is None:
                return
            component_metrics_raw = component.get("metrics", component)
            if isinstance(component_metrics_raw, Mapping):
                for key, value in component_metrics_raw.items():
                    base_metrics[str(key)] = value

        if self._simulator is not None:
            _merge_component(self._simulator(strategy, fold))
        if self._risk_model is not None:
            _merge_component(self._risk_model(strategy, fold))

        merged["metrics"] = base_metrics
        return merged

    def _extract_fold_objective_vector(
        self,
        fold: Mapping[str, Any],
        *,
        program: Program,
        regime_fields: Mapping[str, Any],
    ) -> tuple[float, ...]:
        metrics = fold.get("metrics", {}) if isinstance(fold.get("metrics"), Mapping) else {}

        ret = self._metric_value(
            metrics,
            "total_return",
            "return",
            self._metric,
            default=0.0,
        )
        friction = (
            self._metric_value(metrics, "transaction_cost")
            + self._metric_value(metrics, "slippage")
            + self._metric_value(metrics, "spread_cost")
            + self._metric_value(metrics, "commission")
        )
        execution_penalty = self._metric_value(
            metrics,
            "execution_cap_violation",
            "execution_cap_violations",
            "execution_penalty",
        )
        return_objective = ret - friction - execution_penalty

        drawdown_risk = self._metric_value(metrics, "max_drawdown", "drawdown", "max_dd")
        turnover = (
            self._metric_value(metrics, "turnover", "avg_turnover")
            + execution_penalty
        )
        complexity_penalty = self._metric_value(
            metrics,
            "complexity_penalty",
            default=float(program.size),
        )

        return (
            float(return_objective),
            float(drawdown_risk),
            float(turnover),
            float(regime_fields["regime_stability"]),
            float(regime_fields["walk_forward_stability"]),
            float(regime_fields["regime_coverage"]),
            float(complexity_penalty),
        )

    def _compute_regime_stability_penalty(
        self,
        regime_records: list[Mapping[str, Any]],
        *,
        reason_codes: list[str],
    ) -> float:
        if not regime_records:
            reason_codes.append("missing_regime_trace")
            return 0.0

        if all(
            record.get("regime_stability_explicit", False) for record in regime_records
        ):
            base_stability = float(
                np.mean([record["regime_stability"] for record in regime_records])
            )
            return base_stability

        tiny_ratio = 0.0
        churn_ratio = 0.0
        has_trace = False
        for record in regime_records:
            trace = record.get("trace")
            if not trace:
                reason_codes.append("empty_regime_trace")
                continue
            has_trace = True
            n = len(trace)
            counts = Counter(trace)
            minimum_required = max(
                self._MIN_TINY_REGIME_COUNT,
                int(np.ceil(self._TINY_REGIME_FRACTION * n)),
            )
            min_count = min(counts.values()) if counts else n
            if min_count < minimum_required:
                tiny_ratio = max(
                    tiny_ratio,
                    1.0 - min(1.0, min_count / max(1.0, float(minimum_required))),
                )

            churn_ratio = max(churn_ratio, self._trace_churn_ratio(trace))

        base_stability = float(np.mean([record["regime_stability"] for record in regime_records]))

        if tiny_ratio > 0.0:
            reason_codes.append("small_regime_samples")
            if churn_ratio > self._MAX_REGIME_CHAOS_RATIO:
                reason_codes.append("excessive_churn")
        if not has_trace:
            reason_codes.append("regime_trace_missing")

        penalty = min(1.0, (tiny_ratio * 1.2 + churn_ratio * 0.2))
        return max(0.0, base_stability * (1.0 - penalty))

    def _compute_regime_coverage_penalty(
        self,
        regime_records: list[Mapping[str, Any]],
        *,
        reason_codes: list[str],
    ) -> float:
        explicit_coverages = [
            record["regime_coverage"]
            for record in regime_records
            if record.get("regime_coverage_explicit", False)
        ]
        if explicit_coverages and len(explicit_coverages) == len(regime_records):
            return float(np.mean(explicit_coverages))
        if explicit_coverages:
            reason_codes.append("mixed_regime_coverage_sources")

        traces = [record["trace"] for record in regime_records]
        non_empty = [trace for trace in traces if trace]
        if not non_empty:
            reason_codes.append("regime_coverage_missing")
            return 0.0

        global_unique = sorted({str(item) for trace in non_empty for item in set(trace)})
        if len(global_unique) <= 1:
            reason_codes.append("single_regime_overfit")
            return 0.0

        return float(np.mean([self._trace_coverage(trace) for trace in non_empty]))

    def _compute_walk_forward_stability(
        self,
        return_values: list[float],
        drawdown_values: list[float],
        regime_records: list[Mapping[str, Any]],
        *,
        reason_codes: list[str],
    ) -> float:
        explicit_scores = [
            float(score)
            for score in [
                record.get("walk_forward_stability") for record in regime_records
            ]
            if score is not None and np.isfinite(score)
        ]
        if explicit_scores:
            return float(np.mean(explicit_scores))

        if len(return_values) < 2:
            reason_codes.append("walk_forward_insufficient_folds")
            return 1.0

        valid_returns = [value for value in return_values if np.isfinite(value)]
        valid_drawdowns = [value for value in drawdown_values if np.isfinite(value)]
        if len(valid_returns) < 2 or len(valid_drawdowns) < 2:
            reason_codes.append("walk_forward_insufficient_folds")
            return 1.0

        return_std = float(np.std(valid_returns))
        drawdown_std = float(np.std(valid_drawdowns))

        return_scale = float(np.mean(np.abs(valid_returns)))
        drawdown_scale = float(np.mean(np.abs(valid_drawdowns)))

        return_cv = return_std / return_scale if return_scale > 0 else 0.0
        drawdown_cv = drawdown_std / drawdown_scale if drawdown_scale > 0 else 0.0

        stability = 1.0 - min(1.0, (abs(return_cv) + abs(drawdown_cv)) / 2.0)
        return max(0.0, float(stability))

    def _evaluate_single(self, program: Program) -> FitnessResult:
        strategy = _ProgramStrategy(program)
        started_at = time.perf_counter()

        try:
            fold_results = self._backtest_runner(strategy)
        except Exception as exc:
            raise FitnessEvaluationError(f"Backtest runner failed: {exc}") from exc

        fold_results_seq = list(fold_results)
        folds_truncated = False
        if self._max_folds is not None and len(fold_results_seq) > self._max_folds:
            fold_results_seq = fold_results_seq[: self._max_folds]
            folds_truncated = True

        runtime_seconds = float(time.perf_counter() - started_at)
        runtime_budget_exceeded = bool(
            self._max_runtime_seconds is not None
            and runtime_seconds > self._max_runtime_seconds
        )

        if not fold_results_seq:
            zero_vector = tuple(0.0 for _ in self.OBJECTIVE_NAMES)
            objective_value = (
                zero_vector if self._objective_mode == "vector" else (0.0,)
            )
            return FitnessResult(
                objectives=objective_value,
                metadata=self._make_metadata(
                    metric_value=0.0,
                    n_folds=0,
                    fold_values=[],
                    objective_vector=zero_vector,
                    fold_objective_vectors=[],
                    runtime_seconds=runtime_seconds,
                    folds_truncated=folds_truncated,
                    runtime_budget_exceeded=runtime_budget_exceeded,
                    reason="no_folds",
                    reason_code="no_folds",
                    regime_reason_codes=("no_folds",),
                ),
            )

        metric_values: list[float] = []
        fold_vectors: list[tuple[float, ...]] = []
        regime_records: list[dict[str, Any]] = []
        return_samples: list[float] = []
        drawdown_samples: list[float] = []
        reason_codes: list[str] = []

        for fold in fold_results_seq:
            composed_fold = self._compose_fold_payload(strategy, fold)
            metrics = (
                composed_fold.get("metrics", {})
                if isinstance(composed_fold.get("metrics"), Mapping)
                else {}
            )
            value = metrics.get(self._metric)
            if value is not None:
                try:
                    metric_values.append(float(value))
                except (TypeError, ValueError):
                    pass

            return_raw = self._metric_value_optional(
                metrics,
                "total_return",
                "return",
                self._metric,
            )
            drawdown_raw = self._metric_value_optional(
                metrics,
                "max_drawdown",
                "drawdown",
                "max_dd",
            )
            if return_raw is not None:
                return_samples.append(return_raw)
            if drawdown_raw is not None:
                drawdown_samples.append(drawdown_raw)

            fold_regime_fields = self._extract_fold_regime_fields(
                metrics,
                reason_codes=reason_codes,
            )
            regime_records.append(fold_regime_fields)
            fold_vectors.append(
                self._extract_fold_objective_vector(
                    composed_fold,
                    program=program,
                    regime_fields=fold_regime_fields,
                )
            )

        vector_array = np.asarray(fold_vectors, dtype=np.float64)
        objective_vector = tuple(np.mean(vector_array, axis=0).tolist())

        regime_stability = self._compute_regime_stability_penalty(
            regime_records,
            reason_codes=reason_codes,
        )
        walk_forward_stability = self._compute_walk_forward_stability(
            return_samples,
            drawdown_samples,
            regime_records,
            reason_codes=reason_codes,
        )
        regime_coverage = self._compute_regime_coverage_penalty(
            regime_records,
            reason_codes=reason_codes,
        )

        objective_vector = (
            objective_vector[0],
            objective_vector[1],
            objective_vector[2],
            float(regime_stability),
            float(walk_forward_stability),
            float(regime_coverage),
            objective_vector[6],
        )

        if not metric_values:
            objective_value = (
                objective_vector if self._objective_mode == "vector" else (0.0,)
            )
            return FitnessResult(
                objectives=objective_value,
                metadata=self._make_metadata(
                    metric_value=0.0,
                    n_folds=len(fold_results_seq),
                    fold_values=[],
                    objective_vector=objective_vector,
                    fold_objective_vectors=fold_vectors,
                    runtime_seconds=runtime_seconds,
                    folds_truncated=folds_truncated,
                    runtime_budget_exceeded=runtime_budget_exceeded,
                    reason="metric_missing",
                    reason_code="metric_missing",
                    regime_reason_codes=tuple(dict.fromkeys(reason_codes)),
                ),
            )

        avg_metric = float(np.mean(metric_values))
        objective_value = (
            objective_vector if self._objective_mode == "vector" else (avg_metric,)
        )
        return FitnessResult(
            objectives=objective_value,
            metadata=self._make_metadata(
                metric_value=avg_metric,
                n_folds=len(fold_results_seq),
                fold_values=metric_values,
                objective_vector=objective_vector,
                fold_objective_vectors=fold_vectors,
                runtime_seconds=runtime_seconds,
                folds_truncated=folds_truncated,
                runtime_budget_exceeded=runtime_budget_exceeded,
                regime_reason_codes=tuple(dict.fromkeys(reason_codes)),
            ),
        )
