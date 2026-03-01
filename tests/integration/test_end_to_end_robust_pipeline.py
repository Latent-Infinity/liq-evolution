"""End-to-end robustness tests for Phase 6 hardening."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.config import PrimitiveConfig
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    to_loss_form,
)
from liq.evolution.fitness.multifidelity import MultiFidelityFitnessEvaluator
from liq.evolution.qd.orchestrator import run_qd_evolution
from liq.evolution.primitives import prepare_evaluation_context
from liq.evolution.primitives.registry import build_trading_registry
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.evolution.qd_archive import QDArchive
from liq.gp.program.ast import TerminalNode
from liq.gp.program.eval import evaluate as evaluate_program
from liq.gp.types import FitnessResult


def _make_splits() -> list[WalkForwardSplit]:
    return [
        WalkForwardSplit(train=slice(0, 8), validate=slice(8, 12), test=slice(12, 16), slice_id="time_window:split_0"),
        WalkForwardSplit(train=slice(16, 24), validate=slice(24, 28), test=slice(28, 32), slice_id="time_window:split_1"),
        WalkForwardSplit(train=slice(32, 40), validate=slice(40, 44), test=slice(44, 48), slice_id="time_window:split_2"),
    ]


def _make_context() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=64))
    return prepare_evaluation_context(
        {
            "open": close - rng.uniform(0.0, 1.0, size=64),
            "high": close + rng.uniform(0.0, 1.0, size=64),
            "low": close - rng.uniform(0.0, 1.0, size=64),
            "close": close,
            "volume": rng.uniform(1_000.0, 10_000.0, size=64),
        }
    )


class _SplitAwareLevelEvaluator:
    """Simple level evaluator with walk-forward split slice scores and constraints."""

    def __init__(
        self,
        splits: list[WalkForwardSplit],
        *,
        scale: float = 1.0,
        inject_adversarial: bool = False,
    ) -> None:
        self._splits = splits
        self._scale = scale
        self._inject_adversarial = inject_adversarial
        self.calls: list[int] = []
        self.last_metadata: list[dict[str, Any]] = []

    @property
    def total_program_evals(self) -> int:
        return sum(self.calls)

    def evaluate(
        self,
        programs: list[TerminalNode],
        context: dict[str, Any],
    ) -> list[FitnessResult]:
        del context
        self.calls.append(len(programs))

        results: list[FitnessResult] = []
        for program in programs:
            base_score = float(program.size) * self._scale / 10.0

            split_scores: dict[str, float] = {}
            constraint_violations: dict[str, float] = {}
            split_metrics: dict[str, dict[str, float]] = {}
            for offset, split in enumerate(self._splits):
                split_key = f"{split.slice_id}:train"
                raw_cagr = base_score + (offset + 1) * 0.2
                raw_dd = 0.9 / (offset + 1 + base_score)

                split_scores[f"{split_key}:cagr"] = to_loss_form(raw_cagr, "maximize")
                split_scores[f"{split_key}:max_drawdown"] = to_loss_form(raw_dd, "minimize")
                split_scores[f"{split_key}:constraint:stability"] = 0.05 * (offset + 1)
                split_metrics[split.slice_id] = {"cagr": raw_cagr, "max_drawdown": raw_dd}

                if self._inject_adversarial and offset == 0:
                    violation = 0.25
                    split_scores[f"{split_key}:constraint:adversarial:high_spread"] = violation
                    constraint_violations[f"{split_key}:adversarial:high_spread"] = violation

            raw_objectives = tuple(split_scores.values())
            raw_objective = sum(
                value
                for key, value in split_scores.items()
                if not key.endswith("constraint:stability")
                and "constraint:adversarial" not in key
                and "max_drawdown" in key
            )
            raw_objective = raw_objective / len(self._splits)

            metadata = {
                "objectives": (raw_cagr,),  # used by existing strategy pipelines
                METADATA_KEY_PER_SPLIT_METRICS: split_metrics,
                METADATA_KEY_RAW_OBJECTIVES: raw_objectives,
                METADATA_KEY_SLICE_SCORES: split_scores,
                METADATA_KEY_BEHAVIOR_DESCRIPTORS: {BEHAVIOR_DESCRIPTOR_TURNOVER: min(1.0, base_score / 2.0)},
                METADATA_KEY_CONSTRAINT_VIOLATIONS: constraint_violations,
            }
            self.last_metadata.append(metadata)
            results.append(FitnessResult(objectives=(raw_objective,), metadata=metadata))

        return results


def _make_lexicase_config(seed: int) -> LiqGPConfig:
    return LiqGPConfig(
        population_size=20,
        max_depth=4,
        generations=2,
        seed=seed,
        tournament_size=3,
        elitism_count=2,
        selection_mode="lexicase",
        lexicase_downsample_policy="none",
        parsimony_mode="disabled",
        constant_opt_enabled=False,
        simplification_enabled=False,
        semantic_dedup_enabled=False,
    )


def _run_qd(*, seed: int, coverage_weight: float, evaluator) -> tuple:
    splits = _make_splits()
    registry = build_trading_registry(PrimitiveConfig(enable_liq_ta=False))
    config = _make_lexicase_config(seed=seed)
    context = _make_context()

    result = run_qd_evolution(
        registry=registry,
        config=config,
        evaluator=evaluator,
        context=context,
        behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        archive_bins_per_dim=4,
        archive_bin_capacity=2,
        coverage_weight=coverage_weight,
        coverage_interval=1,
        portfolio_size=4,
    )
    return result


def test_end_to_end_lexicase_qd_multifidelity_pipeline() -> None:
    splits = _make_splits()

    def _build_mf(scale: float, *, adversarial: bool = False) -> MultiFidelityFitnessEvaluator:
        level0 = _SplitAwareLevelEvaluator(
            splits=splits,
            scale=scale,
            inject_adversarial=adversarial,
        )
        level1 = _SplitAwareLevelEvaluator(splits=splits, scale=scale * 1.1)
        return MultiFidelityFitnessEvaluator(
            levels=(level0, level1),
            top_k_per_level=0.5,
            objective_directions=("maximize",),
            promotion_strategy="direction_aware_first",
        )

    mf_eval = _build_mf(1.0, adversarial=True)
    level0 = mf_eval._levels[0]  # type: ignore[assignment]
    level1 = mf_eval._levels[1]  # type: ignore[assignment]
    mf_result = _run_qd(seed=123, coverage_weight=0.8, evaluator=mf_eval)

    # Deterministic seed yields stable portfolio.
    mf_result_repeat = _run_qd(
        seed=123,
        coverage_weight=0.8,
        evaluator=_build_mf(1.0, adversarial=True),
    )
    assert [str(program) for program in mf_result.portfolio] == [
        str(program) for program in mf_result_repeat.portfolio
    ]
    assert mf_result.coverage_report["filled_bins"] >= 1
    assert mf_result.coverage_report["filled_bins"] <= mf_result.coverage_report["total_bins"]
    assert 0.0 <= mf_result.coverage_report["fill_ratio"] <= 1.0
    assert set(mf_result.coverage_report.keys()) == {
        "filled_bins",
        "total_bins",
        "fill_ratio",
        "dimension_histograms",
    }
    assert len(mf_result.portfolio) == 4
    assert set(mf_result.portfolio).issubset(set(mf_result.archive.elites()))
    assert mf_result.coverage_report["fill_ratio"] >= 0.0

    # Archive output is structurally valid and serializable.
    archive_payload = mf_result.archive.to_dict()
    assert archive_payload["n_dims"] == 1
    assert archive_payload["bins_per_dim"] == [4]
    assert archive_payload["bin_capacity"] == 2
    assert len(archive_payload["entries"]) >= mf_result.coverage_report["filled_bins"]

    flat_eval = _SplitAwareLevelEvaluator(splits, scale=1.0)
    flat_result = _run_qd(seed=123, coverage_weight=0.0, evaluator=flat_eval)

    # Coverage-pressure path should not decrease exploration.
    assert mf_result.coverage_report["fill_ratio"] >= flat_result.coverage_report["fill_ratio"]

    # Multi-fidelity must evaluate fewer expensive programs than the flat strategy.
    assert mf_result.coverage_report["filled_bins"] >= flat_result.coverage_report["filled_bins"]
    assert mf_result.coverage_report["fill_ratio"] >= flat_result.coverage_report["fill_ratio"]
    assert level1.total_program_evals < flat_eval.total_program_evals

    # Constraint/lexicase case injection is present in level-0 and therefore selectable.
    keys = {
        key
        for metadata in level0.last_metadata
        for key in metadata[METADATA_KEY_SLICE_SCORES].keys()
    }
    assert any("adversarial" in key for key in keys)
    assert any("constraint" in key for key in keys)
    assert all(
        METADATA_KEY_PER_SPLIT_METRICS in metadata for metadata in level0.last_metadata
    )
    assert all(
        METADATA_KEY_RAW_OBJECTIVES in metadata for metadata in level0.last_metadata
    )

    # Programs are evaluable even for the same config after full pipeline run.
    assert evaluate_program(
        mf_result.portfolio[0],
        _make_context(),
    ).shape == (64,)


def test_coverage_pressure_prefers_underfilled_bins() -> None:
    archive = QDArchive(
        n_dims=1,
        bins_per_dim=4,
        descriptor_bounds=((0.0, 1.0),),
        objective_directions=["maximize"],
        bin_capacity=2,
    )
    top_a = TerminalNode(name="top_a", output_type=float)
    top_b = TerminalNode(name="top_b", output_type=float)
    under = TerminalNode(name="under", output_type=float)

    archive.insert(top_a, (1.0,), (0.08,))
    archive.insert(top_b, (1.0,), (0.08,))
    archive.insert(under, (0.80,), (0.60,))

    rng_random = np.random.default_rng(0)
    rng_coverage = np.random.default_rng(0)

    random_samples = archive.sample(6, rng_random, coverage_weight=0.0)
    coverage_samples = archive.sample(6, rng_coverage, coverage_weight=1.0)

    assert set(p.name for p in random_samples) == {"top_a", "top_b"}  # type: ignore[attr-defined]
    assert set(p.name for p in coverage_samples) == {"under"}  # type: ignore[attr-defined]
