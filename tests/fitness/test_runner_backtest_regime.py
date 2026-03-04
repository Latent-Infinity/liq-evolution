"""Stage-3 Stage-B realistic objective vector tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series
from liq.sim.fx_eval import cvar_from_pnl


def _program() -> TerminalNode:
    return TerminalNode("close", Series)


def _runner(folds: Sequence[dict[str, Any]]):
    def _call(_strategy: Any) -> Sequence[dict[str, Any]]:
        return folds

    return _call


class TestStageBObjectiveVectorContract:
    def test_vector_shape_and_order_are_explicit(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": 0.12,
                    "max_drawdown": 0.05,
                    "turnover": 0.30,
                    "regime_stability": 0.73,
                    "walk_forward_stability": 0.80,
                    "regime_coverage": 0.65,
                    "complexity_penalty": 0.20,
                }
            }
        ]
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            objective_mode="vector",
        )

        [result] = evaluator.evaluate([_program()], {})
        assert len(result.objectives) == 7
        assert result.metadata["objective_vector_version"] == "stage_b_v2"
        assert result.metadata["objective_names"] == (
            "return",
            "drawdown_risk",
            "turnover",
            "regime_stability",
            "walk_forward_stability",
            "regime_coverage",
            "complexity_penalty",
        )

    @pytest.mark.parametrize(
        ("low_cost", "high_cost"),
        [(0.0, 0.005), (0.005, 0.02), (0.02, 0.04)],
        ids=["zero_to_tight", "tight_to_medium", "medium_to_high"],
    )
    def test_friction_degrades_return_objective(
        self,
        low_cost: float,
        high_cost: float,
    ) -> None:
        base = {
            "total_return": 0.10,
            "max_drawdown": 0.04,
            "turnover": 0.25,
            "regime_stability": 1.0,
            "walk_forward_stability": 0.9,
            "regime_coverage": 0.80,
            "complexity_penalty": 0.20,
        }
        low_folds = [
            {
                "metrics": {
                    **base,
                    "transaction_cost": low_cost,
                    "slippage": low_cost,
                }
            }
        ]
        high_folds = [
            {
                "metrics": {
                    **base,
                    "transaction_cost": high_cost,
                    "slippage": high_cost,
                }
            }
        ]
        low_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(low_folds),
            objective_mode="vector",
        )
        high_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(high_folds),
            objective_mode="vector",
        )

        [low_result] = low_eval.evaluate([_program()], {})
        [high_result] = high_eval.evaluate([_program()], {})
        assert float(low_result.objectives[0]) > float(high_result.objectives[0])

    def test_execution_caps_and_turnover_are_reflected(self) -> None:
        baseline = [
            {
                "metrics": {
                    "total_return": 0.08,
                    "max_drawdown": 0.03,
                    "turnover": 0.15,
                    "regime_stability": 0.90,
                    "walk_forward_stability": 0.80,
                    "regime_coverage": 0.70,
                    "complexity_penalty": 0.15,
                    "execution_cap_violation": 0.0,
                }
            }
        ]
        constrained = [
            {
                "metrics": {
                    "total_return": 0.08,
                    "max_drawdown": 0.03,
                    "turnover": 0.15,
                    "regime_stability": 0.90,
                    "walk_forward_stability": 0.80,
                    "regime_coverage": 0.70,
                    "complexity_penalty": 0.15,
                    "execution_cap_violation": 0.40,
                }
            }
        ]

        ev_base = BacktestFitnessEvaluator(
            backtest_runner=_runner(baseline),
            objective_mode="vector",
        )
        ev_constrained = BacktestFitnessEvaluator(
            backtest_runner=_runner(constrained),
            objective_mode="vector",
        )

        [base] = ev_base.evaluate([_program()], {})
        [constrained_result] = ev_constrained.evaluate([_program()], {})
        # Turnover objective is minimized, so higher value means degraded.
        assert constrained_result.objectives[2] > base.objectives[2]
        # Execution caps degrade return.
        assert constrained_result.objectives[0] < base.objectives[0]

    def test_simulator_and_risk_model_are_composed_into_fold_metrics(self) -> None:
        folds = [{"metrics": {"total_return": 0.10, "max_drawdown": 0.05}}]
        calls: list[str] = []

        def _simulator(strategy: Any, fold: dict[str, Any]) -> dict[str, Any]:
            del strategy, fold
            calls.append("sim")
            return {"metrics": {"slippage": 0.02, "turnover": 0.30}}

        def _risk_model(strategy: Any, fold: dict[str, Any]) -> dict[str, Any]:
            del strategy, fold
            calls.append("risk")
            return {
                "metrics": {
                    "regime_penalty": 0.25,
                    "complexity_penalty": 0.10,
                    "regime_coverage": 0.72,
                }
            }

        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            objective_mode="vector",
            simulator=_simulator,
            risk_model=_risk_model,
        )
        [result] = evaluator.evaluate([_program()], {})
        assert calls == ["sim", "risk"]
        # slippage + risk metrics should be reflected in vector
        assert result.objectives[0] < 0.10
        assert result.objectives[3] == 0.75
        assert result.objectives[5] == 0.72

    def test_walk_forward_and_regime_objectives_are_consistent(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": 0.07,
                    "max_drawdown": 0.03,
                    "turnover": 0.11,
                    "regime_penalty": 0.05,
                    "walk_forward_stability": 0.92,
                    "complexity_penalty": 0.10,
                    "regime_trace": [0, 0, 1, 1, 1, 0],
                }
            }
        ]
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            objective_mode="vector",
        )
        [result] = evaluator.evaluate([_program()], {})

        assert result.objectives[3] == 0.95
        assert result.objectives[4] == 0.92
        assert result.objectives[5] == pytest.approx(3 / 6)

    def test_objective_mode_validation_and_runtime_metrics(self) -> None:
        try:
            BacktestFitnessEvaluator(backtest_runner=_runner([]), objective_mode="bad")  # type: ignore[arg-type]
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "objective_mode" in str(exc)

        folds = [{"metrics": {"sharpe_ratio": 1.0}}]
        evaluator = BacktestFitnessEvaluator(backtest_runner=_runner(folds))
        [result] = evaluator.evaluate([_program()], {})
        runtime = result.metadata["runtime_metrics"]
        assert runtime["fold_count"] == 1
        assert runtime["objective_vector_count"] == 1
        assert runtime["objective_mode"] == "scalar"
        assert runtime["evaluation_seconds"] >= 0.0
        assert runtime["folds_truncated"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_folds": 0},
            {"max_runtime_seconds": 0.0},
            {"memory_budget_mb": -1.0},
        ],
        ids=["max_folds", "max_runtime_seconds", "memory_budget_mb"],
    )
    def test_runtime_bound_parameter_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            BacktestFitnessEvaluator(backtest_runner=_runner([]), **kwargs)

    def test_runtime_bounds_are_tracked_and_fold_count_bounded(self) -> None:
        folds = [
            {"metrics": {"sharpe_ratio": 1.0}},
            {"metrics": {"sharpe_ratio": 2.0}},
            {"metrics": {"sharpe_ratio": 3.0}},
        ]
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            max_folds=2,
            max_runtime_seconds=1e-12,
            memory_budget_mb=128.0,
        )
        [result] = evaluator.evaluate([_program()], {})
        runtime = result.metadata["runtime_metrics"]
        assert runtime["fold_count"] == 2
        assert runtime["folds_truncated"] is True
        assert runtime["max_folds"] == 2
        assert runtime["max_runtime_seconds"] == 1e-12
        assert runtime["memory_budget_mb"] == 128.0
        assert isinstance(runtime["runtime_budget_exceeded"], bool)

    def test_non_numeric_and_non_finite_metric_values_fallback(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": "not-a-number",
                    "return": float("inf"),
                    "sharpe_ratio": 0.5,
                    "max_drawdown": "nan",
                    "regime_coverage": 0.5,
                    "complexity_penalty": "bad",
                }
            }
        ]
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            objective_mode="vector",
        )
        [result] = evaluator.evaluate([_program()], {})
        # invalid total_return/return should fall back to sharpe ratio
        assert result.objectives[0] == 0.5


class TestStageBRegimeOverfitPenalties:
    def test_missing_regime_evidence_is_deterministic(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": 0.06,
                    "max_drawdown": 0.01,
                    "turnover": 0.10,
                    "complexity_penalty": 0.1,
                }
            }
        ]
        evaluator = BacktestFitnessEvaluator(backtest_runner=_runner(folds), objective_mode="vector")
        [result] = evaluator.evaluate([_program()], {})

        assert result.objectives[3] == 0.0
        assert result.objectives[5] == 0.0
        assert "missing_regime_evidence" in result.metadata["regime_objective_reason_codes"]
        # Single fold with explicit or implicit walk-forward data still stable by convention.
        assert result.objectives[4] == 1.0

    def test_tiny_regime_samples_receive_monotonic_penalty(self) -> None:
        folds_tiny = [
            {
                "metrics": {
                    "total_return": 0.08,
                    "max_drawdown": 0.02,
                    "turnover": 0.10,
                    "regime_trace": ["up", "up", "up", "up", "up", "down"],
                    "complexity_penalty": 0.10,
                }
            }
        ]
        folds_healthy = [
            {
                "metrics": {
                    "total_return": 0.08,
                    "max_drawdown": 0.02,
                    "turnover": 0.10,
                    "regime_trace": ["up", "up", "down", "down", "up", "down"],
                    "complexity_penalty": 0.10,
                }
            }
        ]
        tiny_eval = BacktestFitnessEvaluator(backtest_runner=_runner(folds_tiny), objective_mode="vector")
        healthy_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds_healthy), objective_mode="vector"
        )

        [tiny] = tiny_eval.evaluate([_program()], {})
        [healthy] = healthy_eval.evaluate([_program()], {})
        assert healthy.objectives[3] > tiny.objectives[3]

    def test_excessive_regime_churn_is_penalized(self) -> None:
        folds_excessive = [
            {
                "metrics": {
                    "total_return": 0.03,
                    "max_drawdown": 0.01,
                    "turnover": 0.1,
                    "regime_trace": [("a" if idx % 2 == 0 else "b") for idx in range(40)],
                    "complexity_penalty": 0.1,
                }
            }
        ]
        folds_stable = [
            {
                "metrics": {
                    "total_return": 0.03,
                    "max_drawdown": 0.01,
                    "turnover": 0.1,
                    "regime_trace": ["a"] * 40,
                    "complexity_penalty": 0.1,
                }
            }
        ]
        excessive_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds_excessive), objective_mode="vector"
        )
        stable_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds_stable), objective_mode="vector"
        )
        [excessive] = excessive_eval.evaluate([_program()], {})
        [stable] = stable_eval.evaluate([_program()], {})
        assert stable.objectives[3] > excessive.objectives[3]

    def test_single_regime_overfit_is_penalized_by_coverage(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": 0.06,
                    "max_drawdown": 0.01,
                    "turnover": 0.10,
                    "regime_trace": ["trend"] * 12,
                    "complexity_penalty": 0.10,
                }
            },
            {
                "metrics": {
                    "total_return": 0.06,
                    "max_drawdown": 0.01,
                    "turnover": 0.10,
                    "regime_trace": ["trend"] * 12,
                    "complexity_penalty": 0.10,
                }
            },
        ]
        evaluator = BacktestFitnessEvaluator(backtest_runner=_runner(folds), objective_mode="vector")
        [result] = evaluator.evaluate([_program()], {})

        assert result.objectives[5] == 0.0
        assert "single_regime_overfit" in result.metadata["regime_objective_reason_codes"]

    def test_walk_forward_stability_is_reduced_for_volatile_fold_returns(self) -> None:
        stable_folds = [
            {
                "metrics": {
                    "total_return": 0.05,
                    "max_drawdown": 0.01,
                    "turnover": 0.06,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
            {
                "metrics": {
                    "total_return": 0.06,
                    "max_drawdown": 0.015,
                    "turnover": 0.07,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
            {
                "metrics": {
                    "total_return": 0.04,
                    "max_drawdown": 0.012,
                    "turnover": 0.05,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
        ]
        volatile_folds = [
            {
                "metrics": {
                    "total_return": 0.05,
                    "max_drawdown": 0.06,
                    "turnover": 0.06,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
            {
                "metrics": {
                    "total_return": -0.03,
                    "max_drawdown": 0.01,
                    "turnover": 0.07,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
            {
                "metrics": {
                    "total_return": 0.02,
                    "max_drawdown": 0.08,
                    "turnover": 0.05,
                    "regime_stability": 0.95,
                    "regime_coverage": 0.9,
                    "regime_trace": ["up", "up", "down", "down"],
                    "complexity_penalty": 0.05,
                }
            },
        ]

        stable_eval = BacktestFitnessEvaluator(backtest_runner=_runner(stable_folds), objective_mode="vector")
        volatile_eval = BacktestFitnessEvaluator(
            backtest_runner=_runner(volatile_folds), objective_mode="vector"
        )
        [stable] = stable_eval.evaluate([_program()], {})
        [volatile] = volatile_eval.evaluate([_program()], {})
        assert stable.objectives[4] > volatile.objectives[4]

    def test_cvar_regime_penalty_transforms_consistently(self) -> None:
        folds = [
            {
                "metrics": {
                    "total_return": 0.09,
                    "max_drawdown": 0.02,
                    "turnover": 0.14,
                    "complexity_penalty": 0.20,
                    "regime_trace": ["up", "up", "down", "down"],
                }
            }
        ]
        expected_regime_penalty = abs(
            cvar_from_pnl([0.01, -0.02, 0.015, -0.01], 0.75)
        )
        folds[0]["metrics"]["regime_penalty"] = expected_regime_penalty

        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_runner(folds),
            objective_mode="vector",
        )
        [result] = evaluator.evaluate([_program()], {})
        assert result.objectives[3] == pytest.approx(1 - expected_regime_penalty)


class TestStageBReplayDeterminism:
    def test_candidate_ranking_replay_is_deterministic_at_scale(self) -> None:
        programs = [TerminalNode(name=f"p{i}", output_type=Series) for i in range(64)]

        def _deterministic_runner(strategy: Any) -> list[dict[str, Any]]:
            prog = strategy._program
            idx = int(prog.name[1:]) if hasattr(prog, "name") else 0
            ret = 1.0 - (idx / 100.0)
            return [
                {
                    "metrics": {
                        "total_return": ret,
                        "max_drawdown": idx / 1000.0,
                        "turnover": idx / 500.0,
                        "regime_stability": 0.90,
                        "walk_forward_stability": 0.95,
                        "regime_coverage": 0.75,
                        "complexity_penalty": float(getattr(prog, "size", 1)),
                    }
                }
            ]

        evaluator = BacktestFitnessEvaluator(
            backtest_runner=_deterministic_runner,
            objective_mode="vector",
        )

        first = evaluator.evaluate(programs, {})
        second = evaluator.evaluate(programs, {})
        rank1 = sorted(range(len(programs)), key=lambda i: first[i].objectives[0], reverse=True)
        rank2 = sorted(range(len(programs)), key=lambda i: second[i].objectives[0], reverse=True)
        assert rank1 == rank2
        assert all(r.metadata["runtime_metrics"]["fold_count"] == 1 for r in first)
