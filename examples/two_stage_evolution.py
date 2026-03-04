#!/usr/bin/env python3
"""Run a small two-stage evolution with pinned seed data."""

from __future__ import annotations

import numpy as np
import polars as pl

from liq.evolution import (
    EvolutionConfig,
    FitnessStageConfig,
    build_gp_config,
    evaluate,
    evolve,
    prepare_evaluation_context,
    wire_objectives,
)
from liq.evolution.primitives.ops_crossover import register_crossover_ops
from liq.evolution.primitives.ops_numeric import register_numeric_ops
from liq.evolution.primitives.series_sources import register_series_sources
from liq.evolution.primitives.ops_comparison import register_comparison_ops
from liq.evolution.primitives.ops_logic import register_logic_ops
from liq.gp.primitives.registry import PrimitiveRegistry


def generate_ohlcv(length: int, seed: int = 20260303) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, length)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, length))
    high = np.maximum(close, open_) * (1.0 + rng.uniform(0.001, 0.006, length))
    low = np.minimum(close, open_) * (1.0 - rng.uniform(0.001, 0.006, length))
    volume = rng.uniform(1_000_000.0, 7_000_000.0, length)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def make_labels(close: np.ndarray, horizon: int = 1) -> np.ndarray:
    labels = np.zeros(close.shape[0], dtype=np.float64)
    if close.size > horizon:
        labels[:-horizon] = (close[horizon:] > close[:-horizon]).astype(float)
    return labels


def mock_backtest_runner(strategy) -> list[dict[str, float]]:
    """Convert strategy output into a deterministic fold metric."""
    frame = pl.DataFrame(
        prepare_evaluation_context(generate_ohlcv(120, seed=12_345))
    )
    try:
        signal = strategy.predict(frame)
    except Exception as exc:
        return [
            {
                "metrics": {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 1.0,
                    "turnover": 0.0,
                    "regime_penalty": 0.0,
                    "complexity_penalty": 1.0,
                    "error_code": "mock_backtest_prediction_failure",
                    "error_detail": str(exc),
                },
            }
        ]
    values = np.asarray(signal.scores, dtype=float)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size == 0:
        sharpe = 0.0
    else:
        denom = np.std(finite, ddof=0)
        sharpe = float(np.mean(finite) / (denom + 1e-6)) if denom > 0 else 0.0
    sharpe = float(np.clip(sharpe, -2.0, 2.0))

    return [
        {
            "metrics": {
                "sharpe_ratio": sharpe,
                "total_return": sharpe * 0.08,
                "max_drawdown": 0.02 + (abs(float(np.min(finite))) / 50.0 if finite.size else 0.0),
                "turnover": float(np.mean(np.abs(np.diff(np.sign(finite)))) if finite.size > 1 else 0.0),
                "regime_penalty": 0.0,
                "complexity_penalty": 0.15,
            },
        }
    ]


def build_demo_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    register_series_sources(registry)
    register_numeric_ops(registry)
    register_comparison_ops(registry)
    register_logic_ops(registry)
    register_crossover_ops(registry)
    return registry


def run_two_stage() -> None:
    train_ohlcv = generate_ohlcv(300, seed=20260303)
    train_ctx = prepare_evaluation_context(train_ohlcv)
    train_ctx["labels"] = make_labels(train_ctx["close"])

    config = EvolutionConfig(
        population_size=24,
        max_depth=6,
        generations=2,
        seed=20260303,
        fitness_stages=FitnessStageConfig(
            label_metric="f1",
            label_top_k=0.25,
            use_backtest=True,
            backtest_top_n=3,
            backtest_metric="sharpe_ratio",
        ),
        run={"stage_b_candidate_budget": 6},
    )
    evaluator = wire_objectives(
        config.fitness_stages,
        backtest_fn=mock_backtest_runner,
        run_config=config.run,
    )

    registry = build_demo_registry()
    gp_config = build_gp_config(config)

    def on_generation(stats) -> None:
        best = stats.best_fitness[0]
        mean = stats.mean_fitness[0]
        print(
            f"generation={stats.generation:02d} "
            f"best_fitness={best:.4f} "
            f"mean={mean:.4f}"
        )

    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=evaluator,
        context=train_ctx,
        callback=on_generation,
    )

    best = result.fitness_history[-1]
    print(f"best fitness: {best.best_fitness!r}")
    print(f"best program size: {result.best_program.size}")

    holdout_ohlcv = generate_ohlcv(150, seed=20260304)
    holdout_ctx = prepare_evaluation_context(holdout_ohlcv)
    holdout_ctx["labels"] = make_labels(holdout_ctx["close"])
    holdout_scores = np.asarray(evaluate(result.best_program, holdout_ctx), dtype=np.float64)
    print(f"holdout score sample: {[float(v) for v in holdout_scores[:6]]}")


if __name__ == "__main__":
    run_two_stage()
