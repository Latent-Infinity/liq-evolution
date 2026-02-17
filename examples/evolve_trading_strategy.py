#!/usr/bin/env python3
"""Evolve a trading signal strategy with liq-evolution.

This example demonstrates the full workflow:
  1. Build a trading primitive registry
  2. Generate synthetic OHLCV data with labels
  3. Configure the evolution pipeline
  4. Evolve using label-based fitness (F1 score)
  5. Inspect per-generation stats via a callback
  6. Serialize the winning genome and deserialize it back
  7. Evaluate the best program on held-out data

Run:
    cd liq-evolution
    uv run python examples/evolve_trading_strategy.py
"""

from __future__ import annotations

import json

import numpy as np

from liq.evolution import (
    EvolutionConfig,
    Genome,
    LabelFitnessEvaluator,
    build_gp_config,
    build_trading_registry,
    deserialize_genome,
    evaluate,
    evolve,
    prepare_evaluation_context,
    serialize_genome,
)
from liq.evolution.config import PrimitiveConfig

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------


def generate_ohlcv(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data as a geometric random walk."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    spread = rng.uniform(0.002, 0.006, n)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(1e6, 5e7, n)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def make_labels(close: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Binary labels: 1.0 where close[i+horizon] > close[i]."""
    labels = np.zeros(len(close))
    labels[:-horizon] = (close[horizon:] > close[:-horizon]).astype(float)
    return labels


# ---------------------------------------------------------------------------
# 2. Main evolution workflow
# ---------------------------------------------------------------------------


def main() -> None:
    # -- Registry --
    # Default config enables numeric, comparison, logic, crossover, temporal,
    # and series-source primitives.  liq-ta indicators are off (no extra deps).
    registry = build_trading_registry(PrimitiveConfig())
    print(f"Registered {len(registry.list_primitives())} primitives")

    # -- Data --
    train_ohlcv = generate_ohlcv(n=800, seed=42)
    test_ohlcv = generate_ohlcv(n=200, seed=99)

    train_ctx = prepare_evaluation_context(train_ohlcv)
    train_ctx["labels"] = make_labels(train_ohlcv["close"])

    test_ctx = prepare_evaluation_context(test_ohlcv)
    test_ctx["labels"] = make_labels(test_ohlcv["close"])

    # -- Config --
    evo_config = EvolutionConfig(
        population_size=200,
        generations=30,
        max_depth=6,
        seed=42,
    )
    gp_config = build_gp_config(evo_config)

    # -- Fitness evaluator --
    evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.1)

    # -- Generation callback --
    def on_generation(stats):
        if stats.generation % 5 == 0 or stats.generation == gp_config.generations - 1:
            print(
                f"  gen {stats.generation:3d}  "
                f"best={stats.best_fitness[0]:.4f}  "
                f"mean={stats.mean_fitness[0]:.4f}  "
                f"size={stats.best_program_size:3d}  "
                f"unique={stats.unique_semantics_ratio:.2%}"
            )

    # -- Evolve --
    print(
        f"\nEvolving {evo_config.population_size} programs "
        f"for {evo_config.generations} generations..."
    )
    result = evolve(
        registry=registry,
        config=gp_config,
        evaluator=evaluator,
        context=train_ctx,
        callback=on_generation,
    )

    # -- Results --
    best_fitness = result.fitness_history[-1].best_fitness[0]
    print(f"\nBest train fitness (F1): {best_fitness:.4f}")
    print(f"Best program size: {result.best_program.size} nodes")
    print(f"Best program: {result.best_program}")

    # -- Out-of-sample evaluation --
    scores = evaluate(result.best_program, test_ctx)
    test_labels = test_ctx["labels"]
    n = len(test_labels)
    k = max(1, int(n * 0.1))
    top_k_idx = np.argsort(scores)[-k:]
    precision = float(np.mean(test_labels[top_k_idx]))
    print(f"\nOut-of-sample precision@10%: {precision:.4f}")

    # -- Serialization round-trip --
    genome = Genome(entry_program=result.best_program)
    payload = serialize_genome(genome)
    print(f"\nSerialized genome ({len(json.dumps(payload))} bytes JSON)")

    restored = deserialize_genome(payload, registry)
    assert restored.entry_program == genome.entry_program
    print("Deserialization round-trip: OK")

    # -- Pareto front --
    print(f"\nPareto front: {len(result.pareto_front)} programs")

    # -- Fitness history summary --
    first = result.fitness_history[0]
    last = result.fitness_history[-1]
    print(
        f"Fitness improved from {first.best_fitness[0]:.4f} "
        f"to {last.best_fitness[0]:.4f} "
        f"over {len(result.fitness_history)} generations"
    )


if __name__ == "__main__":
    main()
