"""Integration tests: evolve with trading primitives and label fitness."""

from __future__ import annotations

import numpy as np

from liq.evolution.config import EvolutionConfig, PrimitiveConfig, build_gp_config
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives.series_sources import prepare_evaluation_context
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.program.simplify import simplify
from liq.gp.types import EvolutionResult, FitnessResult


def _make_toy_data(n: int = 50) -> dict[str, np.ndarray]:
    """Create deterministic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    return {
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    }


def _make_context_with_labels(n: int = 50) -> dict[str, np.ndarray]:
    """Create evaluation context with synthetic labels."""
    ohlcv = _make_toy_data(n)
    ctx = prepare_evaluation_context(ohlcv)
    # Synthetic label: close > rolling mean of close (simplified)
    close = ctx["close"]
    mean_close = np.convolve(close, np.ones(5) / 5, mode="full")[:n]
    mean_close[:4] = close[:4]  # Fill initial bars
    labels = np.where(close > mean_close, 1.0, 0.0)
    ctx["labels"] = labels
    return ctx


def _small_gp_config(*, generations: int = 3, seed: int = 42) -> LiqGPConfig:
    """Small config for fast integration tests."""
    return LiqGPConfig(
        population_size=20,
        max_depth=4,
        generations=generations,
        seed=seed,
        tournament_size=3,
        elitism_count=2,
        constant_opt_enabled=False,
        semantic_dedup_enabled=False,
        simplification_enabled=False,
    )


class TestEvolveWithTradingPrimitives:
    def test_evolve_completes(self) -> None:
        """Evolution completes without error using trading registry."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        gp_config = _small_gp_config()

        result = evolve(registry, gp_config, evaluator, ctx)

        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) > 0

    def test_evolve_produces_valid_program(self) -> None:
        """Best program evaluates to valid float64 array of correct length."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels(n=30)
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        gp_config = _small_gp_config()

        result = evolve(registry, gp_config, evaluator, ctx)

        scores = evaluate(result.best_program, ctx)
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float64
        assert len(scores) == 30

    def test_evolve_reproducible(self) -> None:
        """Same seed produces identical results."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        r1 = evolve(registry, _small_gp_config(seed=42), evaluator, ctx)
        r2 = evolve(registry, _small_gp_config(seed=42), evaluator, ctx)

        scores1 = evaluate(r1.best_program, ctx)
        scores2 = evaluate(r2.best_program, ctx)
        np.testing.assert_array_equal(scores1, scores2)

        assert len(r1.fitness_history) == len(r2.fitness_history)
        for s1, s2 in zip(r1.fitness_history, r2.fitness_history, strict=True):
            assert s1.best_fitness == s2.best_fitness

    def test_evolve_with_label_evaluator(self) -> None:
        """LabelFitnessEvaluator produces FitnessResult objects."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="accuracy", top_k=0.3)
        gp_config = _small_gp_config(generations=2)

        result = evolve(registry, gp_config, evaluator, ctx)

        assert result.fitness_history[0].best_fitness is not None
        # Best fitness should be a tuple with one element
        assert len(result.fitness_history[0].best_fitness) == 1


class TestSerializeRoundtrip:
    def test_serialize_evolved_program(self) -> None:
        """Serialize → deserialize → evaluate produces identical output."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        gp_config = _small_gp_config()

        result = evolve(registry, gp_config, evaluator, ctx)
        program = result.best_program

        # Serialize and deserialize
        data = serialize(program)
        restored = deserialize(data, registry)

        # Evaluate both
        original_scores = evaluate(program, ctx)
        restored_scores = evaluate(restored, ctx)
        np.testing.assert_array_equal(original_scores, restored_scores)


class TestSimplifyEvolved:
    def test_simplify_evolved_program(self) -> None:
        """Simplification reduces or preserves size, doesn't crash."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)
        gp_config = _small_gp_config(generations=5)

        result = evolve(registry, gp_config, evaluator, ctx)
        program = result.best_program
        simplified = simplify(program)

        assert simplified.size <= program.size


class TestNSGA2Integration:
    def test_evolve_with_nsga2(self) -> None:
        """NSGA-II with 2 objectives produces a non-empty Pareto front."""
        from liq.gp.config import FitnessConfig as LiqFitnessConfig

        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()

        # Multi-objective evaluator: fitness + complexity penalty
        class MultiObjectiveEvaluator:
            def evaluate(
                self,
                programs: list,
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                labels = context["labels"]
                results = []
                for p in programs:
                    scores = evaluate(p, context)
                    valid = np.isfinite(scores)
                    if not np.any(valid):
                        results.append(FitnessResult(objectives=(0.0, float(p.size))))
                        continue
                    clean = np.where(valid, scores, -np.inf)
                    k = max(1, int(len(clean) * 0.5))
                    top_idx = np.argsort(clean)[-k:]
                    precision = float(np.mean(labels[top_idx]))
                    results.append(FitnessResult(objectives=(precision, float(p.size))))
                return results

        gp_config = LiqGPConfig(
            population_size=20,
            max_depth=4,
            generations=3,
            seed=42,
            tournament_size=3,
            elitism_count=2,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
            simplification_enabled=False,
            fitness=LiqFitnessConfig(
                objectives=["fitness", "complexity"],
                objective_directions=["maximize", "minimize"],
            ),
        )

        result = evolve(registry, gp_config, MultiObjectiveEvaluator(), ctx)

        assert isinstance(result, EvolutionResult)
        assert len(result.pareto_front) > 0


class TestConfigBridgeIntegration:
    def test_build_gp_config_used_with_evolve(self) -> None:
        """Config bridge produces a config usable by evolve()."""
        evo_config = EvolutionConfig(
            population_size=20,
            max_depth=4,
            generations=2,
            seed=42,
        )
        gp_config = build_gp_config(evo_config)

        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        ctx = _make_context_with_labels()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        result = evolve(registry, gp_config, evaluator, ctx)
        assert isinstance(result, EvolutionResult)
