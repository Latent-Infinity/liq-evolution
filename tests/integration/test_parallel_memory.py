"""Stress-style tests for evaluator batch transport and payload stability."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution.adapters.parallel_eval import ParallelEvaluator


class _LargePayloadEvaluator:
    """Evaluator that emits repeated large metadata payloads."""

    def __init__(self, payload_size: int = 10_000) -> None:
        self.payload_size = payload_size

    def evaluate(
        self, programs: list[int], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del context
        base_payload = list(range(self.payload_size))
        return [
            {
                "program": int(program),
                "trace": base_payload,
                "metadata": {"stage": "stress", "payload_len": self.payload_size},
            }
            for program in programs
        ]


def test_parallel_batch_preserves_large_payload_shape() -> None:
    wrapper = ParallelEvaluator(
        evaluator=_LargePayloadEvaluator(), backend="sequential"
    )
    programs = list(range(64))
    results = wrapper.evaluate_batch(programs, {"labels": np.arange(64)})

    assert len(results) == len(programs)
    assert all(
        item["program"] == program
        for item, program in zip(results, programs, strict=False)
    )
    assert all(len(item["trace"]) == 10_000 for item in results)
    assert all(isinstance(item["metadata"], dict) for item in results)


def test_parallel_evaluate_keeps_context_schema_for_large_input() -> None:
    context = {"alpha": np.arange(128), "beta": np.zeros(128)}
    wrapper = ParallelEvaluator(
        evaluator=_LargePayloadEvaluator(), backend="sequential"
    )
    results = wrapper.evaluate(programs=list(range(16)), context=context)

    assert len(results) == 16
    assert all(result["metadata"]["payload_len"] == 10_000 for result in results)
    assert all(result["metadata"]["stage"] == "stress" for result in results)


def test_no_unbounded_memory_growth_over_repeated_evaluations() -> None:
    """Memory must not grow unboundedly when the same evaluator runs many iterations.

    Uses tracemalloc to measure peak memory across N iterations of a fixed-size
    batch.  The cache is bounded (max_entries=32) so old payloads are evicted.
    We allow a generous tolerance but reject clearly linear growth.
    """
    import tracemalloc

    from liq.evolution.fitness.eval_cache import FitnessEvaluationCache

    tracemalloc.start()

    cache = FitnessEvaluationCache(max_entries=32)
    evaluator = _LargePayloadEvaluator(payload_size=5_000)
    wrapper = ParallelEvaluator(evaluator=evaluator, backend="sequential")

    programs = list(range(48))
    context: dict[str, Any] = {"labels": np.arange(48)}
    n_iterations = 20

    # Warm up (let allocator reach steady state)
    for _ in range(3):
        wrapper.evaluate_batch(programs, context)

    _, baseline_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    for i in range(n_iterations):
        results = wrapper.evaluate_batch(programs, context)
        # Exercise the cache with unique keys so LRU eviction is triggered.
        for j, result in enumerate(results):
            cache.put(
                strategy_hash=f"prog_{j}",
                slice_id=f"iter_{i}",
                evaluator_fingerprint="fp",
                payload=result,
            )

    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Cache should be bounded.
    assert cache.entries <= 32

    # Peak memory after iterations should not be dramatically larger than
    # baseline.  We allow up to 3× to account for allocator fragmentation
    # and transient peaks, but reject truly unbounded growth (which would
    # scale linearly with n_iterations × payload_size).
    assert final_peak < baseline_peak * 3, (
        f"Suspected unbounded memory growth: baseline_peak={baseline_peak}, "
        f"final_peak={final_peak}"
    )
