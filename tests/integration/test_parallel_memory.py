"""Stress-style tests for evaluator batch transport and payload stability."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution.adapters.parallel_eval import ParallelEvaluator


class _LargePayloadEvaluator:
    """Evaluator that emits repeated large metadata payloads."""

    def __init__(self, payload_size: int = 10_000) -> None:
        self.payload_size = payload_size

    def evaluate(self, programs: list[int], context: dict[str, Any]) -> list[dict[str, Any]]:
        del context
        base_payload = list(range(self.payload_size))
        return [
            {
                "program": int(program),
                "trace": base_payload,
                "metadata": {"phase": "stress", "payload_len": self.payload_size},
            }
            for program in programs
        ]


def test_parallel_batch_preserves_large_payload_shape() -> None:
    wrapper = ParallelEvaluator(evaluator=_LargePayloadEvaluator(), backend="sequential")
    programs = list(range(64))
    results = wrapper.evaluate_batch(programs, {"labels": np.arange(64)})

    assert len(results) == len(programs)
    assert all(item["program"] == program for item, program in zip(results, programs))
    assert all(len(item["trace"]) == 10_000 for item in results)
    assert all(isinstance(item["metadata"], dict) for item in results)


def test_parallel_evaluate_keeps_context_schema_for_large_input() -> None:
    context = {"alpha": np.arange(128), "beta": np.zeros(128)}
    wrapper = ParallelEvaluator(evaluator=_LargePayloadEvaluator(), backend="sequential")
    results = wrapper.evaluate(programs=list(range(16)), context=context)

    assert len(results) == 16
    assert all(result["metadata"]["payload_len"] == 10_000 for result in results)
    assert all(result["metadata"]["phase"] == "stress" for result in results)
