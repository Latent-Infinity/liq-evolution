"""Tests for ParallelEvaluator adapter."""

from __future__ import annotations

from typing import Any

import pytest

from liq.evolution.adapters.parallel_eval import ParallelEvaluator
from liq.evolution.errors import AdapterError


class _DummyEvaluator:
    def evaluate(self, programs: list[int], context: dict[str, Any]) -> list[str]:
        return [f"{len(programs)}:{context['tag']}:{program}" for program in programs]


class _CompatEvaluator:
    """Evaluator that only implements the compatibility method."""

    def evaluate_fitness(self, programs: list[int], context: dict[str, Any]) -> list[str]:
        return [f"{len(programs)}:{context['tag']}:{program + 1}" for program in programs]


def test_evaluate_batch_uses_default_evaluator() -> None:
    evaluator = _DummyEvaluator()
    wrapper = ParallelEvaluator(evaluator=evaluator, backend="sequential")
    out = wrapper.evaluate_batch([1, 2, 3], {"tag": "x"})
    assert out == ["3:x:1", "3:x:2", "3:x:3"]


def test_evaluate_batch_falls_back_to_evaluate_fitness() -> None:
    evaluator = _CompatEvaluator()
    wrapper = ParallelEvaluator(evaluator=evaluator, backend="sequential")
    out = wrapper.evaluate_batch([1, 2], {"tag": "y"})
    assert out == ["2:y:2", "2:y:3"]


def test_evaluate_batch_uses_call_override() -> None:
    wrapper = ParallelEvaluator(backend="sequential")
    out = wrapper.evaluate_batch([1, 2], {"tag": "y"}, evaluator=lambda p, c: [p[0]])
    assert out == [1]


def test_evaluate_batch_empty_programs() -> None:
    wrapper = ParallelEvaluator()
    out = wrapper.evaluate_batch([], {"tag": "empty"}, evaluator=lambda p, c: [1, 2, 3])
    assert out == []


def test_invalid_backend_raises() -> None:
    with pytest.raises(AdapterError, match="backend must be"):
        ParallelEvaluator(backend="unknown")


def test_missing_evaluator_raises() -> None:
    wrapper = ParallelEvaluator()
    with pytest.raises(AdapterError, match="requires an evaluator"):
        wrapper.evaluate_batch([1], {"tag": "x"})


def test_evaluate_aliases_batch() -> None:
    evaluator = _DummyEvaluator()
    wrapper = ParallelEvaluator(evaluator=evaluator, backend="sequential")
    out = wrapper.evaluate([1, 2, 3], {"tag": "x"})
    assert out == ["3:x:1", "3:x:2", "3:x:3"]


def test_evaluate_alias_uses_override() -> None:
    wrapper = ParallelEvaluator(backend="sequential")
    out = wrapper.evaluate([1, 2], {"tag": "y"}, evaluator=lambda p, c: [p[0]])
    assert out == [1]
