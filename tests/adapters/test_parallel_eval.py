"""Tests for ParallelEvaluator adapter."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from liq.evolution.adapters.parallel_eval import ParallelEvaluator
from liq.evolution.errors import AdapterError, ParallelExecutionError


class _DummyEvaluator:
    def evaluate(self, programs: list[int], context: dict[str, Any]) -> list[str]:
        return [f"{len(programs)}:{context['tag']}:{program}" for program in programs]


class _CompatEvaluator:
    """Evaluator that only implements the compatibility method."""

    def evaluate_fitness(
        self, programs: list[int], context: dict[str, Any]
    ) -> list[str]:
        return [
            f"{len(programs)}:{context['tag']}:{program + 1}" for program in programs
        ]


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


def test_post_init_rejects_invalid_max_workers() -> None:
    with pytest.raises(AdapterError, match="max_workers must be >= 1"):
        ParallelEvaluator(max_workers=0)


def test_post_init_rejects_invalid_max_in_flight() -> None:
    with pytest.raises(AdapterError, match="max_in_flight must be >= 1"):
        ParallelEvaluator(max_in_flight=0)


def test_evaluate_batch_wraps_unexpected_exceptions() -> None:
    wrapper = ParallelEvaluator(
        evaluator=lambda programs, context: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(ParallelExecutionError, match="Batch evaluation failed"):
        wrapper.evaluate_batch([1], {"tag": "x"})


def test_evaluate_batch_prefers_ray_backend_fallback_when_missing() -> None:
    original_import = builtins.__import__

    def _raise_for_ray(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "ray":
            raise ImportError("ray not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(builtins, "__import__", _raise_for_ray)

    try:
        wrapper = ParallelEvaluator(
            evaluator=lambda programs, context: [
                f"{len(programs)}:{context}" for _ in programs
            ],
            backend="ray",
        )
        assert wrapper.evaluate_batch([1, 2, 3], "abc") == ["3:abc", "3:abc", "3:abc"]
    finally:
        monkeypatch.undo()


def test_make_eval_fn_requires_known_protocol() -> None:
    with pytest.raises(AdapterError, match="Evaluator must implement evaluate"):
        ParallelEvaluator()._make_eval_fn(object())


def test_call_evaluator_rejects_unknown_object() -> None:
    with pytest.raises(AdapterError, match="Evaluator must implement evaluate"):
        ParallelEvaluator._call_evaluator({}, [1], {})


def test_remote_eval_fn_sets_worker_seed_state() -> None:
    def _eval_fn(programs: list[int], _context: dict[str, Any]) -> list[float]:
        import random

        import numpy as np

        return [float(random.randint(0, 9999)), float(np.random.random())]

    run_one = ParallelEvaluator._remote_eval_fn(
        _eval_fn, [1, 2], {"seed": "seed"}, 12345
    )
    run_two = ParallelEvaluator._remote_eval_fn(
        _eval_fn, [1, 2], {"seed": "seed"}, 12345
    )
    assert run_one == run_two
