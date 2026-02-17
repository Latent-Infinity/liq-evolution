"""Tests for ParallelEvaluator ray backend (Phase 4, Step 6).

All tests use mocked ray -- no real ray import is required.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from liq.evolution.adapters.parallel_eval import ParallelEvaluator
from liq.evolution.errors import AdapterError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyEvaluator:
    """Simple evaluator that returns programs doubled."""

    def evaluate(self, programs: list[Any], context: Any) -> list[Any]:
        return [p * 2 for p in programs]


class _FitnessEvaluatorCompat:
    """Evaluator with evaluate_fitness only."""

    def evaluate_fitness(self, programs: list[Any], context: Any) -> list[Any]:
        return [p + 1 for p in programs]


def _callable_evaluator(programs: list[Any], context: Any) -> list[Any]:
    return [p + 1 for p in programs]


class _FakeObjectRef:
    """Hashable stand-in for a ray ObjectRef."""

    _counter = 0

    def __init__(self, value: Any) -> None:
        _FakeObjectRef._counter += 1
        self._id = _FakeObjectRef._counter
        self.value = value

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeObjectRef) and self._id == other._id


def _make_mock_ray(evaluator_fn: Any = None) -> MagicMock:
    """Create a mock ray module.

    The mock wires up the ray.remote -> RemoteFunction -> .remote() -> ObjectRef
    pipeline so that ray.get(ref) returns the evaluated result.
    """
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    # ray.put returns a ref for shared context
    mock_ray.put.return_value = _FakeObjectRef("context_ref")

    # Track submitted chunks for inspection
    submitted_chunks: list[list[Any]] = []

    # ray.remote(fn) returns a RemoteFunction whose .remote() submits work
    remote_function = MagicMock()

    def fake_remote_submit(fn, chunk, ctx_ref, seed):
        submitted_chunks.append(chunk)
        result = fn(chunk, None)  # evaluate immediately
        return _FakeObjectRef(result)

    remote_function.remote.side_effect = fake_remote_submit
    mock_ray.remote.return_value = remote_function

    # ray.get resolves a FakeObjectRef to its value
    def fake_get(ref):
        if isinstance(ref, _FakeObjectRef):
            return ref.value
        return ref

    mock_ray.get.side_effect = fake_get

    # ray.wait returns first pending as done
    def fake_wait(pending, num_returns=1):
        done = pending[:num_returns]
        remaining = pending[num_returns:]
        return done, remaining

    mock_ray.wait.side_effect = fake_wait

    # Attach submitted_chunks for test inspection
    mock_ray._submitted_chunks = submitted_chunks

    return mock_ray


# ---------------------------------------------------------------------------
# Field defaults and validation
# ---------------------------------------------------------------------------


class TestParallelEvaluatorNewFields:
    """Verify new dataclass fields have correct defaults and validation."""

    def test_default_max_tasks_per_worker(self) -> None:
        pe = ParallelEvaluator()
        assert pe.max_tasks_per_worker == 100

    def test_default_memory_limit_mb(self) -> None:
        pe = ParallelEvaluator()
        assert pe.memory_limit_mb == 2048

    def test_default_memory_warn_threshold_mb(self) -> None:
        pe = ParallelEvaluator()
        assert pe.memory_warn_threshold_mb == 1536

    def test_default_auto_fallback(self) -> None:
        pe = ParallelEvaluator()
        assert pe.auto_fallback is True

    def test_default_seed(self) -> None:
        pe = ParallelEvaluator()
        assert pe.seed == 42

    def test_max_tasks_per_worker_zero_rejected(self) -> None:
        with pytest.raises(AdapterError, match="max_tasks_per_worker"):
            ParallelEvaluator(max_tasks_per_worker=0)

    def test_max_tasks_per_worker_negative_rejected(self) -> None:
        with pytest.raises(AdapterError, match="max_tasks_per_worker"):
            ParallelEvaluator(max_tasks_per_worker=-5)

    def test_memory_limit_mb_too_low_rejected(self) -> None:
        with pytest.raises(AdapterError, match="memory_limit_mb"):
            ParallelEvaluator(memory_limit_mb=64)

    def test_memory_warn_threshold_negative_rejected(self) -> None:
        with pytest.raises(AdapterError, match="memory_warn_threshold_mb"):
            ParallelEvaluator(memory_warn_threshold_mb=-1)

    def test_memory_warn_threshold_exceeds_limit_rejected(self) -> None:
        with pytest.raises(AdapterError, match="memory_warn_threshold_mb"):
            ParallelEvaluator(memory_limit_mb=512, memory_warn_threshold_mb=512)

    def test_memory_warn_threshold_gt_limit_rejected(self) -> None:
        with pytest.raises(AdapterError, match="memory_warn_threshold_mb"):
            ParallelEvaluator(memory_limit_mb=512, memory_warn_threshold_mb=1024)


# ---------------------------------------------------------------------------
# Worker seeding
# ---------------------------------------------------------------------------


class TestWorkerSeed:
    """_worker_seed must be deterministic and vary by index."""

    def test_deterministic(self) -> None:
        s1 = ParallelEvaluator._worker_seed(42, 0)
        s2 = ParallelEvaluator._worker_seed(42, 0)
        assert s1 == s2

    def test_varies_by_index(self) -> None:
        s0 = ParallelEvaluator._worker_seed(42, 0)
        s1 = ParallelEvaluator._worker_seed(42, 1)
        assert s0 != s1

    def test_varies_by_base_seed(self) -> None:
        s_a = ParallelEvaluator._worker_seed(42, 0)
        s_b = ParallelEvaluator._worker_seed(99, 0)
        assert s_a != s_b

    def test_returns_int(self) -> None:
        result = ParallelEvaluator._worker_seed(1, 2)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _make_eval_fn
# ---------------------------------------------------------------------------


class TestMakeEvalFn:
    """_make_eval_fn extracts a callable from various evaluator forms."""

    def test_extracts_evaluate_method(self) -> None:
        pe = ParallelEvaluator()
        ev = _DummyEvaluator()
        fn = pe._make_eval_fn(ev)
        assert fn([3], None) == [6]

    def test_passes_through_callable(self) -> None:
        pe = ParallelEvaluator()
        fn = pe._make_eval_fn(_callable_evaluator)
        assert fn([3], None) == [4]

    def test_raises_for_non_evaluator(self) -> None:
        pe = ParallelEvaluator()
        with pytest.raises(AdapterError, match="Evaluator must implement"):
            pe._make_eval_fn("not_an_evaluator")

    def test_uses_evaluate_fitness_fallback(self) -> None:
        pe = ParallelEvaluator()
        fn = pe._make_eval_fn(_FitnessEvaluatorCompat())
        assert fn([3], None) == [4]


# ---------------------------------------------------------------------------
# Ray backend integration (mocked ray)
# ---------------------------------------------------------------------------


class TestEvaluateWithRay:
    """Test the ray code path with a mocked ray module."""

    def test_fallback_on_import_error(self) -> None:
        """When ray is not importable, fall back to sequential."""
        pe = ParallelEvaluator(backend="ray", evaluator=_DummyEvaluator())
        # Setting module to None causes ImportError on import
        with patch.dict(sys.modules, {"ray": None}):
            result = pe.evaluate_batch([1, 2, 3], {})
        assert result == [2, 4, 6]

    def test_ray_put_called_for_context(self) -> None:
        """ray.put is called to share context in the object store."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=2,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            pe._evaluate_with_ray([1, 2, 3, 4], {"key": "val"}, _DummyEvaluator())

        mock_ray.put.assert_called_once_with({"key": "val"})

    def test_programs_chunked_by_max_workers(self) -> None:
        """Programs should be split into max_workers chunks."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=2,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            pe._evaluate_with_ray([1, 2, 3, 4], {}, _DummyEvaluator())

        assert len(mock_ray._submitted_chunks) == 2
        all_programs = [p for chunk in mock_ray._submitted_chunks for p in chunk]
        assert sorted(all_programs) == [1, 2, 3, 4]

    def test_results_maintain_input_order(self) -> None:
        """Flat results must match the original program order."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=2,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            result = pe._evaluate_with_ray([1, 2, 3, 4], {}, _DummyEvaluator())

        assert result == [2, 4, 6, 8]

    def test_remote_definition_has_max_calls(self) -> None:
        """max_calls should be wired from max_tasks_per_worker."""
        mock_ray = _make_mock_ray()
        max_calls = 17
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=2,
            max_tasks_per_worker=max_calls,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            pe._evaluate_with_ray([1, 2, 3, 4], {}, _DummyEvaluator())

        mock_ray.remote.assert_called_once_with(
            ParallelEvaluator._remote_eval_fn, max_calls=max_calls
        )

    def test_chunking_limits_chunk_count_to_worker_count(self) -> None:
        """ceil-based chunking should avoid creating excessive chunks."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=2,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            pe._evaluate_with_ray(list(range(7)), {}, _DummyEvaluator())

        # len=7 with max_workers=2 should produce 2 chunks via ceil division.
        assert len(mock_ray._submitted_chunks) == 2

    def test_ray_init_called_when_not_initialized(self) -> None:
        """ray.init is called if ray is not initialized."""
        mock_ray = _make_mock_ray()
        mock_ray.is_initialized.return_value = False

        pe = ParallelEvaluator(
            backend="ray", evaluator=_DummyEvaluator(), max_workers=1
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            pe._evaluate_with_ray([1], {}, _DummyEvaluator())

        mock_ray.init.assert_called_once_with(
            ignore_reinit_error=True, include_dashboard=False, log_to_driver=False
        )

    def test_max_in_flight_respected(self) -> None:
        """When max_in_flight is set, backpressure limiting should apply."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=4,
            max_in_flight=2,
        )

        programs = list(range(8))
        with patch.dict(sys.modules, {"ray": mock_ray}):
            result = pe._evaluate_with_ray(programs, {}, _DummyEvaluator())

        # All 4 chunks submitted (8 programs / 2 per chunk = 4 chunks)
        assert len(mock_ray._submitted_chunks) == 4
        # ray.wait was called for backpressure (chunks exceed max_in_flight)
        assert mock_ray.wait.call_count > 0
        # Results are correct
        assert result == [p * 2 for p in programs]

    def test_single_program_works(self) -> None:
        """Edge case: a single program should produce one chunk."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            max_workers=4,
        )

        with patch.dict(sys.modules, {"ray": mock_ray}):
            result = pe._evaluate_with_ray([5], {}, _DummyEvaluator())

        assert result == [10]
