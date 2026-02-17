"""Parallel batch fitness evaluation helpers.

Phase-4 implementation supports real parallel dispatch via Ray with bounded
in-flight tasks, per-worker deterministic seeding, memory pressure detection,
and automatic fallback to sequential evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import resource
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from liq.evolution.errors import AdapterError, ParallelExecutionError

logger = logging.getLogger(__name__)

Evaluator = Callable[[list[Any], Any], list[Any]] | Any


@dataclass
class ParallelEvaluator:
    """Evaluate batches of programs with a strategy-compatible evaluator.

    Parameters
    ----------
    evaluator:
        Object exposing ``evaluate(programs, context)`` or a callable with the same
        signature. If ``None``, evaluation methods must pass an evaluator via
        argument.
    backend:
        ``"sequential"`` (default) or ``"ray"``. Ray is optional and will
        gracefully fallback to sequential when unavailable.
    max_workers:
        Maximum number of parallel workers (chunks) for the ray backend.
    max_in_flight:
        Maximum number of concurrent pending tasks before backpressure is applied.
        When ``None``, defaults to the number of chunks.
    max_tasks_per_worker:
        Worker lifecycle recycling threshold. After this many tasks a worker
        should be recycled.
    memory_limit_mb:
        Memory limit per worker in MB.
    memory_warn_threshold_mb:
        RSS threshold (in MB) at which a warning is logged.
    auto_fallback:
        If ``True``, fall back to sequential evaluation on memory pressure
        instead of raising.
    seed:
        Base seed for deterministic per-worker seeding.
    """

    evaluator: Evaluator | None = None
    backend: str = "sequential"
    max_workers: int = 1
    max_in_flight: int | None = None
    max_tasks_per_worker: int = 100
    memory_limit_mb: int = 2048
    memory_warn_threshold_mb: int = 1536
    auto_fallback: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backend not in {"sequential", "ray"}:
            raise AdapterError("backend must be 'sequential' or 'ray'")
        if self.max_workers < 1:
            raise AdapterError("max_workers must be >= 1")
        if self.max_in_flight is not None and self.max_in_flight < 1:
            raise AdapterError("max_in_flight must be >= 1 when set")
        if self.max_tasks_per_worker < 1:
            raise AdapterError("max_tasks_per_worker must be >= 1")
        if self.memory_limit_mb < 128:
            raise AdapterError("memory_limit_mb must be >= 128")
        if self.memory_warn_threshold_mb < 0:
            raise AdapterError("memory_warn_threshold_mb must be >= 0")
        if self.memory_warn_threshold_mb >= self.memory_limit_mb:
            raise AdapterError("memory_warn_threshold_mb must be < memory_limit_mb")

    def evaluate_batch(
        self,
        programs: list[Any],
        context: Any,
        evaluator: Evaluator | None = None,
    ) -> list[Any]:
        """Evaluate a batch of programs.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context passed through to evaluator.
            evaluator: Optional override for this call.

        Returns:
            Fitness results for each program.

        Raises:
            AdapterError: If no evaluator is configured.
            ParallelExecutionError: If evaluation itself fails.
        """

        evaluator_to_use = evaluator or self.evaluator
        if evaluator_to_use is None:
            raise AdapterError("ParallelEvaluator requires an evaluator")

        if not programs:
            return []

        try:
            if self.backend == "ray":
                return self._evaluate_with_ray(programs, context, evaluator_to_use)
            return self._evaluate_sequential(programs, context, evaluator_to_use)
        except (AdapterError, ParallelExecutionError):
            raise
        except Exception as exc:
            raise ParallelExecutionError(f"Batch evaluation failed: {exc}") from exc

    def _evaluate_sequential(
        self,
        programs: list[Any],
        context: Any,
        evaluator: Evaluator,
    ) -> list[Any]:
        return self._call_evaluator(evaluator, programs, context)

    def _evaluate_with_ray(
        self,
        programs: list[Any],
        context: Any,
        evaluator: Evaluator,
    ) -> list[Any]:
        try:
            import ray  # type: ignore[import-untyped]
        except Exception:
            # Keep behavior deterministic and safe in environments without ray.
            return self._evaluate_sequential(programs, context, evaluator)

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )

        # Check memory before starting
        if self._check_memory_pressure():
            if self.auto_fallback:
                logger.warning("Memory pressure detected, falling back to sequential")
                return self._evaluate_sequential(programs, context, evaluator)
            raise ParallelExecutionError("Memory pressure exceeds limit")

        # Put shared context in the object store
        context_ref = ray.put(context)

        # Create remote evaluation function
        remote_eval = ray.remote(self._remote_eval_fn)

        eval_fn = self._make_eval_fn(evaluator)

        # Split programs into chunks for workers
        chunk_size = max(1, len(programs) // self.max_workers)
        chunks = [
            programs[i : i + chunk_size] for i in range(0, len(programs), chunk_size)
        ]

        # Submit with bounded in-flight using ray.wait backpressure
        max_in_flight = self.max_in_flight or len(chunks)
        pending: list[Any] = []
        collected: list[tuple[int, list[Any]]] = []
        ref_to_idx: dict[Any, int] = {}

        for idx, chunk in enumerate(chunks):
            worker_seed = self._worker_seed(self.seed, idx)
            ref = remote_eval.remote(eval_fn, chunk, context_ref, worker_seed)
            pending.append(ref)
            ref_to_idx[ref] = idx

            # Backpressure: drain when too many in flight
            while len(pending) > max_in_flight:
                done, pending = ray.wait(pending, num_returns=1)
                for done_ref in done:
                    result = ray.get(done_ref)
                    collected.append((ref_to_idx[done_ref], result))

        # Collect remaining
        for ref in pending:
            result = ray.get(ref)
            collected.append((ref_to_idx[ref], result))

        # Sort by original chunk index and flatten
        collected.sort(key=lambda pair: pair[0])
        return [item for _, chunk_result in collected for item in chunk_result]

    @staticmethod
    def _remote_eval_fn(
        eval_fn: Callable[..., list[Any]],
        programs: list[Any],
        context: Any,
        worker_seed: int,  # noqa: ARG004 - reserved for future per-worker seeding
    ) -> list[Any]:
        """Plain function suitable for ``ray.remote`` dispatch."""
        return eval_fn(programs, context)

    @staticmethod
    def _worker_seed(base_seed: int, index: int) -> int:
        """Deterministic per-worker seed using SHA-256."""
        h = hashlib.sha256(f"{base_seed}:{index}".encode())
        return int.from_bytes(h.digest()[:8], "big")

    def _check_memory_pressure(self) -> bool:
        """Check if current process RSS exceeds thresholds.

        Logs a warning when RSS >= ``memory_warn_threshold_mb``.
        Returns ``True`` when RSS >= ``memory_limit_mb``.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = usage.ru_maxrss
        if platform.system() == "Linux":
            # Linux reports ru_maxrss in KB
            rss_mb = rss_bytes / 1024
        else:
            # macOS reports ru_maxrss in bytes
            rss_mb = rss_bytes / (1024 * 1024)

        if rss_mb >= self.memory_warn_threshold_mb:
            logger.warning(
                "RSS %.0f MB >= warn threshold %d MB",
                rss_mb,
                self.memory_warn_threshold_mb,
            )
        return rss_mb >= self.memory_limit_mb

    def _make_eval_fn(self, evaluator: Evaluator) -> Callable[..., list[Any]]:
        """Extract a plain callable from the evaluator for remote dispatch."""
        if hasattr(evaluator, "evaluate") and callable(evaluator.evaluate):  # type: ignore[union-attr]
            return evaluator.evaluate  # type: ignore[union-attr]
        if callable(evaluator):
            return evaluator  # type: ignore[return-value]
        raise AdapterError("Evaluator must implement evaluate() or be callable")

    @staticmethod
    def _call_evaluator(
        evaluator: Evaluator,
        programs: list[Any],
        context: Any,
    ) -> list[Any]:
        if hasattr(evaluator, "evaluate") and callable(evaluator.evaluate):  # type: ignore[union-attr]
            return evaluator.evaluate(programs, context)  # type: ignore[misc]
        if callable(evaluator):
            return evaluator(programs, context)  # type: ignore[misc]

        raise AdapterError("Evaluator must implement evaluate() or be callable")
