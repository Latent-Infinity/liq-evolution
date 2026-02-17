"""Tests for ParallelEvaluator memory pressure checks (Phase 4, Step 6).

Uses mocked resource.getrusage to control RSS values without requiring
actual memory allocation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from liq.evolution.adapters.parallel_eval import ParallelEvaluator
from liq.evolution.errors import ParallelExecutionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyEvaluator:
    def evaluate(self, programs: list[Any], context: Any) -> list[Any]:
        return [p * 2 for p in programs]


def _make_mock_ray() -> MagicMock:
    """Minimal mock ray module so _evaluate_with_ray gets past the import."""
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    return mock_ray


def _mock_getrusage_with_rss_bytes(rss_bytes: int) -> MagicMock:
    """Return a mock for resource.getrusage that reports the given RSS.

    On macOS ru_maxrss is in bytes; on Linux it is in KB.  Tests patch
    platform.system() to "Darwin" so the code divides by 1024*1024.
    """
    usage = MagicMock()
    usage.ru_maxrss = rss_bytes
    return usage


# ---------------------------------------------------------------------------
# _check_memory_pressure
# ---------------------------------------------------------------------------


class TestCheckMemoryPressure:
    """Memory pressure detection via _check_memory_pressure."""

    def test_returns_false_under_limit(self) -> None:
        """RSS well below limit -> no pressure."""
        pe = ParallelEvaluator(memory_limit_mb=2048, memory_warn_threshold_mb=1536)
        # 512 MB in bytes
        usage = _mock_getrusage_with_rss_bytes(512 * 1024 * 1024)
        with (
            patch("liq.evolution.adapters.parallel_eval.resource") as mock_resource,
            patch("liq.evolution.adapters.parallel_eval.platform") as mock_platform,
        ):
            mock_resource.getrusage.return_value = usage
            mock_resource.RUSAGE_SELF = 0
            mock_platform.system.return_value = "Darwin"
            assert pe._check_memory_pressure() is False

    def test_returns_true_over_limit(self) -> None:
        """RSS above limit -> pressure detected."""
        pe = ParallelEvaluator(memory_limit_mb=2048, memory_warn_threshold_mb=1536)
        # 2100 MB in bytes
        usage = _mock_getrusage_with_rss_bytes(2100 * 1024 * 1024)
        with (
            patch("liq.evolution.adapters.parallel_eval.resource") as mock_resource,
            patch("liq.evolution.adapters.parallel_eval.platform") as mock_platform,
        ):
            mock_resource.getrusage.return_value = usage
            mock_resource.RUSAGE_SELF = 0
            mock_platform.system.return_value = "Darwin"
            assert pe._check_memory_pressure() is True

    def test_warning_logged_at_warn_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning is logged when RSS >= warn threshold but < limit."""
        pe = ParallelEvaluator(memory_limit_mb=2048, memory_warn_threshold_mb=1536)
        # 1600 MB in bytes  (>= 1536 warn, < 2048 limit)
        usage = _mock_getrusage_with_rss_bytes(1600 * 1024 * 1024)
        with (
            patch("liq.evolution.adapters.parallel_eval.resource") as mock_resource,
            patch("liq.evolution.adapters.parallel_eval.platform") as mock_platform,
            caplog.at_level(
                logging.WARNING, logger="liq.evolution.adapters.parallel_eval"
            ),
        ):
            mock_resource.getrusage.return_value = usage
            mock_resource.RUSAGE_SELF = 0
            mock_platform.system.return_value = "Darwin"
            result = pe._check_memory_pressure()

        assert result is False  # under limit
        assert "warn threshold" in caplog.text.lower() or "RSS" in caplog.text

    def test_linux_rss_conversion(self) -> None:
        """On Linux, ru_maxrss is in KB, not bytes."""
        pe = ParallelEvaluator(memory_limit_mb=2048, memory_warn_threshold_mb=1536)
        # 512 MB in KB
        usage = _mock_getrusage_with_rss_bytes(512 * 1024)
        with (
            patch("liq.evolution.adapters.parallel_eval.resource") as mock_resource,
            patch("liq.evolution.adapters.parallel_eval.platform") as mock_platform,
        ):
            mock_resource.getrusage.return_value = usage
            mock_resource.RUSAGE_SELF = 0
            mock_platform.system.return_value = "Linux"
            assert pe._check_memory_pressure() is False

    def test_configurable_thresholds(self) -> None:
        """Custom thresholds are respected."""
        pe = ParallelEvaluator(memory_limit_mb=256, memory_warn_threshold_mb=200)
        # 260 MB in bytes (over 256 limit)
        usage = _mock_getrusage_with_rss_bytes(260 * 1024 * 1024)
        with (
            patch("liq.evolution.adapters.parallel_eval.resource") as mock_resource,
            patch("liq.evolution.adapters.parallel_eval.platform") as mock_platform,
        ):
            mock_resource.getrusage.return_value = usage
            mock_resource.RUSAGE_SELF = 0
            mock_platform.system.return_value = "Darwin"
            assert pe._check_memory_pressure() is True


# ---------------------------------------------------------------------------
# Auto-fallback integration
# ---------------------------------------------------------------------------


class TestAutoFallback:
    """Memory pressure triggers sequential fallback or error."""

    def test_auto_fallback_true_falls_back(self) -> None:
        """auto_fallback=True -> sequential evaluation on memory pressure."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            auto_fallback=True,
            memory_limit_mb=256,
            memory_warn_threshold_mb=200,
        )
        with (
            patch.dict(sys.modules, {"ray": mock_ray}),
            patch.object(pe, "_check_memory_pressure", return_value=True),
        ):
            # Should fall back to sequential, not raise
            result = pe.evaluate_batch([1, 2, 3], {})
        assert result == [2, 4, 6]

    def test_auto_fallback_false_raises(self) -> None:
        """auto_fallback=False -> ParallelExecutionError on memory pressure."""
        mock_ray = _make_mock_ray()
        pe = ParallelEvaluator(
            backend="ray",
            evaluator=_DummyEvaluator(),
            auto_fallback=False,
            memory_limit_mb=256,
            memory_warn_threshold_mb=200,
        )
        with (
            patch.dict(sys.modules, {"ray": mock_ray}),
            patch.object(pe, "_check_memory_pressure", return_value=True),
            pytest.raises(ParallelExecutionError, match="[Mm]emory pressure"),
        ):
            pe.evaluate_batch([1, 2, 3], {})
