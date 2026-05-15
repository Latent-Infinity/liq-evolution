"""Indicator backend throughput benchmarks (deterministic, CI-safe)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from liq.evolution.primitives.indicators_liq_ta import (
    LiqFeaturesBackend,
    LiqTAIndicatorBackend,
)


def _artifact_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "benchmarks"
        / "indicator_throughput_baseline.json"
    )


def _load_artifact() -> dict[str, Any]:
    return json.loads(_artifact_path().read_text(encoding="utf-8"))


def _build_benchmark_data(seed: int, size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(size))
    high = close + rng.uniform(0.0, 2.0, size)
    low = close - rng.uniform(0.0, 2.0, size)
    open_ = close - rng.uniform(0.0, 1.0, size)
    volume = rng.uniform(1_000.0, 5_000.0, size)
    return {
        "data": close,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class LegacyBenchmarkBackend:
    """Legacy benchmark fixture emulating pre-migration bridge overhead."""

    def __init__(self, normalization_passes: int) -> None:
        self._legacy = LiqTAIndicatorBackend()
        self._normalization_passes = normalization_passes

    def _bridge_payload(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        payload = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in data.items()
            if key in {"data", "open", "high", "low", "close", "volume"}
        }
        payload["ts"] = np.arange(len(payload["close"]), dtype=np.int64)
        df = pl.DataFrame(payload)
        for col in ("data", "open", "high", "low", "close", "volume"):
            arr = df[col].to_numpy()
            for _ in range(self._normalization_passes):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, -1e9, 1e9)
            payload[col] = arr
        return payload

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        payload = self._bridge_payload(data)
        if name in {"rsi", "atr"}:
            return self._legacy.compute(name, params, payload, **kwargs)
        if name == "natr":
            period = int(params.get("period", 14))
            atr = self._legacy.compute("atr", {"period": period}, payload, **kwargs)
            out = np.full_like(payload["close"], np.nan, dtype=np.float64)
            mask = np.isfinite(atr) & (payload["close"] != 0.0)
            out[mask] = (atr[mask] / payload["close"][mask]) * 100.0
            return out
        if name == "abnormal_turnover":
            period = int(params.get("period", 20))
            volume = np.asarray(payload["volume"], dtype=np.float64)
            mean = np.full_like(volume, np.nan, dtype=np.float64)
            if period > 0 and volume.size >= period:
                csum = np.cumsum(volume, dtype=np.float64)
                mean[period - 1 :] = (
                    csum[period - 1 :] - np.concatenate(([0.0], csum[:-period]))
                ) / period
            ratio = np.full_like(volume, np.nan, dtype=np.float64)
            mask = np.isfinite(mean) & (mean != 0.0)
            ratio[mask] = volume[mask] / mean[mask]
            return np.where(np.isfinite(ratio), ratio, np.nan)
        raise ValueError(f"unsupported benchmark indicator: {name}")


def _time_per_call_cached(
    backend: Any,
    indicators: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    *,
    warmup_calls: int,
    repetitions: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in indicators:
        name = str(spec["name"])
        params = dict(spec["params"])
        for _ in range(warmup_calls):
            backend.compute(name, params, data)
        started = time.perf_counter()
        for _ in range(repetitions):
            backend.compute(name, params, data)
        elapsed = time.perf_counter() - started
        out[name] = elapsed / repetitions
    return out


def _time_per_call_uncached(
    backend_factory: Any,
    indicators: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    *,
    repetitions: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in indicators:
        name = str(spec["name"])
        params = dict(spec["params"])
        started = time.perf_counter()
        for _ in range(repetitions):
            backend = backend_factory()
            backend.compute(name, params, data)
        elapsed = time.perf_counter() - started
        out[name] = elapsed / repetitions
    return out


def _p90_slowdown_x(
    *,
    baseline_time: dict[str, float],
    candidate_time: dict[str, float],
) -> float:
    common = sorted(set(baseline_time).intersection(candidate_time))
    assert common, "benchmark must have at least one common indicator"
    ratios = np.asarray(
        [candidate_time[name] / baseline_time[name] for name in common],
        dtype=np.float64,
    )
    return float(np.quantile(ratios, 0.9))


class TestIndicatorThroughputBenchmarks:
    def test_artifact_is_versioned_and_reproducible(self) -> None:
        artifact = _load_artifact()
        assert artifact["artifact_version"] == "indicator_backend_benchmark_v1"
        assert artifact["seed"] == 20260303
        assert artifact["input_size"] == 2048
        assert Path(_artifact_path()).exists()

        a = _build_benchmark_data(artifact["seed"], artifact["input_size"])
        b = _build_benchmark_data(artifact["seed"], artifact["input_size"])
        for key in ("data", "open", "high", "low", "close", "volume"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_representative_indicators_supported_by_benchmark_fixture(self) -> None:
        artifact = _load_artifact()
        data = _build_benchmark_data(artifact["seed"], artifact["input_size"])
        indicators = list(artifact["indicators"])
        backend = LegacyBenchmarkBackend(
            normalization_passes=int(artifact["legacy_fixture"]["normalization_passes"])
        )
        for spec in indicators:
            output = backend.compute(spec["name"], spec["params"], data)
            assert output.shape == data["close"].shape
            assert output.dtype == np.float64

    def test_cached_call_regression_cap(self) -> None:
        artifact = _load_artifact()
        data = _build_benchmark_data(artifact["seed"], artifact["input_size"])
        indicators = list(artifact["indicators"])
        workload = artifact["cached_workload"]

        baseline_backend = LegacyBenchmarkBackend(
            normalization_passes=int(artifact["legacy_fixture"]["normalization_passes"])
        )
        candidate_backend = LiqFeaturesBackend()

        baseline_time = _time_per_call_cached(
            baseline_backend,
            indicators,
            data,
            warmup_calls=int(workload["warmup_calls"]),
            repetitions=int(workload["repetitions"]),
        )
        candidate_time = _time_per_call_cached(
            candidate_backend,
            indicators,
            data,
            warmup_calls=int(workload["warmup_calls"]),
            repetitions=int(workload["repetitions"]),
        )
        slowdown_x = _p90_slowdown_x(
            baseline_time=baseline_time,
            candidate_time=candidate_time,
        )
        assert slowdown_x < float(workload["p90_regression_cap_x"])

    def test_uncached_call_regression_cap(self) -> None:
        artifact = _load_artifact()
        data = _build_benchmark_data(artifact["seed"], artifact["input_size"])
        indicators = list(artifact["indicators"])
        workload = artifact["uncached_workload"]

        baseline_factory = lambda: LegacyBenchmarkBackend(
            normalization_passes=int(artifact["legacy_fixture"]["normalization_passes"])
        )
        candidate_factory = LiqFeaturesBackend

        baseline_time = _time_per_call_uncached(
            baseline_factory,
            indicators,
            data,
            repetitions=int(workload["repetitions"]),
        )
        candidate_time = _time_per_call_uncached(
            candidate_factory,
            indicators,
            data,
            repetitions=int(workload["repetitions"]),
        )
        slowdown_x = _p90_slowdown_x(
            baseline_time=baseline_time,
            candidate_time=candidate_time,
        )
        assert slowdown_x < float(workload["p90_regression_cap_x"])
