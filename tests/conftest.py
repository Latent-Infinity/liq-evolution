"""Shared test fixtures for liq-evolution."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_ohlcv() -> dict[str, np.ndarray]:
    """Deterministic 50-bar OHLCV data for testing."""
    rng = np.random.default_rng(42)
    n = 50
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return {
        "open": close + rng.standard_normal(n) * 0.1,
        "high": close + np.abs(rng.standard_normal(n) * 0.3),
        "low": close - np.abs(rng.standard_normal(n) * 0.3),
        "close": close,
        "volume": np.abs(rng.standard_normal(n) * 1000) + 1000,
    }
