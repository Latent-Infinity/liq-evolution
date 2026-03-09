"""Stage-4 tests for regime-aware GP strategy behavior."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from liq.evolution.adapters.runner_strategy import (
    GPStrategyAdapter,
    _load_seed_programs,
)
from liq.evolution.adapters.signal_output import GPSignalOutput, RegimeState
from liq.evolution.errors import AdapterError
from liq.gp.config import GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _program() -> TerminalNode:
    return TerminalNode(name="close", output_type=Series)


def _adapter(**kwargs: Any) -> GPStrategyAdapter:
    adapter = GPStrategyAdapter(
        PrimitiveRegistry(),
        GPConfig(population_size=16, max_depth=4, generations=1),
        evaluator=MagicMock(),
        **kwargs,
    )
    adapter._program = _program()
    return adapter


def _turnover_ratio(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    signs = np.sign(values)
    non_zero_signs = signs[signs != 0]
    if non_zero_signs.size <= 1:
        return 0.0
    return float(
        np.count_nonzero(non_zero_signs[1:] != non_zero_signs[:-1])
        / (non_zero_signs.size - 1)
    )


class TestStage4RegimeOutputContract:
    def test_regime_state_is_typed_and_validated(self) -> None:
        state = RegimeState(
            label="trend",
            confidence=0.8,
            occupancy=0.7,
            reason_code="ok",
            turnover=0.0,
        )
        out = GPSignalOutput(
            scores=pl.Series("scores", [0.1, 0.2]),
            regime_state=state,
        )
        assert out.regime_state is state

        with pytest.raises(ValueError):
            RegimeState(label="trend", confidence=1.1, occupancy=0.5)

        with pytest.raises(ValueError):
            RegimeState(label="trend", confidence=0.8, occupancy=-0.1)

        with pytest.raises(ValueError):
            RegimeState(label="invalid", confidence=0.8, occupancy=0.8)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            RegimeState(label="trend", confidence=0.8, occupancy=0.8, reason_code="")

        with pytest.raises(ValueError):
            RegimeState(label="trend", confidence=0.8, occupancy=0.8, turnover=-1.0)

    def test_signal_output_validation_errors(self) -> None:
        with pytest.raises(TypeError):
            GPSignalOutput(scores=[0.1, 0.2])  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            GPSignalOutput(
                scores=pl.Series("scores", [0.1, 0.2]),
                labels=[0.0, 1.0],  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError):
            GPSignalOutput(
                scores=pl.Series("scores", [0.1, 0.2]),
                labels=pl.Series("labels", [0.0]),
            )

        with pytest.raises(TypeError):
            GPSignalOutput(
                scores=pl.Series("scores", [0.1]),
                metadata=cast(Any, []),
            )

        with pytest.raises(TypeError):
            GPSignalOutput(
                scores=pl.Series("scores", [0.1]),
                regime_state="trend",  # type: ignore[arg-type]
            )

    def test_empty_output_is_explicit(self) -> None:
        adapter = _adapter()
        out = adapter.predict(pl.DataFrame({"close": []}))
        assert out.scores.len() == 0
        assert out.regime_state is not None
        assert out.regime_state.label == "empty"
        assert out.regime_state.reason_code == "empty_features"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_no_trade_output_is_explicit(self, mock_eval) -> None:
        mock_eval.return_value = np.array([0.8, -0.7, 0.9], dtype=np.float64)
        adapter = _adapter()
        features = pl.DataFrame(
            {
                "close": [1.0, 2.0, 3.0],
                "regime_confidence": [0.2, 0.3, 0.4],
                "regime_occupancy": [1.0, 1.0, 1.0],
            }
        )
        out = adapter.predict(features)
        np.testing.assert_array_equal(out.scores.to_numpy(), np.zeros(3, dtype=np.float64))
        assert out.regime_state is not None
        assert out.regime_state.label == "no_trade"
        assert out.regime_state.reason_code == "threshold_gate"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_fallback_output_is_explicit_for_invalid_regime_inputs(self, mock_eval) -> None:
        raw = np.array([0.3, -0.2], dtype=np.float64)
        mock_eval.return_value = raw
        adapter = _adapter()
        features = pl.DataFrame(
            {
                "close": [1.0, 2.0],
                "regime_confidence": [float("nan"), 0.9],
                "regime_occupancy": [1.0, 1.0],
            }
        )
        out = adapter.predict(features)
        np.testing.assert_array_equal(out.scores.to_numpy(), raw)
        assert out.regime_state is not None
        assert out.regime_state.label == "fallback"
        assert out.regime_state.reason_code == "invalid_regime_inputs"


class TestStage4RegimeHysteresis:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"regime_confidence_threshold": 1.1},
            {"regime_occupancy_threshold": -0.1},
            {"regime_hysteresis_margin": -0.01},
            {"regime_min_persistence": 0},
        ],
        ids=[
            "invalid_confidence_threshold",
            "invalid_occupancy_threshold",
            "invalid_hysteresis_margin",
            "invalid_min_persistence",
        ],
    )
    def test_constructor_validation(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _adapter(**kwargs)

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_small_regime_noise_does_not_create_pathological_turnover(self, mock_eval) -> None:
        raw = np.array([0.12, -0.04, 0.03, -0.02, 0.11, 0.12, 0.13], dtype=np.float64)
        mock_eval.return_value = raw
        adapter = _adapter(regime_hysteresis_margin=0.1, regime_min_persistence=2)
        features = pl.DataFrame(
            {
                "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "regime_confidence": [0.9] * 7,
                "regime_occupancy": [1.0] * 7,
            }
        )
        out = adapter.predict(features)
        assert out.regime_state is not None
        assert out.regime_state.turnover < _turnover_ratio(raw)

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_hard_regime_transition_is_stable_and_switches_after_persistence(self, mock_eval) -> None:
        raw = np.array([0.2, 0.22, 0.18, -0.25, -0.27, -0.3], dtype=np.float64)
        mock_eval.return_value = raw
        adapter = _adapter(regime_hysteresis_margin=0.05, regime_min_persistence=2)
        features = pl.DataFrame(
            {
                "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "regime_confidence": [0.9] * 6,
                "regime_occupancy": [1.0] * 6,
            }
        )
        out = adapter.predict(features)
        values = out.scores.to_numpy()
        assert values[3] == 0.0
        assert values[-2] < 0.0
        assert values[-1] < 0.0
        assert out.regime_state is not None
        assert out.regime_state.label == "range"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_score_length_mismatch_routes_to_fallback(self, mock_eval) -> None:
        mock_eval.return_value = np.array([0.1], dtype=np.float64)
        adapter = _adapter()
        features = pl.DataFrame({"close": [1.0, 2.0]})
        out = adapter.predict(features)
        assert out.regime_state is not None
        assert out.regime_state.label == "fallback"
        assert out.regime_state.reason_code == "score_length_mismatch"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_non_finite_scores_route_to_fallback(self, mock_eval) -> None:
        mock_eval.return_value = np.array([0.1, float("inf")], dtype=np.float64)
        adapter = _adapter()
        features = pl.DataFrame({"close": [1.0, 2.0]})
        out = adapter.predict(features)
        np.testing.assert_array_equal(out.scores.to_numpy(), np.array([0.1, 0.0]))
        assert out.regime_state is not None
        assert out.regime_state.reason_code == "non_finite_scores"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_occupancy_threshold_gate_is_no_trade(self, mock_eval) -> None:
        mock_eval.return_value = np.array([0.3, 0.4], dtype=np.float64)
        adapter = _adapter(regime_occupancy_threshold=0.5)
        features = pl.DataFrame(
            {
                "close": [1.0, 2.0],
                "regime_confidence": [0.9, 0.9],
                "regime_occupancy": [0.1, 0.1],
            }
        )
        out = adapter.predict(features)
        assert out.regime_state is not None
        assert out.regime_state.label == "no_trade"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_missing_regime_features_is_documented(self, mock_eval) -> None:
        mock_eval.return_value = np.array([0.8, 0.81, 0.82], dtype=np.float64)
        adapter = _adapter()
        features = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        out = adapter.predict(features)
        assert out.regime_state is not None
        assert out.regime_state.reason_code == "regime_features_missing"

    def test_internal_empty_score_path_is_explicit(self) -> None:
        adapter = _adapter()
        gated, regime_state = adapter._apply_regime_gating(
            np.asarray([], dtype=np.float64),
            pl.DataFrame({"close": []}),
        )
        assert gated.size == 0
        assert regime_state.reason_code == "empty_scores"

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_long_regime_persistence_has_low_churn(self, mock_eval) -> None:
        raw = np.array([0.2] * 10, dtype=np.float64)
        mock_eval.return_value = raw
        adapter = _adapter(regime_hysteresis_margin=0.05, regime_min_persistence=2)
        features = pl.DataFrame(
            {
                "close": list(range(1, 11)),
                "regime_confidence": [0.95] * 10,
                "regime_occupancy": [1.0] * 10,
            }
        )
        out = adapter.predict(features)
        assert out.regime_state is not None
        assert out.regime_state.turnover == 0.0
        assert np.all(out.scores.to_numpy() > 0.0)


class TestStage4SeedLoadingFallbacks:
    def test_load_seed_programs_none_returns_empty(self) -> None:
        assert _load_seed_programs(None, PrimitiveRegistry()) == []

    def test_load_seed_programs_rejects_non_list_payload(self, tmp_path) -> None:
        path = tmp_path / "bad-seeds.json"
        path.write_text('{"seed_programs": {"program": "not-a-list"}}', encoding="utf-8")
        with pytest.raises(AdapterError):
            _load_seed_programs(path, PrimitiveRegistry())
