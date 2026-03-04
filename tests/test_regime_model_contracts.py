"""Contract tests for typed regime model primitives."""

from __future__ import annotations

import pytest

from liq.evolution.program import TerminalNode
from liq.evolution.regime_model import (
    RegimeDetector,
    RegimeExpert,
    RegimeGate,
    RegimeId,
    RegimeModel,
    RegimeRisk,
    RegimeWeights,
)
from liq.gp.types import BoolSeries, Series


def _detector() -> RegimeDetector:
    return RegimeDetector(RegimeId.trend, TerminalNode(name="detector", output_type=BoolSeries))


def _gate() -> RegimeGate:
    return RegimeGate(RegimeId.trend, TerminalNode(name="gate", output_type=BoolSeries))


def _expert(name: str) -> RegimeExpert:
    return RegimeExpert(RegimeId.trend, TerminalNode(name=name, output_type=Series))


def _risk() -> RegimeRisk:
    return RegimeRisk(RegimeId.neutral, TerminalNode(name="risk", output_type=Series))


class TestRegimeIdContracts:
    """Validate declared regime labels and enum coercion."""

    def test_declared_labels_cover_expected_values(self) -> None:
        values = {item.value for item in RegimeId}
        assert values == {
            "trend",
            "range",
            "neutral",
            "fallback",
            "no_trade",
            "empty",
        }

    def test_invalid_regime_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            RegimeId("invalid")  # type: ignore[arg-type]


class TestRegimeWeightsContracts:
    """Validate weighted blend constraints for expert blocks."""

    def test_non_empty_weights_required(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            RegimeWeights(())

    def test_non_numeric_entries_rejected(self) -> None:
        with pytest.raises(TypeError):
            RegimeWeights(("a",))  # type: ignore[arg-type]

    def test_bool_weights_rejected(self) -> None:
        with pytest.raises(TypeError):
            RegimeWeights((True,))  # type: ignore[arg-type]

    def test_negative_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RegimeWeights((-0.1,))

    def test_nonn_finite_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            RegimeWeights((float("nan"),))

    def test_len_and_iteration_match_tuple_payload(self) -> None:
        weights = RegimeWeights((0.0, 1.5, 2))
        assert len(weights) == 3
        assert tuple(weights) == (0.0, 1.5, 2.0)


class TestRegimeModelContracts:
    """Validate required role composition and deterministic defaults."""

    def test_model_requires_at_least_one_expert(self) -> None:
        with pytest.raises(ValueError, match="requires at least one expert"):
            RegimeModel(
                detector=_detector(),
                gate=_gate(),
                experts=(),
            )

    def test_model_defaults_weight_to_uniform_ones(self) -> None:
        model = RegimeModel(
            detector=_detector(),
            gate=_gate(),
            experts=(_expert("e1"), _expert("e2")),
        )
        assert model.weights is not None
        assert model.weights.values == (1.0, 1.0)

    def test_model_rejects_mismatched_weight_length(self) -> None:
        with pytest.raises(ValueError, match="length must match number"):
            RegimeModel(
                detector=_detector(),
                gate=_gate(),
                experts=(_expert("e1"),),
                weights=RegimeWeights((1.0, 2.0)),
            )

    def test_model_accepts_optional_risk_block(self) -> None:
        model = RegimeModel(
            detector=_detector(),
            gate=_gate(),
            experts=(_expert("e1"),),
            risk=_risk(),
            weights=RegimeWeights((0.5,)),
        )
        assert model.risk is not None
        assert isinstance(model.risk, RegimeRisk)

    def test_immutability(self) -> None:
        model = RegimeModel(
            detector=_detector(),
            gate=_gate(),
            experts=(_expert("e1"),),
        )
        with pytest.raises(AttributeError):
            model.detector = RegimeDetector(
                RegimeId.trend,
                TerminalNode(name="detector2", output_type=BoolSeries),
            )  # type: ignore[misc]
