"""Edge-case tests to increase coverage for validation/constraints.py."""

from __future__ import annotations

import pytest

from liq.evolution.validation import (
    ConstraintPolicy,
    constraint_max_leverage,
    constraint_no_future_reference,
    constraint_no_negative_cash,
)
from liq.evolution.validation.constraints import (
    _coerce_finite_float,
    _coerce_finite_nonneg,
    _extract_sequence_floats,
)
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _program() -> TerminalNode:
    return TerminalNode("close", Series)


class _BadFloat(int):
    """Type that is isinstance(int) but cannot convert via float()."""

    def __float__(self) -> float:
        raise TypeError("cannot coerce")


class TestCoerceFiniteNonneg:
    """Cover edge paths in _coerce_finite_nonneg."""

    def test_bool_true_returns_one(self) -> None:
        assert _coerce_finite_nonneg(True) == 1.0

    def test_bool_false_returns_zero(self) -> None:
        assert _coerce_finite_nonneg(False) == 0.0

    def test_string_returns_none(self) -> None:
        assert _coerce_finite_nonneg("abc") is None

    def test_none_returns_none(self) -> None:
        assert _coerce_finite_nonneg(None) is None

    def test_nan_returns_none(self) -> None:
        assert _coerce_finite_nonneg(float("nan")) is None

    def test_inf_returns_none(self) -> None:
        assert _coerce_finite_nonneg(float("inf")) is None

    def test_negative_inf_returns_none(self) -> None:
        assert _coerce_finite_nonneg(float("-inf")) is None

    def test_zero_returns_none(self) -> None:
        assert _coerce_finite_nonneg(0.0) is None

    def test_negative_returns_none(self) -> None:
        assert _coerce_finite_nonneg(-1.0) is None

    def test_positive_int_returns_float(self) -> None:
        assert _coerce_finite_nonneg(5) == 5.0

    def test_badly_coercible_int_subclass_returns_none(self) -> None:
        assert _coerce_finite_nonneg(_BadFloat(1)) is None

    def test_positive_float_returns_float(self) -> None:
        assert _coerce_finite_nonneg(3.14) == 3.14


class TestCoerceFiniteFloat:
    """Cover edge paths in _coerce_finite_float."""

    def test_bool_true_returns_one(self) -> None:
        assert _coerce_finite_float(True) == 1.0

    def test_bool_false_returns_zero(self) -> None:
        assert _coerce_finite_float(False) == 0.0

    def test_string_returns_none(self) -> None:
        assert _coerce_finite_float("abc") is None

    def test_none_returns_none(self) -> None:
        assert _coerce_finite_float(None) is None

    def test_inf_returns_none(self) -> None:
        assert _coerce_finite_float(float("inf")) is None

    def test_nan_returns_none(self) -> None:
        assert _coerce_finite_float(float("nan")) is None

    def test_negative_returns_value(self) -> None:
        assert _coerce_finite_float(-5.0) == -5.0

    def test_badly_coercible_int_subclass_returns_none(self) -> None:
        assert _coerce_finite_float(_BadFloat(1)) is None


class TestExtractSequenceFloats:
    """Cover edge paths in _extract_sequence_floats."""

    def test_non_list_returns_empty(self) -> None:
        assert _extract_sequence_floats("not_a_list") == []

    def test_dict_returns_empty(self) -> None:
        assert _extract_sequence_floats({"key": 1}) == []

    def test_none_values_skipped(self) -> None:
        assert _extract_sequence_floats([1.0, None, 2.0]) == [1.0, 2.0]

    def test_string_values_skipped(self) -> None:
        assert _extract_sequence_floats([1.0, "bad", 2.0]) == [1.0, 2.0]

    def test_inf_values_skipped(self) -> None:
        assert _extract_sequence_floats([1.0, float("inf"), 2.0]) == [1.0, 2.0]

    def test_absolute_mode(self) -> None:
        result = _extract_sequence_floats([-3.0, 2.0], absolute=True)
        assert result == [3.0, 2.0]


class TestConstraintNoFutureReferenceEdgePaths:
    """Cover remaining paths in constraint_no_future_reference."""

    def test_direct_flag_bool_true(self) -> None:
        """Covers the bool_penalty path via direct_flag."""
        result = constraint_no_future_reference(_program(), {"future_reference": True})
        assert result is not None
        assert result["future_reference"] == 1.0

    def test_direct_value_penalty_path(self) -> None:
        """Covers the direct_scalar path via future_reference_penalty."""
        result = constraint_no_future_reference(
            _program(), {"future_reference_penalty": 0.5}
        )
        assert result is not None
        assert result["future_reference"] == 0.5

    def test_all_three_flags_uses_max(self) -> None:
        result = constraint_no_future_reference(
            _program(),
            {
                "future_reference": True,
                "future_reference_detected": 0.5,
                "future_reference_penalty": 2.0,
            },
        )
        assert result is not None
        assert result["future_reference"] == 2.0

    def test_non_numeric_markers_return_none(self) -> None:
        result = constraint_no_future_reference(
            _program(),
            {"future_reference": "invalid", "future_reference_detected": None},
        )
        assert result is None


class TestConstraintNoNegativeCashEdgePaths:
    """Cover remaining paths in constraint_no_negative_cash."""

    def test_non_mapping_traces_returns_none(self) -> None:
        result = constraint_no_negative_cash(_program(), {"traces": "not_a_dict"})
        assert result is None

    def test_empty_cash_trace_returns_none(self) -> None:
        result = constraint_no_negative_cash(_program(), {"traces": {"cash_trace": []}})
        assert result is None

    def test_missing_cash_trace_key(self) -> None:
        result = constraint_no_negative_cash(
            _program(), {"traces": {"other_trace": [1.0]}}
        )
        assert result is None


class TestConstraintMaxLeverageEdgePaths:
    """Cover remaining paths in constraint_max_leverage."""

    def test_zero_equity_skipped_in_trace(self) -> None:
        """Bars with zero equity should be skipped, not divide by zero."""
        result = constraint_max_leverage(
            _program(),
            {
                "traces": {
                    "position_trace": [2.0, 3.0],
                    "equity_curve": [0.0, 1.0],
                },
            },
        )
        assert result is not None
        assert result["max_leverage"] == 2.0

    def test_within_limit_returns_none(self) -> None:
        result = constraint_max_leverage(
            _program(),
            {"max_leverage": 0.5},
        )
        assert result is None

    def test_no_leverage_data_returns_none(self) -> None:
        result = constraint_max_leverage(_program(), {})
        assert result is None

    def test_traces_without_position_or_equity(self) -> None:
        result = constraint_max_leverage(_program(), {"traces": {"other": [1.0]}})
        assert result is None


class TestConstraintPolicyEdgePaths:
    """Cover remaining ConstraintPolicy paths."""

    def test_enable_default_checks_populates_checks(self) -> None:
        policy = ConstraintPolicy(enable_default_checks=True)
        assert len(policy.checks) == 3

    def test_default_checks_detect_violations(self) -> None:
        policy = ConstraintPolicy(enable_default_checks=True)
        violations = policy.evaluate(
            _program(),
            {"future_reference_detected": 0.5},
        )
        assert "future_reference" in violations

    def test_non_finite_penalty_values_ignored(self) -> None:
        """Check that non-finite penalty values from checks are ignored."""

        def bad_check(_program: object, _payload: object) -> dict[str, float] | None:
            return {"test_violation": float("inf")}

        policy = ConstraintPolicy(checks=(bad_check,))
        violations = policy.evaluate(_program(), {})
        assert "test_violation" not in violations

    def test_none_penalty_from_coercion_ignored(self) -> None:
        """Check value that coerces to None (string) is ignored."""

        def string_penalty_check(
            _program: object, _payload: object
        ) -> dict[str, float] | None:
            return {"test": "not_a_float"}  # type: ignore[dict-item]

        policy = ConstraintPolicy(checks=(string_penalty_check,))
        violations = policy.evaluate(_program(), {})
        assert "test" not in violations

    def test_robustness_rollout_with_partial_limits_only(self) -> None:
        policy = ConstraintPolicy(
            robustness_rollout="standard",
            regime_coverage_floor=0.8,
        )
        violations = policy.evaluate(_program(), {"metrics": {"regime_coverage": 0.7}})

        assert "robustness:regime_coverage_below_floor" in violations
        assert "robustness:turnover_missing" not in violations
        assert "robustness:drawdown_missing" not in violations

    def test_rejects_invalid_robustness_rollout(self) -> None:
        with pytest.raises(ValueError, match="robustness_rollout"):
            ConstraintPolicy(robustness_rollout="beta")

    def test_rejects_negative_parametric_limits(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ConstraintPolicy(turnover_cap=-0.1)

    def test_rejects_regime_coverage_over_one(self) -> None:
        with pytest.raises(ValueError, match="regime_coverage_floor"):
            ConstraintPolicy(regime_coverage_floor=1.1)

    def test_rejects_non_numeric_parametric_limits(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            ConstraintPolicy(drawdown_cap="oops")  # type: ignore[arg-type]
