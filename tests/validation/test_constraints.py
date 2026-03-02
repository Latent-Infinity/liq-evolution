"""Tests for strategy constraint hooks and policy enforcement."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from liq.evolution.validation import (
    ConstraintPolicy,
    constraint_max_leverage,
    constraint_no_future_reference,
    constraint_no_negative_cash,
)
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _program() -> TerminalNode:
    return TerminalNode("close", Series)


class TestConstraintPolicy:
    def test_future_reference_check_from_payload_marker(self) -> None:
        program = _program()
        policy = ConstraintPolicy(checks=(constraint_no_future_reference,))

        violations = policy.evaluate(
            program,
            {
                "future_reference_detected": 0.75,
            },
        )

        assert violations["future_reference"] == 0.75

    def test_max_leverage_check_from_trace(self) -> None:
        program = _program()
        policy = ConstraintPolicy(checks=(constraint_max_leverage,))

        violations = policy.evaluate(
            program,
            {
                "traces": {
                    "position_trace": [3.0, 3.0],
                    "equity_curve": [1.0, 1.5],
                }
            },
        )

        assert violations["max_leverage"] == 2.0

    def test_policy_evaluates_multiple_checks_and_keeps_max_penalty(self) -> None:
        program = _program()
        policy = ConstraintPolicy(
            checks=(constraint_no_future_reference, constraint_max_leverage)
        )

        violations = policy.evaluate(
            program,
            {
                "future_reference_detected": 0.25,
                "traces": {
                    "position_trace": [3.0, 3.0],
                    "equity_curve": [1.0, 1.0],
                },
            },
        )

        assert violations["future_reference"] == 0.25
        assert violations["max_leverage"] == 2.0


class TestConstraintPropertyCases:
    """Property-style checks for constraint edge behavior."""

    @given(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    def test_future_reference_marker_penalty_is_positive_when_marked(
        self, value: float
    ) -> None:
        program = _program()
        payload = {"future_reference_detected": value}
        violations = constraint_no_future_reference(program, payload)

        if value > 0.0:
            assert violations is not None
            assert violations["future_reference"] == value
        else:
            assert violations is None

    @given(
        st.lists(
            st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_no_negative_cash_is_max_negative_abs_value(
        self, cash_trace: list[float]
    ) -> None:
        program = _program()
        payload = {"traces": {"cash_trace": cash_trace}}

        violations = constraint_no_negative_cash(program, payload)
        expected = min([c for c in cash_trace if c < 0.0], default=0.0)
        if expected < 0.0:
            assert violations == {"negative_cash": abs(expected)}
        else:
            assert violations is None

    @given(
        st.lists(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        ),
        st.lists(
            st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        ),
    )
    def test_max_leverage_matches_trace_ratio(
        self, positions: list[float], equities: list[float]
    ) -> None:
        program = _program()
        payload = {
            "traces": {
                "position_trace": positions,
                "equity_curve": equities,
            },
        }

        limit = 1.0
        expected = 0.0
        for position, equity in zip(positions, equities, strict=False):
            leverage = abs(position) / abs(equity)
            expected = max(expected, leverage - limit)

        violations = constraint_max_leverage(program, payload)
        if expected <= 0.0:
            assert violations is None
        else:
            assert violations == {"max_leverage": expected}
