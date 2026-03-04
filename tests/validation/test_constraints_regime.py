"""Stage-6 robustness gate tests for strategy constraints."""

from __future__ import annotations

import pytest

from liq.evolution.validation import ConstraintPolicy
from liq.gp.program.ast import Program, TerminalNode


def _program() -> Program:
    return TerminalNode(name="signal", output_type=float)


def test_constraint_policy_rejects_invalid_robustness_rollout() -> None:
    with pytest.raises(ValueError, match="robustness_rollout"):
        ConstraintPolicy(robustness_rollout="beta")


def test_constraint_policy_emits_explainable_robustness_reasons() -> None:
    policy = ConstraintPolicy(
        robustness_rollout="standard",
        regime_coverage_floor=0.60,
        turnover_cap=0.35,
        drawdown_cap=0.20,
    )
    violations = policy.evaluate(
        _program(),
        {
            "metrics": {
                "regime_coverage": 0.25,
                "turnover": 0.50,
                "max_drawdown": 0.35,
            }
        },
    )

    assert violations["robustness:regime_coverage_below_floor"] == pytest.approx(0.35)
    assert violations["robustness:turnover_above_cap"] == pytest.approx(0.15)
    assert violations["robustness:max_drawdown_above_cap"] == pytest.approx(0.15)


def test_constraint_policy_staged_rollout_adjusts_thresholds() -> None:
    payload = {"metrics": {"regime_coverage": 0.55, "turnover": 0.54, "max_drawdown": 0.23}}
    canary = ConstraintPolicy(
        robustness_rollout="canary",
        regime_coverage_floor=0.60,
        turnover_cap=0.50,
        drawdown_cap=0.25,
    )
    strict = ConstraintPolicy(
        robustness_rollout="strict",
        regime_coverage_floor=0.60,
        turnover_cap=0.50,
        drawdown_cap=0.25,
    )

    assert canary.evaluate(_program(), payload) == {}
    strict_violations = strict.evaluate(_program(), payload)
    assert strict_violations["robustness:regime_coverage_below_floor"] == pytest.approx(
        0.11
    )
    assert strict_violations["robustness:turnover_above_cap"] == pytest.approx(0.09)
    assert strict_violations["robustness:max_drawdown_above_cap"] == pytest.approx(
        0.005
    )


def test_constraint_policy_missing_metrics_emit_structured_reasons() -> None:
    policy = ConstraintPolicy(
        robustness_rollout="standard",
        regime_coverage_floor=0.55,
        turnover_cap=0.30,
        drawdown_cap=0.20,
    )
    violations = policy.evaluate(_program(), payload={})
    assert violations == {
        "robustness:regime_coverage_missing": 1.0,
        "robustness:turnover_missing": 1.0,
        "robustness:drawdown_missing": 1.0,
    }
