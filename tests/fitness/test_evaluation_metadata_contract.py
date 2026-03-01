"""Tests for phase-0 fitness metadata contract validation."""

from __future__ import annotations

import math
import pytest

from liq.evolution.fitness.evaluation_schema import (
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    validate_slice_id,
    to_loss_form,
    validate_evaluation_metadata,
)


def _valid_metadata() -> dict:
    """Return a fully populated, schema-valid metadata payload."""
    return {
        METADATA_KEY_PER_SPLIT_METRICS: {
            "split:a": {
                "train": 0.1,
                "val": 0.2,
            }
        },
        METADATA_KEY_RAW_OBJECTIVES: (1.0,),
        METADATA_KEY_BEHAVIOR_DESCRIPTORS: {
            "holding_period_proxy": 0.5,
            "turnover": 0.1,
        },
        METADATA_KEY_CONSTRAINT_VIOLATIONS: {},
        METADATA_KEY_SLICE_SCORES: {
            "time_window:2023-01": 0.2,
            "instrument:SPY": 0.1,
        },
    }


class TestValidateEvaluationMetadata:
    """validate_evaluation_metadata enforces the phase-0 shape contract."""

    def test_valid_payload_with_required_slice_scores(self) -> None:
        errors = validate_evaluation_metadata(
            _valid_metadata(),
            expected_objective_count=1,
            require_slice_scores=True,
            objective_directions=("maximize",),
        )
        assert errors == []

    def test_valid_payload_without_slice_scores(self) -> None:
        metadata = _valid_metadata()
        metadata.pop("slice_scores")
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=False,
            objective_directions=("maximize",),
        )
        assert errors == []

    def test_fails_on_missing_required_keys(self) -> None:
        errors = validate_evaluation_metadata(
            {},
            expected_objective_count=1,
        )
        keys = {err.field for err in errors}
        assert METADATA_KEY_PER_SPLIT_METRICS in keys
        assert METADATA_KEY_RAW_OBJECTIVES in keys
        assert METADATA_KEY_BEHAVIOR_DESCRIPTORS in keys
        assert METADATA_KEY_CONSTRAINT_VIOLATIONS in keys

    def test_fails_on_objective_dimensionality_mismatch(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_RAW_OBJECTIVES] = (1.0, 2.0)
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            objective_directions=("maximize",),
        )
        assert any("length" in err.message for err in errors)

    def test_fails_on_malformed_objective_direction(self) -> None:
        errors = validate_evaluation_metadata(
            _valid_metadata(),
            expected_objective_count=1,
            objective_directions=("up",),
        )
        assert any("must be 'maximize' or 'minimize'" in err.message for err in errors)

    def test_fails_when_slice_scores_not_dict_of_floats(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = {"a": "bad"}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any(err.field == METADATA_KEY_SLICE_SCORES for err in errors)

    def test_fails_when_slice_scores_contain_nan_or_inf(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = {"time_window:bad": float("inf")}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any("finite" in err.message for err in errors)

    def test_fails_on_unsupported_behavior_descriptor_keys(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_BEHAVIOR_DESCRIPTORS] = {"unknown": 1.0}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            objective_directions=("maximize",),
        )
        assert any("unsupported behavior descriptor key" in err.message for err in errors)

    def test_fails_on_invalid_slice_id(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = {"badprefix:2023": 0.1}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any("unsupported slice id" in err.message for err in errors)

    def test_validates_legacy_window_slice_id(self) -> None:
        assert validate_slice_id("window_0:start=0:end=5")
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = {"window_0:start=0:end=5": 0.1}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert errors == []


class TestToLossForm:
    """to_loss_form converts raw objectives into finite loss values."""

    def test_maps_nan_to_penalty(self) -> None:
        value = to_loss_form(float("nan"), "maximize")
        assert value == 1e6

    def test_maps_inf_to_penalty(self) -> None:
        value = to_loss_form(float("inf"), "minimize")
        assert value == 1e6

    def test_maps_pos_inf_to_penalty_in_maximize(self) -> None:
        value = to_loss_form(float("inf"), "maximize")
        assert value == 1e6

    def test_maps_negative_inf_in_maximize_to_penalty(self) -> None:
        value = to_loss_form(float("-inf"), "maximize")
        assert value == 1e6

    def test_maps_negative_inf_in_minimize_to_penalty(self) -> None:
        value = to_loss_form(float("-inf"), "minimize")
        assert value == 1e6

    def test_nan_to_penalty_in_minimize(self) -> None:
        value = to_loss_form(float("nan"), "minimize")
        assert value == 1e6

    def test_clamps_large_values(self) -> None:
        value = to_loss_form(1e308, "minimize", max_finite=1e12)
        assert math.isfinite(value)
        assert value == 1e12

    def test_clamps_negative_large_values(self) -> None:
        value = to_loss_form(-1e308, "minimize", max_finite=1e2)
        assert value == -1e2

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction must be 'maximize' or 'minimize'"):
            to_loss_form(1.0, "bad")  # type: ignore[arg-type]
