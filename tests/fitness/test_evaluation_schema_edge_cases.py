"""Edge-case tests to increase coverage for evaluation_schema.py."""

from __future__ import annotations

from liq.evolution.fitness.evaluation_schema import (
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    SchemaValidationError,
    validate_evaluation_metadata,
    validate_objective_vector,
    validate_slice_id,
)


def _valid_metadata() -> dict:
    return {
        METADATA_KEY_PER_SPLIT_METRICS: {
            "split:a": {"train": 0.1, "val": 0.2}
        },
        METADATA_KEY_RAW_OBJECTIVES: (1.0,),
        METADATA_KEY_BEHAVIOR_DESCRIPTORS: {
            "holding_period_proxy": 0.5,
            "turnover": 0.1,
        },
        METADATA_KEY_CONSTRAINT_VIOLATIONS: {},
        METADATA_KEY_SLICE_SCORES: {
            "time_window:2023-01": 0.2,
        },
    }


class TestNonMappingMetadata:
    """Cover the early-return when metadata is not a Mapping."""

    def test_non_mapping_metadata_returns_error(self) -> None:
        errors = validate_evaluation_metadata(
            "not_a_dict",  # type: ignore[arg-type]
            expected_objective_count=1,
        )
        assert len(errors) == 1
        assert errors[0].field == "metadata"
        assert "must be a dict" in errors[0].message


class TestPerSplitMetricsValidation:
    """Cover validation branches for per_split_metrics."""

    def test_non_mapping_per_split_metrics(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_PER_SPLIT_METRICS] = "bad"
        errors = validate_evaluation_metadata(
            metadata, expected_objective_count=1
        )
        assert any(
            err.field == METADATA_KEY_PER_SPLIT_METRICS
            and "dict[str, dict[str, float]]" in err.message
            for err in errors
        )

    def test_non_string_split_id_in_per_split_metrics(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_PER_SPLIT_METRICS] = {123: {"train": 0.1}}
        errors = validate_evaluation_metadata(
            metadata, expected_objective_count=1
        )
        assert any(
            "split ids must be strings" in err.message for err in errors
        )


class TestDictStrToFloatValidation:
    """Cover validation branches for dict[str, float] helper."""

    def test_non_string_key_in_slice_scores(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = {123: 0.5}
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any("keys must be strings" in err.message for err in errors)

    def test_non_mapping_slice_scores(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_SLICE_SCORES] = "bad"
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any(
            err.field == METADATA_KEY_SLICE_SCORES
            and "must be a dict" in err.message
            for err in errors
        )


class TestRequiredSliceScoresMissing:
    """Cover the require_slice_scores=True + missing path."""

    def test_slice_scores_required_but_missing(self) -> None:
        metadata = _valid_metadata()
        metadata.pop(METADATA_KEY_SLICE_SCORES)
        errors = validate_evaluation_metadata(
            metadata,
            expected_objective_count=1,
            require_slice_scores=True,
        )
        assert any(
            "is required when selection uses lexicase" in err.message
            for err in errors
        )


class TestConstraintViolationsValidation:
    """Cover the negative-value constraint violation branch."""

    def test_negative_constraint_violation_value(self) -> None:
        metadata = _valid_metadata()
        metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS] = {"test_v": -0.5}
        errors = validate_evaluation_metadata(
            metadata, expected_objective_count=1
        )
        assert any(
            "violation values should be non-negative" in err.message
            for err in errors
        )


class TestObjectiveDirectionsValidation:
    """Cover objective_directions length mismatch branch."""

    def test_directions_length_mismatch(self) -> None:
        errors = validate_evaluation_metadata(
            _valid_metadata(),
            expected_objective_count=1,
            objective_directions=("maximize", "minimize"),
        )
        assert any(
            "length" in err.message and "objective_directions" in err.field
            for err in errors
        )


class TestObjectiveVectorValidation:
    """Cover objective vector edge branches."""

    def test_non_sequence_raw_objectives(self) -> None:
        issues: list[SchemaValidationError] = []
        result = validate_objective_vector("not_a_tuple", 1, issues=issues)
        assert result is None
        assert any("must be a sequence" in err.message for err in issues)

    def test_non_numeric_objective_value(self) -> None:
        issues: list[SchemaValidationError] = []
        result = validate_objective_vector(("bad",), 1, issues=issues)
        assert result is None
        assert any("must be numeric" in err.message for err in issues)

    def test_non_finite_objective_value(self) -> None:
        issues: list[SchemaValidationError] = []
        result = validate_objective_vector(
            (float("inf"),), 1, issues=issues
        )
        assert result is None
        assert any("must be finite" in err.message for err in issues)


class TestSliceIdValidation:
    """Cover all slice ID prefix branches."""

    def test_event_prefix(self) -> None:
        assert validate_slice_id("event:earnings")

    def test_liquidity_prefix(self) -> None:
        assert validate_slice_id("liquidity:high")

    def test_instrument_prefix(self) -> None:
        assert validate_slice_id("instrument:SPY")

    def test_time_window_prefix(self) -> None:
        assert validate_slice_id("time_window:2023")

    def test_legacy_window_prefix(self) -> None:
        assert validate_slice_id("window_0:start=0:end=5")

    def test_invalid_prefix_rejected(self) -> None:
        assert not validate_slice_id("bad:prefix")
