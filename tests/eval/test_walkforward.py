"""Tests for walk-forward split utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.eval.walkforward import (
    build_walk_forward_splits,
    summarize_regime_participation,
)


def _hours(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(count)]


def test_build_walk_forward_splits_rolling_and_deterministic() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 31)
    first = build_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
    )
    second = build_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
    )

    assert [split.slice_id for split in first] == [split.slice_id for split in second]
    assert len(first) == 4
    assert first[0].train == slice(0, 5)
    assert first[0].validate == slice(5, 9)
    assert first[0].test == slice(9, 12)
    assert first[1].train == slice(5, 10)


def test_build_walk_forward_splits_anchored_regime() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 26)
    splits = build_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=4,
        anchored=True,
        embargo_bars=1,
    )

    assert len(splits) == 3
    assert splits[0].train == slice(0, 4)
    assert splits[0].validate == slice(5, 7)
    assert splits[0].test == slice(8, 10)
    assert splits[1].train == slice(0, 8)
    assert splits[1].validate == slice(9, 11)
    assert splits[1].test == slice(12, 14)
    assert splits[2].train == slice(0, 12)
    assert splits[2].validate == slice(13, 15)
    assert splits[2].test == slice(16, 18)
    assert isinstance(splits[0].validate, slice)
    assert isinstance(splits[0].train, slice)
    assert isinstance(splits[1].validate, slice)
    assert isinstance(splits[1].train, slice)
    assert splits[0].validate.start == splits[0].train.stop + splits[0].embargo_bars
    assert splits[1].validate.start == splits[1].train.stop + splits[1].embargo_bars


def test_build_walk_forward_splits_with_shuffle_seed_is_deterministic() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 31)
    first = build_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
        shuffle_folds=True,
        deterministic_seed=7,
    )
    second = build_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
        shuffle_folds=True,
        deterministic_seed=7,
    )

    assert [split.slice_id for split in first] == [split.slice_id for split in second]


def test_summarize_regime_participation_is_fold_scoped() -> None:
    splits = [
        WalkForwardSplit(train=slice(0, 4), validate=slice(4, 6), test=slice(6, 8)),
        WalkForwardSplit(train=slice(2, 6), validate=slice(6, 9), test=slice(9, 11)),
    ]
    regime_series = ["trend", "trend", "trend", "trend", "range", "trend", "range", "trend", "trend", "trend", "range"]
    summaries = summarize_regime_participation(splits, regime_series)

    assert summaries[0] == {"range": 2, "trend": 6}
    assert summaries[1] == {"range": 3, "trend": 6}


def test_summarize_regime_participation_with_timestamps() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 12)
    splits = build_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=2,
        test_window=2,
        step_size=4,
    )
    regime_series = ["trend"] * 12
    regime_series[2] = "range"
    regime_series[7] = "range"

    summaries = summarize_regime_participation(
        splits,
        regime_series,
        timestamps=timestamps,
    )
    assert len(summaries) == 2
    assert summaries[0] == {"range": 2, "trend": 6}
    assert summaries[1] == {"range": 1, "trend": 7}


@pytest.mark.parametrize(
    ("train_window", "validate_window", "test_window", "step_size"),
    [
        (0, 2, 2, 1),
        (2, 0, 2, 1),
        (2, 2, 0, 1),
        (2, 2, 2, 0),
    ],
)
def test_build_walk_forward_splits_rejects_non_positive_window_sizes(
    train_window: int,
    validate_window: int,
    test_window: int,
    step_size: int,
) -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)

    with pytest.raises(ValueError, match="must be"):
        build_walk_forward_splits(
            timestamps,
            train_window=train_window,
            validate_window=validate_window,
            test_window=test_window,
            step_size=step_size,
        )


def test_build_walk_forward_splits_rejects_negative_adjustments() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)

    with pytest.raises(ValueError, match="must be non-negative"):
        build_walk_forward_splits(
            timestamps,
            train_window=4,
            validate_window=3,
            test_window=2,
            step_size=4,
            label_lookahead_bars=-1,
        )


def test_build_walk_forward_splits_anchored_short_series_raises() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 8)

    with pytest.raises(ValueError, match="no complete splits"):
        build_walk_forward_splits(
            timestamps,
            train_window=4,
            validate_window=3,
            test_window=3,
            step_size=4,
            anchored=True,
        )


def test_build_walk_forward_splits_anchored_explicit_lookahead_checks() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 20)

    with pytest.raises(ValueError, match="lookahead"):
        build_walk_forward_splits(
            timestamps,
            train_window=4,
            validate_window=3,
            test_window=2,
            step_size=4,
            anchored=True,
            label_lookahead_bars=10,
        )


def test_build_walk_forward_splits_large_step_has_single_fold() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 30)

    splits = build_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=20,
        anchored=True,
    )

    assert len(splits) == 1
    assert splits[0].train == slice(0, 4)
    assert splits[0].validate == slice(4, 6)
    assert splits[0].test == slice(6, 8)


def test_build_walk_forward_splits_anchored_shuffle_is_deterministic() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 26)
    first = build_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=4,
        anchored=True,
        shuffle_folds=True,
        deterministic_seed=9,
    )
    second = build_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=4,
        anchored=True,
        shuffle_folds=True,
        deterministic_seed=9,
    )

    assert [split.slice_id for split in first] == [split.slice_id for split in second]


def test_summarize_regime_participation_requires_non_empty_regime_series() -> None:
    splits = [WalkForwardSplit(train=slice(0, 2), validate=slice(2, 3), test=slice(3, 4))]

    with pytest.raises(ValueError, match="regime_series must be non-empty"):
        summarize_regime_participation(splits, [])


def test_summarize_regime_participation_accepts_regime_mapping() -> None:
    splits = [
        WalkForwardSplit(train=slice(0, 4), validate=slice(4, 6), test=slice(6, 8)),
    ]
    regime_series = {0: "trend", 1: "trend", 2: "trend", 3: "trend", 4: "trend", 5: "range", 6: "trend", 7: "trend"}

    summaries = summarize_regime_participation(splits, regime_series)

    assert summaries == [{"range": 1, "trend": 7}]


def test_summarize_regime_participation_with_empty_splits_returns_empty() -> None:
    assert summarize_regime_participation([], regime_series=[1, 2, 3]) == []


def test_summarize_regime_participation_non_sequence_regime_series_rejected() -> None:
    splits = [
        WalkForwardSplit(train=slice(0, 1), validate=slice(1, 2), test=slice(2, 3)),
    ]

    with pytest.raises(TypeError, match="must be a sequence"):
        summarize_regime_participation(splits, "invalid-regime-series")
