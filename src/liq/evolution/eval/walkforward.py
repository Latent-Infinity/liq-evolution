"""Walk-forward split helpers for stage-3 regime-aware evaluation."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from liq.datasets.walk_forward import WalkForwardSplit, generate_walk_forward_splits

__all__ = ["build_walk_forward_splits", "summarize_regime_participation"]


def build_walk_forward_splits(
    timestamps: Sequence[datetime],
    *,
    train_window: int,
    validate_window: int,
    test_window: int,
    step_size: int,
    anchored: bool = False,
    embargo_bars: int = 0,
    purge_bars: int = 0,
    label_lookahead_bars: int = 0,
    shuffle_folds: bool = False,
    deterministic_seed: int | None = None,
) -> list[WalkForwardSplit]:
    """Build walk-forward splits with optional anchored regime.

    The helper intentionally separates from :func:`generate_walk_forward_splits`
    by exposing explicit anchored/rolling behavior and optional deterministic
    fold shuffling for reproducibility.
    """
    if train_window <= 0 or validate_window <= 0 or test_window <= 0:
        raise ValueError("window sizes must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if embargo_bars < 0 or purge_bars < 0 or label_lookahead_bars < 0:
        raise ValueError("embargo_bars, purge_bars, and label_lookahead_bars must be non-negative")

    if not anchored:
        splits = generate_walk_forward_splits(
            timestamps,
            train_window=train_window,
            validate_window=validate_window,
            test_window=test_window,
            step_size=step_size,
            embargo_bars=embargo_bars,
            label_lookahead_bars=label_lookahead_bars,
        )
        if shuffle_folds:
            rng = random.Random(deterministic_seed if deterministic_seed is not None else 0)
            splits = sorted(splits, key=lambda split: split.slice_id)
            rng.shuffle(splits)
        return splits

    n = len(timestamps)
    splits: list[WalkForwardSplit] = []
    split_idx = 0
    cursor = 0
    while True:
        train_stop = train_window + cursor
        validate_start = train_stop + embargo_bars + purge_bars
        validate_stop = validate_start + max(1, validate_window - 1)
        test_start = validate_stop + embargo_bars + purge_bars
        test_stop = test_start + test_window

        if test_stop > n:
            break

        if test_stop + step_size >= n:
            break

        if train_stop <= label_lookahead_bars:
            raise ValueError("lookahead too large for train window")
        if validate_stop <= train_stop:  # pragma: no cover
            raise ValueError("lookahead too large for validate window")
        if test_stop <= validate_stop:  # pragma: no cover
            raise ValueError("lookahead too large for test window")

        train_stop = train_stop - label_lookahead_bars
        validate_stop = validate_stop - label_lookahead_bars
        test_stop = test_stop - label_lookahead_bars
        validate_start = max(0, validate_start - label_lookahead_bars)
        test_start = max(0, test_start - label_lookahead_bars)

        splits.append(
            WalkForwardSplit(
                train=slice(0, train_stop),
                validate=slice(validate_start, validate_stop),
                test=slice(test_start, test_stop),
                slice_id=(
                    f"time_window:anchored:{split_idx}"
                    f":start=0:end={train_window + cursor}"
                ),
                embargo_bars=embargo_bars + purge_bars,
            )
        )

        split_idx += 1
        cursor += step_size
        if cursor >= n:  # pragma: no cover
            break

    if not splits:
        raise ValueError("walk-forward parameters produced no complete splits")

    if shuffle_folds:
        rng = random.Random(deterministic_seed if deterministic_seed is not None else 0)
        splits = sorted(splits, key=lambda split: split.slice_id)
        rng.shuffle(splits)
    return splits


def _to_list_or_raise(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError("regime_series must be a sequence")


def summarize_regime_participation(
    splits: Sequence[WalkForwardSplit],
    regime_series: Sequence[Any] | Mapping[int, Any],
    *,
    timestamps: Sequence[datetime] | None = None,
) -> list[dict[str, int]]:
    """Count regime participation per split across walk-forward windows.

    Returns counts by serialized regime label for each fold.
    """
    if not splits:
        return []

    if isinstance(regime_series, Mapping):
        values = list(regime_series.values())
    else:
        values = list(_to_list_or_raise(regime_series))

    if not values:
        raise ValueError("regime_series must be non-empty")

    summaries: list[dict[str, int]] = []
    for split in splits:
        bars = split
        if timestamps is not None:
            bars = split.to_bar_slices(timestamps)

        counter: Counter[str] = Counter()
        for slice_ in (bars.train, bars.validate, bars.test):
            if isinstance(slice_, slice):
                segment = values[slice_]
            else:
                if timestamps is None:
                    raise TypeError("timestamp tuple slices require timestamps")
                start_ts, end_ts = slice_
                segment = [
                    value
                    for timestamp, value in zip(timestamps, values, strict=False)
                    if start_ts <= timestamp < end_ts
                ]
            counter.update(str(value) for value in segment)
        summaries.append(dict(sorted(counter.items(), key=lambda item: item[0])))
    return summaries
