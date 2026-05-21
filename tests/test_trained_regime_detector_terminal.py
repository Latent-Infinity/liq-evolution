"""Tests for trained regime detector terminal materialization."""

from __future__ import annotations

import polars as pl

from liq.evolution.terminals import TrainedRegimeDetectorTerminal, TrainedRegimeTerminal
from liq.gp.types import BoolSeries


class _Classifier:
    def predict(self, features: pl.DataFrame) -> pl.Series:
        return pl.Series("cluster_id", [0, 1, 0][: features.height])


class _Labeler:
    def label(self, cluster_ids: pl.Series) -> pl.Series:
        labels = ["trend" if value == 0 else "range" for value in cluster_ids.to_list()]
        return pl.Series("regime", labels)


def test_materialize_creates_bool_columns_and_terminal_descriptors() -> None:
    features = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    terminal = TrainedRegimeDetectorTerminal(_Classifier(), _Labeler())

    result = terminal.materialize(features)

    assert result.frame.columns == ["svm_regime_is_trend", "svm_regime_is_range"]
    assert result.frame["svm_regime_is_trend"].to_list() == [True, False, True]
    assert result.frame["svm_regime_is_range"].to_list() == [False, True, False]
    assert result.terminals == (
        TrainedRegimeTerminal("svm_regime_is_trend", BoolSeries),
        TrainedRegimeTerminal("svm_regime_is_range", BoolSeries),
    )


def test_materialize_preserves_first_seen_label_order() -> None:
    features = pl.DataFrame({"x": [1.0, 2.0]})
    terminal = TrainedRegimeDetectorTerminal(_Classifier(), _Labeler(), terminal_name_prefix="regime")

    result = terminal.materialize(features)

    assert [descriptor.name for descriptor in result.terminals] == ["regime_trend", "regime_range"]
