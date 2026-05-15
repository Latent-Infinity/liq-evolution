"""Tests for trained regime detector terminal materialization."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from liq.core import RegimeId
from liq.evolution.terminals.trained_regime_detector import (
    TrainedRegimeDetectorTerminal,
)
from liq.features.regime import RegimeClassifier, RegimeLabeler
from liq.gp.types import BoolSeries


@dataclass
class StubClassifier(RegimeClassifier):
    fitted: bool = True
    n_regimes: int = 3

    def fit(self, features: pl.DataFrame, y: pl.Series | None = None) -> StubClassifier:
        self.fitted = True
        return self

    def predict(self, features: pl.DataFrame) -> pl.Series:
        if not self.fitted:
            raise RuntimeError("classifier must be fit")
        return pl.Series("cluster_id", [index % 3 for index in range(features.height)])

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "regime_0": [1.0, 0.0, 0.0] * (features.height // 3),
                "regime_1": [0.0, 1.0, 0.0] * (features.height // 3),
                "regime_2": [0.0, 0.0, 1.0] * (features.height // 3),
            }
        )


@dataclass
class StubLabeler(RegimeLabeler):
    fitted: bool = True

    @property
    def mapping(self) -> dict[int, RegimeId]:
        if not self.fitted:
            return {}
        return {0: RegimeId.trend, 1: RegimeId.range, 2: RegimeId.neutral}

    def fit(self, classifier: RegimeClassifier, features: pl.DataFrame) -> StubLabeler:
        self.fitted = True
        return self

    def label(self, cluster_ids: pl.Series) -> pl.Series:
        return pl.Series(
            "regime",
            [self.mapping[int(cluster_id)].value for cluster_id in cluster_ids],
        )


def _features() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_trained_regime_terminal_materializes_six_boolean_series_and_terminals() -> (
    None
):
    materialized = TrainedRegimeDetectorTerminal(
        classifier=StubClassifier(),
        labeler=StubLabeler(),
    ).materialize(_features())

    assert materialized.frame.columns == [
        "svm_regime_cluster",
        "svm_regime_is_trend",
        "svm_regime_is_range",
        "svm_regime_is_neutral",
        "svm_regime_is_fallback",
        "svm_regime_is_no_trade",
        "svm_regime_is_empty",
    ]
    assert materialized.frame["svm_regime_is_trend"].to_list() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert materialized.frame["svm_regime_is_fallback"].to_list() == [False] * 6
    assert [terminal.output_type for terminal in materialized.terminals] == [
        BoolSeries
    ] * 6


def test_trained_regime_terminal_rejects_unfitted_classifier_or_labeler() -> None:
    with pytest.raises(RuntimeError, match="classifier"):
        TrainedRegimeDetectorTerminal(
            classifier=StubClassifier(fitted=False), labeler=StubLabeler()
        ).materialize(_features())

    with pytest.raises(RuntimeError, match="labeler"):
        TrainedRegimeDetectorTerminal(
            classifier=StubClassifier(), labeler=StubLabeler(fitted=False)
        ).materialize(_features())


def test_trained_regime_terminal_is_deterministic() -> None:
    terminal = TrainedRegimeDetectorTerminal(
        classifier=StubClassifier(), labeler=StubLabeler()
    )

    first = terminal.materialize(_features())
    second = terminal.materialize(_features())

    assert first.frame.rows() == second.frame.rows()
    assert first.terminals == second.terminals
