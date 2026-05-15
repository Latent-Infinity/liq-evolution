"""Materialize trained SVM regime outputs as GP detector terminals."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from liq.core import RegimeId
from liq.features.regime import RegimeClassifier, RegimeLabeler
from liq.gp.program.ast import TerminalNode
from liq.gp.types import BoolSeries

_REGIME_TERMINALS = (
    RegimeId.trend,
    RegimeId.range,
    RegimeId.neutral,
    RegimeId.fallback,
    RegimeId.no_trade,
    RegimeId.empty,
)


@dataclass(frozen=True)
class TrainedRegimeDetectorMaterialization:
    """Precomputed context columns and terminal nodes for trained regimes."""

    frame: pl.DataFrame
    terminals: tuple[TerminalNode, ...]


@dataclass(frozen=True)
class TrainedRegimeDetectorTerminal:
    """Expose trained regime classifier output as immutable BoolSeries terminals."""

    classifier: RegimeClassifier
    labeler: RegimeLabeler
    prefix: str = "svm_regime"

    def materialize(
        self, features: pl.DataFrame
    ) -> TrainedRegimeDetectorMaterialization:
        clusters = self._predict_clusters(features)
        labels = self._label_clusters(clusters)
        columns: dict[str, list[int | bool]] = {
            f"{self.prefix}_cluster": clusters.to_list()
        }
        terminals: list[TerminalNode] = []
        label_values = labels.to_list()
        for regime in _REGIME_TERMINALS:
            column_name = f"{self.prefix}_is_{regime.value}"
            columns[column_name] = [label == regime.value for label in label_values]
            terminals.append(TerminalNode(name=column_name, output_type=BoolSeries))
        return TrainedRegimeDetectorMaterialization(
            frame=pl.DataFrame(columns),
            terminals=tuple(terminals),
        )

    def _predict_clusters(self, features: pl.DataFrame) -> pl.Series:
        try:
            return self.classifier.predict(features)
        except RuntimeError as error:
            raise RuntimeError(
                "classifier must be fitted before materializing terminals"
            ) from error

    def _label_clusters(self, clusters: pl.Series) -> pl.Series:
        if not self.labeler.mapping:
            raise RuntimeError("labeler must be fitted before materializing terminals")
        return self.labeler.label(clusters)
