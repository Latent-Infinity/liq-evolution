"""Materialization helpers for trained regime detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from liq.gp.types import BoolSeries


class _RegimeClassifierProtocol(Protocol):
    """Classifier protocol required by the terminal materializer."""

    def predict(self, features: pl.DataFrame) -> pl.Series: ...


class _RegimeLabelerProtocol(Protocol):
    """Labeler protocol required by the terminal materializer."""

    def label(self, cluster_ids: pl.Series) -> pl.Series: ...


@dataclass(frozen=True)
class TrainedRegimeTerminal:
    """Minimal terminal descriptor used by Phase 4 evidence capture."""

    name: str
    output_type: object = BoolSeries


@dataclass(frozen=True)
class MaterializedRegimeTerminals:
    """Result of trained regime terminal materialization."""

    frame: pl.DataFrame
    terminals: tuple[TrainedRegimeTerminal, ...]


__all__ = [
    "MaterializedRegimeTerminals",
    "TrainedRegimeDetectorTerminal",
    "TrainedRegimeTerminal",
]


@dataclass
class TrainedRegimeDetectorTerminal:
    """Build boolean regime membership terminals from a trained classifier."""

    classifier: _RegimeClassifierProtocol
    labeler: _RegimeLabelerProtocol
    terminal_name_prefix: str = "svm_regime_is"

    def materialize(self, features: pl.DataFrame) -> MaterializedRegimeTerminals:
        """Materialize one-hot terminal columns for regime labels."""
        cluster_ids = self.classifier.predict(features)
        label_series = self.labeler.label(cluster_ids).cast(pl.Utf8)
        labels = label_series.to_list()
        ordered_labels: list[str] = []
        seen: set[str] = set()
        for label in labels:
            text = str(label)
            if text not in seen:
                ordered_labels.append(text)
                seen.add(text)

        columns: dict[str, pl.Series] = {}
        terminals: list[TrainedRegimeTerminal] = []
        for label in ordered_labels:
            terminal_name = f"{self.terminal_name_prefix}_{label}"
            columns[terminal_name] = (label_series == label)
            terminals.append(TrainedRegimeTerminal(name=terminal_name))

        return MaterializedRegimeTerminals(
            frame=pl.DataFrame(columns),
            terminals=tuple(terminals),
        )
