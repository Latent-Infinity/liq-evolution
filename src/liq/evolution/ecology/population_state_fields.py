"""Typed storage layout shared by the array-backed population behaviors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.types import AgentId, BarWindow, Genome, UtcTimestamp

from .agent_contracts import (
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    Lineage,
    PendingReading,
)


class _PopulationStateFields(ABC):
    _ids: tuple[AgentId, ...]
    _row_of: dict[AgentId, int]
    history_capacity: int
    gene_schema_version: str
    gene_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    learned_names: tuple[str, ...]
    _learned_at: dict[str, int]
    _entry_at: int
    _features: int
    _genes: np.ndarray
    _learned: np.ndarray
    _intended: np.ndarray
    _realised: np.ndarray
    _history: np.ndarray
    _retained: np.ndarray
    _lineage: tuple[Lineage, ...]
    _versions: tuple[AgentVersions, ...]
    _born_under: set[str]
    learning: LearningConfig | None
    _update: online.OnlineUpdate | None
    _forgetting_at: int | None
    _shown: tuple[UtcTimestamp, np.ndarray] | None
    _bounds_reached: list[BoundReached]

    @abstractmethod
    def _row(self, agent_id: AgentId) -> int: ...

    @abstractmethod
    def _admit(self, row: int, held: AgentSnapshot) -> None: ...

    @abstractmethod
    def genome(self, agent_id: AgentId) -> Genome: ...

    @abstractmethod
    def learned_state(self, agent_id: AgentId) -> Mapping[str, float]: ...

    @abstractmethod
    def history(self, agent_id: AgentId) -> tuple[float, ...]: ...

    @abstractmethod
    def _forgetting(self) -> np.ndarray: ...

    @abstractmethod
    def _refuse_if_nothing_learns(self) -> None: ...

    @abstractmethod
    def _refuse_a_vocabulary_nobody_was_born_under(self, window: BarWindow) -> None: ...

    @abstractmethod
    def _resume_pending(
        self, pending: PendingReading | None
    ) -> tuple[UtcTimestamp, np.ndarray] | None: ...
