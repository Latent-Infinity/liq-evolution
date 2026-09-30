"""Population construction behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Self

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.config import (
    LearningConfig,
    PopulationConfig,
)
from liq.evolution.ecology.types import (
    UtcTimestamp,
)

from .agent_contracts import (
    AgentBirth,
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    ForgettingFactorOutsideItsRange,
    Lineage,
    PopulationSnapshot,
)
from .agent_genes import (
    ENTRY_THRESHOLD,
    FLAT,
    FORGETTING_FACTOR,
    STORAGE_DTYPE,
    _gene_layout,
)

if TYPE_CHECKING:
    pass


from .population_state_fields import _PopulationStateFields


class _ConstructionMixin(_PopulationStateFields):
    def __init__(self, snapshot: PopulationSnapshot) -> None:
        """Hold the agents ``snapshot`` describes, refusing a set that is not one.

        Args:
            snapshot: The agents to hold and the shape to hold them in.

        Raises:
            ValueError: If nobody is alive, if two agents share an identity, if
                the agents do not share one gene vocabulary, one learned-state
                vocabulary and one gene schema version, if a genome switches on
                a feature it carries no weight for or carries no entry gene, if
                an outcome history is longer than the ceiling it is restored
                under, or if that ceiling retains nothing.
            ForgettingFactorOutsideItsRange: If the population learns and some
                genome carries a forgetting factor the configuration disallows.
            KeyError: If the population learns and the agents were not born
                declaring the columns the update writes, or carrying the gene
                it reads.
        """
        agents = snapshot.agents
        if not agents:
            raise ValueError(
                "a population with no agents is not a population: nothing would "
                "be stepped, and every figure computed over it would be a "
                "figure over an empty set rather than an absent one"
            )
        if snapshot.history_capacity < 1:
            raise ValueError(
                "an outcome history that retains nothing is not a history; "
                f"history_capacity was {snapshot.history_capacity}"
            )
        self._ids = tuple(held.agent_id for held in agents)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError(
                "two agents share an identity, so neither could be scored, "
                "snapshotted or inherited from without the other"
            )
        self._row_of = {agent_id: row for row, agent_id in enumerate(self._ids)}
        self.history_capacity = snapshot.history_capacity
        self.gene_schema_version = agents[0].genome.schema_version
        self.gene_names, self.feature_names = _gene_layout(agents[0].genome)
        self.learned_names = tuple(sorted(agents[0].learned_state))
        self._learned_at = {
            name: column for column, name in enumerate(self.learned_names)
        }
        self._entry_at = self.gene_names.index(ENTRY_THRESHOLD)
        self._features = len(self.feature_names)

        count = len(agents)
        self._genes = np.zeros((count, len(self.gene_names)), dtype=STORAGE_DTYPE)
        self._learned = np.zeros((count, len(self.learned_names)), dtype=STORAGE_DTYPE)
        self._intended = np.zeros(count, dtype=STORAGE_DTYPE)
        self._realised = np.zeros(count, dtype=STORAGE_DTYPE)
        self._history = np.zeros((count, self.history_capacity), dtype=STORAGE_DTYPE)
        self._retained = np.zeros(count, dtype=np.int64)
        for row, held in enumerate(agents):
            self._admit(row, held)
        self._lineage = tuple(
            Lineage(agent_id=held.agent_id, parents=held.parents, born_at=held.born_at)
            for held in agents
        )
        self._versions = tuple(
            AgentVersions(
                feature_schema_version=held.feature_schema_version,
                model_version=held.model_version,
            )
            for held in agents
        )
        self._born_under = {
            version.feature_schema_version for version in self._versions
        }
        self.learning = snapshot.learning
        self._update: online.OnlineUpdate | None = None
        self._forgetting_at: int | None = None
        self._shown: tuple[UtcTimestamp, np.ndarray] | None = None
        self._bounds_reached: list[BoundReached] = []
        if self.learning is not None:
            self._start_learning(self.learning)

    def _start_learning(self, configured: LearningConfig) -> None:
        """Bind the update to its columns and refuse a rate nobody declared."""
        if FORGETTING_FACTOR not in self.gene_names:
            raise KeyError(
                f"a population that learns reads {FORGETTING_FACTOR!r} from "
                f"every genome and these carry {sorted(self.gene_names)}; how "
                "fast an agent forgets is inherited, never defaulted"
            )
        self._forgetting_at = self.gene_names.index(FORGETTING_FACTOR)
        rates = self._genes[:, self._forgetting_at]
        outside = (rates < configured.forgetting_minimum) | (
            rates > configured.forgetting_maximum
        )
        if outside.any():
            strangers = tuple(
                agent_id
                for agent_id, out in zip(self._ids, outside.tolist(), strict=True)
                if out
            )
            raise ForgettingFactorOutsideItsRange(
                f"{strangers} carry forgetting factors outside the declared "
                f"range [{configured.forgetting_minimum}, "
                f"{configured.forgetting_maximum}]: "
                f"{sorted(set(rates[outside].tolist()))}"
            )
        self._update = online.OnlineUpdate(
            config=configured,
            features=self.feature_names,
            column_of=self._learned_at,
        )

    @classmethod
    def founded(
        cls,
        births: Sequence[AgentBirth],
        *,
        population: PopulationConfig | None = None,
        learning: LearningConfig | None = None,
    ) -> Self:
        """Start a population from ``births``, each holding nothing and wanting nothing.

        Args:
            births: Who is alive at the start, and what each was born with.
            population: What the population is held under — the outcome-history
                ceiling among it. Defaults to the schema's own declaration
                rather than to a number written at this call, so the ceiling a
                run dropped outcomes at is one of the keys its provenance
                carries.
            learning: What the online update is turned by, or ``None`` for a
                population that does not learn. When it is given, every agent
                additionally starts out carrying the update's own columns at
                the declared cold start, because what an estimator holds is
                declared by the configuration that declares the estimator.

        Returns:
            PopulationState: The population, before any decision point.
        """
        held = PopulationConfig() if population is None else population
        started: Mapping[str, float] = {}
        if learning is not None and births:
            _, features = _gene_layout(births[0].genome)
            started = online.cold_start(features, learning)
        return cls(
            PopulationSnapshot(
                agents=tuple(
                    AgentSnapshot(
                        agent_id=birth.agent_id,
                        genome=birth.genome,
                        learned_state={**birth.learned_state, **started},
                        intended=FLAT,
                        realised=FLAT,
                        history=(),
                        parents=birth.parents,
                        born_at=birth.born_at,
                        feature_schema_version=birth.feature_schema_version,
                        model_version=birth.model_version,
                    )
                    for birth in births
                ),
                history_capacity=held.history_capacity,
                learning=learning,
            )
        )

    @classmethod
    def restored(cls, snapshot: PopulationSnapshot) -> Self:
        """Take a population back out of ``snapshot``, ready to go on deciding.

        What comes back stands on its own: it holds no reference to the
        population the snapshot was taken from, and nothing it goes on to do
        consults one.

        Args:
            snapshot: The population as it was put away.

        Returns:
            PopulationState: The same population, mid-walk.
        """
        return cls(snapshot)
