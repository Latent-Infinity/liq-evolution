"""Population snapshot behavior."""

from __future__ import annotations

from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
)

from .agent_contracts import (
    AgentBornUnderAnotherVocabulary,
    AgentSnapshot,
    PopulationSnapshot,
    StorageReport,
)
from .population_state_fields import _PopulationStateFields


class _SnapshotMixin(_PopulationStateFields):
    def snapshot(self) -> PopulationSnapshot:
        """Put the whole population away, whole, as it stands."""
        return PopulationSnapshot(
            agents=tuple(
                AgentSnapshot(
                    agent_id=agent_id,
                    genome=self.genome(agent_id),
                    learned_state=self.learned_state(agent_id),
                    intended=float(self._intended[row]),
                    realised=float(self._realised[row]),
                    history=self.history(agent_id),
                    parents=self._lineage[row].parents,
                    born_at=self._lineage[row].born_at,
                    feature_schema_version=self._versions[row].feature_schema_version,
                    model_version=self._versions[row].model_version,
                )
                for row, agent_id in enumerate(self._ids)
            ),
            history_capacity=self.history_capacity,
            learning=self.learning,
        )

    def storage(self) -> StorageReport:
        """Return the shape the population is actually held in."""
        return StorageReport(
            agents=len(self._ids),
            gene_columns=len(self.gene_names),
            feature_columns=self._features,
            learned_columns=len(self.learned_names),
            history_capacity=self.history_capacity,
            dtype=str(self._genes.dtype),
            contiguous=all(
                array.flags["C_CONTIGUOUS"]
                for array in (self._genes, self._learned, self._history)
            ),
        )

    def _row(self, agent_id: AgentId) -> int:
        """The row ``agent_id`` occupies, or a refusal naming who is alive."""
        try:
            return self._row_of[agent_id]
        except KeyError:
            raise KeyError(
                f"{agent_id!r} is not alive in this population; the living are "
                f"{self._ids}"
            ) from None

    def _admit(self, row: int, held: AgentSnapshot) -> None:
        """Write one agent into ``row``, refusing one that does not fit the layout."""
        genes = held.genome.genes
        if held.genome.schema_version != self.gene_schema_version:
            raise ValueError(
                f"{held.agent_id!r} carries gene schema "
                f"{held.genome.schema_version!r} where the population holds "
                f"{self.gene_schema_version!r}; one population reads one gene "
                "vocabulary, because a column means different things under two"
            )
        if set(genes) != set(self.gene_names):
            raise ValueError(
                f"{held.agent_id!r} carries genes {sorted(genes)} where the "
                f"population's vocabulary is {sorted(self.gene_names)}; a gene "
                "nobody else carries has no column, and a column nobody fills "
                "would be read as zero"
            )
        if sorted(held.learned_state) != sorted(self.learned_names):
            raise ValueError(
                f"{held.agent_id!r} declares it can learn "
                f"{sorted(held.learned_state)} where the population declares "
                f"{sorted(self.learned_names)}; what is learnable is a column, "
                "so it is the same set for everyone or it is not a column"
            )
        if len(held.history) > self.history_capacity:
            raise ValueError(
                f"{held.agent_id!r} retained {len(held.history)} outcomes and is "
                f"being restored under a ceiling of {self.history_capacity}; "
                "restoring it would drop outcomes the snapshot recorded"
            )
        self._genes[row] = [genes[name] for name in self.gene_names]
        self._learned[row] = [held.learned_state[name] for name in self.learned_names]
        self._intended[row] = held.intended
        self._realised[row] = held.realised
        self._history[row, : len(held.history)] = held.history
        self._retained[row] = len(held.history)

    def _refuse_a_vocabulary_nobody_was_born_under(self, window: BarWindow) -> None:
        """Refuse a decision point stamped with a vocabulary some agent is not of."""
        if self._born_under == {window.feature_schema_version}:
            return
        strangers = tuple(
            agent_id
            for agent_id, version in zip(self._ids, self._versions, strict=True)
            if version.feature_schema_version != window.feature_schema_version
        )
        raise AgentBornUnderAnotherVocabulary(
            f"the decision point speaks feature vocabulary "
            f"{window.feature_schema_version!r} and {strangers} were born under "
            f"{sorted(self._born_under - {window.feature_schema_version})}; "
            "their weights refer to names that vocabulary does not define, so a "
            "score computed under it would not be a worse score but a meaningless one"
        )
