"""Population snapshot behavior."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    UtcTimestamp,
)

from .agent_contracts import (
    AgentBornUnderAnotherVocabulary,
    AgentSnapshot,
    PendingReading,
    PopulationSnapshot,
    StorageReport,
)
from .agent_genes import STORAGE_DTYPE
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
                    exposures=self.exposures(agent_id),
                    parents=self._lineage_of[agent_id].parents,
                    born_at=self._lineage_of[agent_id].born_at,
                    feature_schema_version=self._versions[row].feature_schema_version,
                    model_version=self._versions[row].model_version,
                    mutation_scale=self._lineage_of[agent_id].mutation_scale,
                    spawn_seed=self._lineage_of[agent_id].spawn_seed,
                    learned_state_disposition=self._lineage_of[
                        agent_id
                    ].learned_state_disposition,
                )
                for row, agent_id in enumerate(self._ids)
            ),
            history_capacity=self.history_capacity,
            learning=self.learning,
            bounds_reached=tuple(self._bounds_reached),
            pending=self._pending(),
            departed=tuple(self._lineage_of[agent_id] for agent_id in self._departed),
            admission_refusals=self._refusals,
            reference_returns=self.reference_returns(),
        )

    def _hold_the_reference_returns(self, returns: tuple[float, ...]) -> None:
        """Take the population's reference returns into a store of the ceiling's size.

        Refused rather than trimmed when the snapshot carries more than the
        ceiling retains, for the same reason an outcome history is.
        """
        if len(returns) > self.history_capacity:
            raise ValueError(
                f"the snapshot retained {len(returns)} reference returns and is "
                f"being restored under a ceiling of {self.history_capacity}; "
                "restoring it would drop reference returns the snapshot recorded"
            )
        self._reference = np.zeros(self.history_capacity, dtype=STORAGE_DTYPE)
        self._reference[: len(returns)] = returns
        self._reference_retained = len(returns)

    def _pending(self) -> PendingReading | None:
        """The reading awaiting its outcome, by agent, or ``None`` if none is."""
        if self._shown is None:
            return None
        as_of, standardised = self._shown
        return PendingReading(
            as_of=as_of,
            readings=MappingProxyType(
                {
                    agent_id: tuple(standardised[row].tolist())
                    for row, agent_id in enumerate(self._ids)
                }
            ),
        )

    def _resume_pending(
        self, pending: PendingReading | None
    ) -> tuple[UtcTimestamp, np.ndarray] | None:
        """Take back the reading a snapshot was put away awaiting the outcome of.

        Refused rather than trimmed or padded when it does not fit: a reading
        for an agent not alive here, or missing one that is, or of another
        width, is a reading an outcome could not be paired with.
        """
        if pending is None:
            return None
        if self._update is None:
            raise ValueError(
                "the snapshot carries a reading awaiting its outcome, and the "
                "population it restores to does not learn, so nothing could "
                "ever take that outcome"
            )
        if set(pending.readings) != set(self._ids):
            raise ValueError(
                f"the pending reading is for {sorted(pending.readings)} where "
                f"the living are {sorted(self._ids)}; an outcome is owed to "
                "exactly the agents that formed the reading"
            )
        widths = {len(reading) for reading in pending.readings.values()}
        if widths != {self._features}:
            raise ValueError(
                f"the pending reading carries {sorted(widths)} values per agent "
                f"where the population reads {self._features} features"
            )
        return pending.as_of, np.asarray(
            [pending.readings[agent_id] for agent_id in self._ids],
            dtype=STORAGE_DTYPE,
        )

    def storage(self) -> StorageReport:
        """Return the shape the population is actually held in."""
        return StorageReport(
            agents=len(self._ids),
            capacity=self._capacity,
            reallocations=self._reallocations,
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
        """Write one agent into ``row``, refusing one that does not fit the layout.

        The only writer of an agent's columns, whichever path brought it in.
        """
        self._refuse_a_misfit(held)
        genes = held.genome.genes
        self._genes[row] = [genes[name] for name in self.gene_names]
        self._learned[row] = [held.learned_state[name] for name in self.learned_names]
        self._intended[row] = held.intended
        self._realised[row] = held.realised
        self._history[row, : len(held.history)] = held.history
        self._retained[row] = len(held.history)
        self._exposures[row, : len(held.exposures)] = held.exposures
        self._exposures_retained[row] = len(held.exposures)

    def _refuse_a_misfit(self, held: AgentSnapshot) -> None:
        """Refuse an agent that does not fit the population's layout."""
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
        if len(held.exposures) > self.history_capacity:
            raise ValueError(
                f"{held.agent_id!r} retained {len(held.exposures)} exposures and "
                f"is being restored under a ceiling of {self.history_capacity}; "
                "restoring it would drop exposures the snapshot recorded"
            )

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
