"""Population access behavior."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from liq.evolution.ecology.types import (
    AgentId,
    Genome,
)

from .agent_contracts import (
    AgentVersions,
    Lineage,
)
from .population_state_fields import _PopulationStateFields


class _AccessMixin(_PopulationStateFields):
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Return the identities of the living agents, in the order held."""
        return self._ids

    def genome(self, agent_id: AgentId) -> Genome:
        """Return the heritable part of one agent, whole."""
        row = self._genes[self._row(agent_id)]
        return Genome(
            genes=MappingProxyType(
                dict(zip(self.gene_names, row.tolist(), strict=True))
            ),
            schema_version=self.gene_schema_version,
        )

    def learned_state(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return what one agent has learned, separately from its genome."""
        row = self._learned[self._row(agent_id)]
        return MappingProxyType(
            dict(zip(self.learned_names, row.tolist(), strict=True))
        )

    def learn(self, agent_id: AgentId, values: Mapping[str, float]) -> None:
        """Write what ``agent_id`` has learned, leaving everything else as it was.

        Args:
            agent_id: Whose learned state to write.
            values: Learned name to value. Names not given keep their value.

        Raises:
            KeyError: If the agent is not alive, or if a name was not declared
                at its birth. What an agent can learn is part of what it was
                born as; a name arriving later would be a value nothing chose
                to give it and nothing would ever inherit.
        """
        row = self._row(agent_id)
        for name, value in values.items():
            if name not in self._learned_at:
                raise KeyError(
                    f"{agent_id!r} was not born able to learn {name!r}; what an "
                    f"agent can learn is declared at its birth, and it declared "
                    f"{self.learned_names}"
                )
            self._learned[row, self._learned_at[name]] = value

    def intended(self, agent_id: AgentId) -> float:
        """Return the exposure one agent most recently wanted."""
        return float(self._intended[self._row(agent_id)])

    def realised(self, agent_id: AgentId) -> float:
        """Return the exposure one agent is actually holding."""
        return float(self._realised[self._row(agent_id)])

    def hold(self, agent_id: AgentId, exposure: float) -> None:
        """Record that ``agent_id`` is now holding ``exposure``.

        Written by whatever knows what actually traded, never inferred from what
        was wanted: the gap between the two is the thing the realised-fill rule
        exists to keep visible. Every call is also appended to the agent's
        bounded exposure history, which is what its descriptors read.
        """
        row = self._row(agent_id)
        self._realised[row] = exposure
        self._exposures_retained[row] = _append_bounded(
            self._exposures[row], int(self._exposures_retained[row]), exposure
        )

    def exposures(self, agent_id: AgentId) -> tuple[float, ...]:
        """Return one agent's retained realised exposures, oldest first."""
        row = self._row(agent_id)
        return tuple(
            self._exposures[row, : int(self._exposures_retained[row])].tolist()
        )

    def record_outcome(self, agent_id: AgentId, value: float) -> None:
        """Append one of ``agent_id``'s own outcomes to its history.

        Once the ceiling is reached the oldest outcome is dropped. That is a
        loss and it is deliberate; :meth:`history` hands back what is retained,
        never a padded window pretending to be a full one.
        """
        row = self._row(agent_id)
        self._retained[row] = _append_bounded(
            self._history[row], int(self._retained[row]), value
        )

    def history(self, agent_id: AgentId) -> tuple[float, ...]:
        """Return one agent's retained outcomes, oldest first."""
        row = self._row(agent_id)
        return tuple(self._history[row, : int(self._retained[row])].tolist())

    def record_reference_return(self, value: float) -> None:
        """Append the reference's return over one decision point, for the population.

        Written once per decision point by the accounting step, from the prices
        the accounts were marked on, and bounded by the same ceiling as every
        agent's outcome history, so the last outcomes of any agent line up with
        the last entries here.
        """
        self._reference_retained = _append_bounded(
            self._reference, self._reference_retained, value
        )

    def reference_returns(self) -> tuple[float, ...]:
        """Return the retained reference returns, oldest first."""
        return tuple(self._reference[: self._reference_retained].tolist())

    def lineage(self, agent_id: AgentId) -> Lineage:
        """Return where one agent came from, alive or dead.

        Raises:
            KeyError: If no agent of that identity ever lived here.
        """
        try:
            return self._lineage_of[agent_id]
        except KeyError:
            raise KeyError(
                f"{agent_id!r} never lived in this population; the living are "
                f"{self._ids} and the departed {tuple(self._departed)}"
            ) from None

    def versions(self, agent_id: AgentId) -> AgentVersions:
        """Return what one agent was born under."""
        return self._versions[self._row(agent_id)]


def _append_bounded(buffer: np.ndarray, retained: int, value: float) -> int:
    """Append ``value`` to a bounded history held in ``buffer``; return how many it holds.

    Once the buffer is full the oldest value is dropped, so what is retained is
    always the most recent, oldest first.
    """
    if retained < len(buffer):
        buffer[retained] = value
        return retained + 1
    buffer[:-1] = buffer[1:]
    buffer[-1] = value
    return retained
