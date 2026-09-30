"""Population access behavior."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

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
        exists to keep visible.
        """
        self._realised[self._row(agent_id)] = exposure

    def record_outcome(self, agent_id: AgentId, value: float) -> None:
        """Append one of ``agent_id``'s own outcomes to its history.

        Once the ceiling is reached the oldest outcome is dropped. That is a
        loss and it is deliberate; :meth:`history` hands back what is retained,
        never a padded window pretending to be a full one.
        """
        row = self._row(agent_id)
        retained = int(self._retained[row])
        if retained < self.history_capacity:
            self._history[row, retained] = value
            self._retained[row] = retained + 1
            return
        self._history[row, :-1] = self._history[row, 1:]
        self._history[row, -1] = value

    def history(self, agent_id: AgentId) -> tuple[float, ...]:
        """Return one agent's retained outcomes, oldest first."""
        row = self._row(agent_id)
        return tuple(self._history[row, : int(self._retained[row])].tolist())

    def lineage(self, agent_id: AgentId) -> Lineage:
        """Return where one agent came from."""
        return self._lineage[self._row(agent_id)]

    def versions(self, agent_id: AgentId) -> AgentVersions:
        """Return what one agent was born under."""
        return self._versions[self._row(agent_id)]
