"""Births into, and deaths from, a population that is already walking.

**Over spare room, not a rebuilt store.** Each array is held with room for more
agents than are alive; the living occupy its leading rows, and what the rest of
the population reads are views of those rows. A death moves the last living row
into the dead agent's row, so the living stay packed and one agent's columns
move rather than everyone's. A birth takes the next free row. Only a birth with
no free row left grows the arrays, by doubling, and the storage report counts
every time that happened — so "without rebuilding the store" is a number a
caller can read rather than a property of this code it has to trust.

**One birth rule.** An admission goes through the same birth function
:meth:`founded` does, and the same refusal: a birth carries no learned state or
exactly the whole of what the update declares; a birth carrying nothing starts
from its own priors, with its founding gain held and recorded; a start outside
the bounds is refused. Here, unlike at founding, the population already exists
when a start is refused, so it counts the refusal, and a run can report how many
offspring it turned away.

**When.** Who is alive changes only between an outcome and the next decision
point. A reading awaiting its outcome is owed to exactly the agents that formed
it, so a birth or death then is refused rather than squared with it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from liq.evolution.ecology.types import AgentId, UtcTimestamp

from .agent_contracts import (
    AgentBirth,
    AgentVersions,
    DeathKind,
    Lineage,
    ReadingAwaitsItsOutcome,
    StartingStateOutsideItsBound,
)
from .agent_genes import FORGETTING_FACTOR, STORAGE_DTYPE
from .population_construction import _born, _lineage
from .population_state_fields import _PopulationStateFields

#: How much room a population grows by when a birth finds none: doubling, so
#: the number of reallocations over a run grows with the logarithm of its peak
#: size rather than with its births.
_GROWTH = 2


class _TurnoverMixin(_PopulationStateFields):
    def admit(self, birth: AgentBirth) -> Lineage:
        """Bring one agent into the living population, by the birth rule founding uses.

        Args:
            birth: Who is born, of whom, at which instant, and what it carries.
                In a population that learns it carries either nothing, and
                starts from its own genome's priors with its founding gain held
                under the state bound and recorded, or exactly the columns the
                update declares, admitted verbatim and never held.

        Returns:
            Lineage: Where the agent came from and how it was made.

        Raises:
            ReadingAwaitsItsOutcome: If a decision point has been answered and
                its outcome not yet shown.
            ValueError: If an agent of that identity is alive or has lived
                here, if it carries the aggregate book's reserved identity, if
                the birth states no instant, if it carries part of the update's
                columns or a name the update does not declare, or if it does not
                fit the population's layout.
            ForgettingFactorOutsideItsRange: If its forgetting factor is
                outside the declared range.
            StartingStateOutsideItsBound: If its starting state is outside its
                bounds. Counted in :meth:`admission_refusals`. In every case
                nothing is admitted and nothing else changes.
        """
        self._refuse_turnover_while_a_reading_awaits()
        if birth.agent_id in self._lineage_of:
            raise ValueError(
                f"{birth.agent_id!r} has already lived in this population; one "
                "identity is one life, or its lineage and its accounts would be "
                "two agents' under one name"
            )
        if birth.born_at is None:
            raise ValueError(
                f"{birth.agent_id!r} is born into a population already walking "
                "with no stated instant; only a founder is alive before the "
                "walk began"
            )
        born = _born(birth, self.learning)
        self._refuse_a_misfit(born.held)
        if self._update is not None:
            ids = (birth.agent_id,)
            self._refuse_rates_outside_the_range(
                np.asarray([birth.genome.genes[FORGETTING_FACTOR]]), ids
            )
            try:
                self._refuse_a_start_outside_the_bound(
                    np.asarray(
                        [[born.held.learned_state[n] for n in self.learned_names]],
                        dtype=STORAGE_DTYPE,
                    ),
                    np.asarray([born.carried], dtype=np.bool_),
                    ids,
                )
            except StartingStateOutsideItsBound:
                self._refusals += 1
                raise
        row = len(self._ids)
        if row == self._capacity:
            self._allocate(self._capacity * _GROWTH)
            self._reallocations += 1
        self._versions.append(
            AgentVersions(
                feature_schema_version=birth.feature_schema_version,
                model_version=birth.model_version,
            )
        )
        self._live((*self._ids, birth.agent_id))
        self._admit(row, born.held)
        lineage = _lineage(born.held)
        self._lineage_of[birth.agent_id] = lineage
        if born.founding_hold is not None:
            self._bounds_reached.append(born.founding_hold)
        return lineage

    def cull(self, agent_id: AgentId, *, died_at: UtcTimestamp) -> Lineage:
        """Remove one agent the evidence told apart as worse. See :meth:`depart`."""
        return self.depart(agent_id, died_at=died_at, kind=DeathKind.CULL)

    def retire(
        self, agent_id: AgentId, *, died_at: UtcTimestamp, protected: bool = False
    ) -> Lineage:
        """Remove one agent by age. See :meth:`depart`.

        Args:
            agent_id: Who dies.
            died_at: The instant it is removed (UTC).
            protected: Whether the evidence had told it apart as better, so it
                was retired only because nobody else was left to retire.
        """
        kind = DeathKind.PROTECTED_RETIREMENT if protected else DeathKind.RETIREMENT
        return self.depart(agent_id, died_at=died_at, kind=kind)

    def depart(
        self, agent_id: AgentId, *, died_at: UtcTimestamp, kind: DeathKind
    ) -> Lineage:
        """Remove one agent from the living, keeping where it came from and how it died.

        The one way out, whatever decided the death: the kind is written on the
        lineage and changes nothing else. Its row is taken by the last living
        agent, so the arrays are not rebuilt and no other agent's state changes;
        only the order the living are held in does.

        Args:
            agent_id: Who dies.
            died_at: The instant it is removed (UTC).
            kind: What decided the death. A label, recorded and never acted on.

        Returns:
            Lineage: Its lineage, closed with its death instant and kind, and
            readable from :meth:`lineage` and :meth:`departed` from now on.

        Raises:
            ReadingAwaitsItsOutcome: If a decision point has been answered and
                its outcome not yet shown.
            KeyError: If the agent is not alive.
            ValueError: If the death instant names no zone, or is before the
                agent's birth, if the agent is the last one alive, or if the
                kind is not one of the declared kinds.
        """
        self._refuse_turnover_while_a_reading_awaits()
        row = self._row(agent_id)
        closed = replace(
            self._lineage_of[agent_id], died_at=died_at, death_kind=DeathKind(kind)
        )
        assert closed.died_at is not None
        if closed.born_at is not None and closed.died_at < closed.born_at:
            raise ValueError(
                f"{agent_id!r} would die at {closed.died_at.isoformat()}, before "
                f"it was born at {closed.born_at.isoformat()}"
            )
        if len(self._ids) == 1:
            raise ValueError(
                f"{agent_id!r} is the last agent alive; a population with no "
                "agents is not a population"
            )
        last = len(self._ids) - 1
        for store in self._stores.values():
            store[row] = store[last]
        self._versions[row] = self._versions[last]
        self._versions.pop()
        ids = list(self._ids)
        ids[row] = ids[last]
        ids.pop()
        self._live(tuple(ids))
        self._lineage_of[agent_id] = closed
        self._departed.append(agent_id)
        return closed

    def departed(self) -> tuple[Lineage, ...]:
        """Return the lineage of every agent removed, in the order removed."""
        return tuple(self._lineage_of[agent_id] for agent_id in self._departed)

    def admission_refusals(self) -> int:
        """Return how many births admission refused for a start outside its bounds."""
        return self._refusals

    def _allocate(self, capacity: int) -> None:
        """Hold the arrays with room for ``capacity`` agents, keeping the living."""
        living = len(self._ids)
        shapes = {
            "_genes": (len(self.gene_names),),
            "_learned": (len(self.learned_names),),
            "_intended": (),
            "_realised": (),
            "_history": (self.history_capacity,),
            "_retained": (),
            "_exposures": (self.history_capacity,),
            "_exposures_retained": (),
        }
        stores: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            dtype = np.int64 if name.endswith("_retained") else STORAGE_DTYPE
            store = np.zeros((capacity, *shape), dtype=dtype)
            if name in self._stores:
                store[:living] = self._stores[name][:living]
            stores[name] = store
        self._stores = stores
        self._capacity = capacity
        self._expose(living)

    def _expose(self, living: int) -> None:
        """Point every array the population reads at the living rows of its store."""
        for name, store in self._stores.items():
            setattr(self, name, store[:living])

    def _live(self, ids: tuple[AgentId, ...]) -> None:
        """Make ``ids``, in row order, the living."""
        self._ids = ids
        self._row_of = {agent_id: row for row, agent_id in enumerate(ids)}
        self._born_under = {
            version.feature_schema_version for version in self._versions
        }
        self._expose(len(ids))

    def _refuse_turnover_while_a_reading_awaits(self) -> None:
        """Refuse a change of who is alive while an outcome is owed."""
        if self._shown is not None:
            raise ReadingAwaitsItsOutcome(
                f"a reading formed at {self._shown[0].isoformat()} awaits its "
                "outcome, which is owed to exactly the agents that formed it; "
                "admit or cull after the outcome is shown"
            )
