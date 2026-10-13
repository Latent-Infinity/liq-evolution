"""Evidence that where an agent came from outlives the agent and its archive entry.

Two removals happen to an agent independently. It dies — it leaves the living
population — and its entry in the diversity record is evicted when an entry
that dominates it lands in the same cell. Neither removal may take its lineage
with it: who its parents were and the bar it was born at must still be
answerable from the population, and again from a snapshot of the population
restored afterwards. So must the kind of its death: a cull, decided by
evidence, or a retirement, decided by age — the two are told apart in every
record (``liq-docs/plans/oracle-v3-indistinguishable-cull-2026-09-30.md`` §4.1),
and a retirement is checked here beside the cull.

The eviction is observed, not assumed. The check offers a dominating entry to
the agent's own cell and asserts the agent's entry is gone from the platform's
archive before it asks for the lineage, so a record that never evicts cannot
pass this by having nothing to lose.

The two removals are also held apart, in both directions: a death removes no
entry from the record, and an eviction kills no living agent.

Nothing here is market data. The population is the declared arithmetic
stream's genome in ``tests/support/level_shift_stream.py``, and the figures
offered to the record are declared probe values, not measurements.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import MappingProxyType

from liq.evolution.ecology import ArchiveEntry
from liq.evolution.ecology.adapters import ArrayGenomePopulation
from liq.evolution.ecology.agent import AgentBirth, PopulationState
from liq.evolution.ecology.types import AgentId
from tests.support import level_shift_stream as stream

DESCRIPTOR_SCHEMA_VERSION = "lineage-descriptors-1"

#: The two founders the offspring is born of, and a bystander that outlives it.
MOTHER = "mother"
FATHER = "father"
CHILD = "child"
SIBLING = "sibling"

#: The stream's decision points the offspring is born and dies at.
BORN_AT = 3
DIES_AT = 7

#: Declared probe figures, not measurements: the offspring's entry, and a
#: strictly higher one that must displace it from a cell holding one entry.
ITS_FIGURE = 0.0
DOMINATING_FIGURE = 1.0

#: How fast the declared genome forgets; one value inside the declared range,
#: because nothing here is a claim about the gene.
FORGETS_AT = 0.97


def _instant(index: int) -> datetime:
    """The stream's decision instant at ``index``."""
    return stream.windows()[index].as_of


def _offspring() -> AgentBirth:
    """An offspring of both founders, born mid-walk."""
    return replace(
        stream.birth(CHILD, forgetting=FORGETS_AT),
        parents=(MOTHER, FATHER),
        born_at=_instant(BORN_AT),
    )


def _population() -> ArrayGenomePopulation:
    """Two founders and their offspring, under the platform's archive."""
    state = PopulationState.founded(
        (
            stream.birth(MOTHER, forgetting=FORGETS_AT),
            stream.birth(FATHER, forgetting=FORGETS_AT),
        )
    )
    state.admit(_offspring())
    return ArrayGenomePopulation(
        state=state, descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION
    )


def _entry(
    population: ArrayGenomePopulation, agent_id: AgentId, figure: float
) -> ArchiveEntry:
    """An entry describing ``agent_id`` as the population has it now."""
    return ArchiveEntry(
        agent_id=agent_id,
        genome=population.genome(agent_id),
        descriptor=population.descriptor(agent_id),
        objectives=MappingProxyType({"score": figure}),
        recorded_at=_instant(BORN_AT),
    )


def _held_by_the_archive(population: ArrayGenomePopulation) -> set[AgentId]:
    """Whose entries the platform's archive itself holds, read off the archive."""
    return {
        entry.agent_id
        for entry in population.archive.elites()  # type: ignore[attr-defined]
        if isinstance(entry, ArchiveEntry)
    }


def test_lineage_survives_eviction() -> None:
    """A dead agent evicted from the archive still answers who bore it, and when.

    The offspring is recorded, dies, and has its entry displaced by a
    dominating one in the same cell. Its parents and birth bar are then read
    from the population and from a snapshot restored afterwards.
    """
    population = _population()
    offspring_entry = _entry(population, CHILD, ITS_FIGURE)
    assert population.record(offspring_entry)

    population.state.cull(CHILD, died_at=_instant(DIES_AT))
    assert CHILD not in population.agent_ids()

    rival = _entry(population, MOTHER, DOMINATING_FIGURE)
    assert rival.descriptor == offspring_entry.descriptor, (
        "the dominating entry must fall in the offspring's own cell"
    )
    assert population.record(rival)

    assert CHILD not in _held_by_the_archive(population), "nothing was evicted"
    assert offspring_entry not in population.recorded()

    for state in (
        population.state,
        PopulationState.restored(population.state.snapshot()),
    ):
        lineage = state.lineage(CHILD)
        assert lineage.parents == (MOTHER, FATHER)
        assert lineage.born_at == _instant(BORN_AT)
        assert lineage.died_at == _instant(DIES_AT)
        assert lineage.death_kind == "cull"


def test_a_retired_agents_lineage_and_death_kind_survive_eviction() -> None:
    """A retired agent evicted from the archive still answers who bore it, when, and how it died.

    Retirement is the death decided by age rather than evidence, and a retired
    agent the evidence had marked better is a protected retirement. Both kinds
    are recovered from the population and from a snapshot restored afterwards,
    beside a culled agent's, so a record that wrote one kind for every death
    cannot pass.
    """
    population = _population()
    offspring_entry = _entry(population, CHILD, ITS_FIGURE)
    assert population.record(offspring_entry)
    population.state.admit(replace(_offspring(), agent_id=SIBLING))

    population.state.retire(CHILD, died_at=_instant(DIES_AT))
    population.state.retire(SIBLING, died_at=_instant(DIES_AT), protected=True)
    assert CHILD not in population.agent_ids()

    assert population.record(_entry(population, MOTHER, DOMINATING_FIGURE))
    assert CHILD not in _held_by_the_archive(population), "nothing was evicted"

    for state in (
        population.state,
        PopulationState.restored(population.state.snapshot()),
    ):
        retired = state.lineage(CHILD)
        assert retired.parents == (MOTHER, FATHER)
        assert retired.born_at == _instant(BORN_AT)
        assert retired.died_at == _instant(DIES_AT)
        assert retired.death_kind == "retirement"
        assert state.lineage(SIBLING).death_kind == "protected_retirement"
        assert state.lineage(MOTHER).death_kind is None


def test_a_death_removes_no_entry_from_the_record() -> None:
    """Dying leaves the living population, not the diversity record."""
    population = _population()
    entry = _entry(population, CHILD, ITS_FIGURE)
    population.record(entry)
    before = population.recorded()

    population.state.cull(CHILD, died_at=_instant(DIES_AT))

    assert population.recorded() == before
    assert CHILD in _held_by_the_archive(population)


def test_an_eviction_kills_no_living_agent() -> None:
    """Losing a cell to a dominating entry leaves the agent alive and unchanged."""
    population = _population()
    entry = _entry(population, CHILD, ITS_FIGURE)
    population.record(entry)
    living = population.agent_ids()
    genome = population.genome(CHILD)

    population.record(_entry(population, MOTHER, DOMINATING_FIGURE))

    assert CHILD not in _held_by_the_archive(population)
    assert population.agent_ids() == living
    assert population.genome(CHILD) == genome
