"""The contract every population is run against.

Tier 2. Two separations are the whole of what is checked: an agent's heritable
part is reachable apart from what it has learned, so a birth can carry one
without the other; and the diversity record decides what it keeps, so the
ecology offers an entry and reads the answer rather than placing or evicting
anything itself.

Which variation a birth applies is the adapter's business. What is contracted is
that a birth produces one offspring genome in the parents' own vocabulary, that
the same parents and seed produce it again, and that spawning does not by itself
change who is alive.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import ClassVar

import pytest

from liq.evolution.ecology import AgentPopulation, ArchiveEntry, Descriptor, Genome
from liq.evolution.ecology.adapters import NullAgentPopulation

RECORDED_AT = datetime(2000, 1, 1, tzinfo=UTC)

# Declared probe objectives, not measurements: one named figure offered twice
# unchanged and then higher, so a record is asked to keep, to refuse and to
# reconsider without being told which answer is right.
OFFERED_SCORES = (0.0, 0.0, 1.0)


def offer(population: AgentPopulation, agent_id: str, score: float) -> ArchiveEntry:
    """Return an entry describing ``agent_id`` as the population currently has it."""
    return ArchiveEntry(
        agent_id=agent_id,
        genome=population.genome(agent_id),
        descriptor=population.descriptor(agent_id),
        objectives=MappingProxyType({"score": score}),
        recorded_at=RECORDED_AT,
    )


class AgentPopulationContract:
    """What a population promises, whichever variation and record it holds.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyAgentPopulation(AgentPopulationContract):
            adapter_factory = staticmethod(MyAgentPopulation)

    The factory returns a fresh population for each test, so a record filled by
    one test is never read by another.
    """

    adapter_factory: ClassVar[Callable[[], AgentPopulation]]

    @pytest.fixture
    def adapter(self) -> AgentPopulation:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    def test_the_adapter_satisfies_the_port(self, adapter: AgentPopulation) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, AgentPopulation)

    def test_the_living_agents_are_separately_identifiable(
        self, adapter: AgentPopulation
    ) -> None:
        """A population that cannot name its members cannot be scored per agent."""
        agent_ids = adapter.agent_ids()
        assert isinstance(agent_ids, tuple)
        assert agent_ids != ()
        assert len(set(agent_ids)) == len(agent_ids)
        assert all(agent_id for agent_id in agent_ids)

    def test_the_heritable_and_the_learned_parts_are_reached_separately(
        self, adapter: AgentPopulation
    ) -> None:
        """A single combined state could not express what a birth carries across."""
        for agent_id in adapter.agent_ids():
            genome = adapter.genome(agent_id)
            learned = adapter.learned_state(agent_id)
            assert isinstance(genome, Genome)
            assert isinstance(learned, Mapping)
            assert genome.genes != {}
            assert genome.schema_version != ""
            assert genome.genes is not learned
            for name, value in genome.genes.items():
                assert name
                assert math.isfinite(value)
            for name, value in learned.items():
                assert name
                assert math.isfinite(value)

    def test_behaviour_is_described_independently_of_quality(
        self, adapter: AgentPopulation
    ) -> None:
        """Diversity is kept on what an agent does, on a comparable scale."""
        for agent_id in adapter.agent_ids():
            descriptor = adapter.descriptor(agent_id)
            assert isinstance(descriptor, Descriptor)
            assert descriptor.values != {}
            assert descriptor.schema_version != ""
            for name, value in descriptor.values.items():
                assert name
                assert 0.0 <= value <= 1.0

    def test_a_birth_yields_one_offspring_in_the_parents_vocabulary(
        self, adapter: AgentPopulation
    ) -> None:
        """The ecology asks for a child, not for an operator; it gets one genome."""
        parents = adapter.agent_ids()[:2]
        lineage = [adapter.genome(agent_id) for agent_id in parents]
        offspring = adapter.spawn(parents=parents, seed=17)
        assert isinstance(offspring, Genome)
        assert offspring.schema_version == lineage[0].schema_version
        inherited = {name for genome in lineage for name in genome.genes}
        assert offspring.genes != {}
        assert set(offspring.genes) <= inherited
        for value in offspring.genes.values():
            assert math.isfinite(value)

    def test_a_birth_is_reproducible(self, adapter: AgentPopulation) -> None:
        """The same parents and the same seed produce the same offspring."""
        parents = adapter.agent_ids()[:2]
        assert adapter.spawn(parents=parents, seed=17) == adapter.spawn(
            parents=parents, seed=17
        )

    def test_a_birth_does_not_by_itself_change_who_is_alive(
        self, adapter: AgentPopulation
    ) -> None:
        """Spawning hands back a genome; placing it is a separate decision."""
        before = adapter.agent_ids()
        adapter.spawn(parents=before[:2], seed=17)
        assert adapter.agent_ids() == before

    def test_the_record_decides_what_it_keeps_and_says_so(
        self, adapter: AgentPopulation
    ) -> None:
        """An entry is offered, not placed; the answer and the contents agree."""
        agent_id = adapter.agent_ids()[0]
        for score in OFFERED_SCORES:
            entry = offer(adapter, agent_id, score)
            before = adapter.recorded()
            kept = adapter.record(entry)
            after = adapter.recorded()
            assert isinstance(kept, bool)
            if kept:
                assert entry in after
            else:
                assert after == before

    def test_one_entry_offered_twice_is_held_once(
        self, adapter: AgentPopulation
    ) -> None:
        """Offering the same entry again describes no further behaviour."""
        entry = offer(adapter, adapter.agent_ids()[0], 0.0)
        adapter.record(entry)
        after_first = adapter.recorded()
        adapter.record(entry)
        assert adapter.recorded() == after_first

    def test_the_record_hands_back_an_immutable_snapshot(
        self, adapter: AgentPopulation
    ) -> None:
        """Reading the record cannot change it, and a snapshot stays as taken."""
        agent_id = adapter.agent_ids()[0]
        assert adapter.recorded() == adapter.recorded()
        adapter.record(offer(adapter, agent_id, 0.0))
        snapshot = adapter.recorded()
        assert isinstance(snapshot, tuple)
        assert all(isinstance(entry, ArchiveEntry) for entry in snapshot)
        held = list(snapshot)
        adapter.record(offer(adapter, agent_id, 1.0))
        assert list(snapshot) == held


class TestNullAgentPopulation(AgentPopulationContract):
    """The stand-in, held to the same contract as any provider-backed population."""

    adapter_factory = staticmethod(NullAgentPopulation)
