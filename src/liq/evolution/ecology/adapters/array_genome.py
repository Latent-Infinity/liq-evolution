"""Keeping the population, over the arrays the population is actually held in.

This is the capability "hold the living agents and the record of what behaviours
have been seen", satisfied over :class:`~liq.evolution.ecology.agent.PopulationState`.
It owns no state of its own beyond the record below: every question about an
agent is answered by the store, so there is one representation of a population
and not two that have to be kept in step.

**Two halves of this are deliberately the least-committed thing that satisfies
the contract, and saying which is the point.**

*Variation.* A birth here is uniform recombination: for each gene, the offspring
takes the value one of its parents actually carries, chosen by the seed. It
invents no value, and — this is why it was chosen over averaging with a
perturbation — it introduces no magnitude, no scale and no rate that anybody
would later have to justify. Which operator family this population should really
breed with is a question to be settled by measurement, against the array-genome
primitives, and a number written here now would be a tuned parameter nobody
tuned. With one parent it is a clone, which is what "recombination of one" means
and is stated rather than special-cased.

*The record.* Every distinct entry offered is kept, and offering the same entry
twice holds it once. That is a decision, and it is reported as one. What a
diversity record *evicts* — which cell an entry falls in, what it displaces,
what coverage it leaves — belongs to the archive this ecology composes rather
than defines, and inventing an eviction policy here would be a second archive
growing quietly beside the one that already exists. So none is invented: this
keeps what it is offered and says so, until the real record is wired in behind
the same port.

*The descriptors.* Two figures, both read off what the agent does rather than
how well it does it, both normalised to the unit interval, and both stamped with
a vocabulary the caller names. They are provisional in the same way as the two
above: the descriptor set this ecology will actually niche on is a separate
piece of work, and when it arrives it bumps the schema version rather than
silently changing what an old archive's coordinates meant.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from liq.evolution.ecology.agent import (
    MASK_PREFIX,
    SWITCHED_ON_AT,
    PopulationState,
)
from liq.evolution.ecology.types import (
    AgentId,
    ArchiveEntry,
    Descriptor,
    Genome,
)

__all__ = [
    "EXPOSURE_WANTED",
    "FEATURES_READ",
    "ArrayGenomePopulation",
]

#: Descriptor name: how much of what it could read an agent switches on. A
#: behaviour, not a quality — a parsimonious agent and a greedy one are
#: different, and neither is thereby better.
FEATURES_READ = "features_read"

#: Descriptor name: whether the agent currently wants to be in the market. The
#: rule wants all of its account or none of it, so this is already on the unit
#: interval by construction rather than by clipping.
EXPOSURE_WANTED = "exposure_wanted"


@dataclass
class ArrayGenomePopulation:
    """Hold the living agents over an array-genome store, and what was recorded.

    Attributes:
        state: The population itself. Every question about an agent's heritable
            part, its learned part or where it sits is answered from here, so
            this adapter cannot drift from the thing it is describing.
        descriptor_schema_version: Vocabulary the descriptor names below belong
            to. Required rather than defaulted, because an archive whose
            coordinates carry no vocabulary cannot be compared with another one
            safely and would be compared anyway.
    """

    state: PopulationState
    descriptor_schema_version: str

    _kept: list[ArchiveEntry] = field(default_factory=list, init=False, repr=False)

    def agent_ids(self) -> tuple[AgentId, ...]:
        """Return the identities of the living agents."""
        return self.state.agent_ids()

    def genome(self, agent_id: AgentId) -> Genome:
        """Return the heritable part of one agent."""
        return self.state.genome(agent_id)

    def learned_state(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return what one agent has learned, separately from its genome."""
        return self.state.learned_state(agent_id)

    def descriptor(self, agent_id: AgentId) -> Descriptor:
        """Return where one agent currently sits in behaviour space."""
        genes = self.state.genome(agent_id).genes
        features = self.state.feature_names
        switched_on = sum(
            1 for name in features if genes[f"{MASK_PREFIX}{name}"] >= SWITCHED_ON_AT
        )
        return Descriptor(
            values=MappingProxyType(
                {
                    FEATURES_READ: switched_on / len(features) if features else 0.0,
                    EXPOSURE_WANTED: abs(self.state.intended(agent_id)),
                }
            ),
            schema_version=self.descriptor_schema_version,
        )

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Genome:
        """Return one offspring genome of ``parents``, with variation applied.

        Raises:
            ValueError: If no parents were named. A birth of nobody would have
                to invent an entire genome, which is a different operation from
                inheritance and is not this one.
        """
        if not parents:
            raise ValueError(
                "a birth needs at least one parent: an offspring of nobody would "
                "be an initialisation, not an inheritance"
            )
        lineage = [self.state.genome(agent_id) for agent_id in parents]
        draw = random.Random(seed)
        return Genome(
            genes=MappingProxyType(
                {
                    name: lineage[draw.randrange(len(lineage))].genes[name]
                    for name in sorted(lineage[0].genes)
                }
            ),
            schema_version=lineage[0].schema_version,
        )

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the diversity record; report whether it was kept."""
        if entry in self._kept:
            return False
        self._kept.append(entry)
        return True

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the diversity record currently holds."""
        return tuple(self._kept)
