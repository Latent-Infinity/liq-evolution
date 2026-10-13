"""What a birth carries across: the offspring the population port returned, and a disposition.

The heritable half of a birth is the population port's alone. The parents are
whoever its selection returned, and the genome is exactly the offspring its
``spawn`` returned. Nothing here varies, copies or repairs a gene. What is decided
here is the other half: what the offspring starts out having learned. That is a
learned-state disposition rule, and it is owned by the ecology rather than by any
evolution engine.

* **Lamarckian.** The offspring carries its first parent's whole learned state,
  verbatim — every column the update declares, read through the port at the
  instant of the birth. The first parent is the first one the selection returned
  for this birth, which is what makes the choice reproducible.
* **Baldwinian.** The offspring carries nothing and starts from its own genome's
  priors.

Either way the birth goes through the population's one admission path, which
refuses a start outside the bounds rather than repairing it. The disposition is
then recorded on the lineage, per birth.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from liq.evolution.ecology.config import Inheritance
from liq.evolution.ecology.ports import AgentPopulation
from liq.evolution.ecology.types import AgentId, Offspring, UtcTimestamp

from .agent_contracts import AgentBirth, AgentVersions

__all__ = ["birth_of"]


def birth_of(
    *,
    agent_id: AgentId,
    offspring: Offspring,
    parents: Sequence[AgentId],
    population: AgentPopulation,
    versions: AgentVersions,
    inheritance: Inheritance,
    born_at: UtcTimestamp,
    spawn_seed: int,
) -> AgentBirth:
    """The birth of ``offspring``, carrying what ``inheritance`` says it carries.

    Args:
        agent_id: The identity it is born under.
        offspring: What the population port's ``spawn`` returned. Its genome is
            the newborn's genome exactly.
        parents: Whom the selection returned for this birth, in order.
        population: Where the first parent's learned state is read from.
        versions: The vocabularies the first parent was born under; the
            offspring is born under the same.
        inheritance: What crosses the birth.
        born_at: The decision instant of the birth (UTC).
        spawn_seed: The seed the genome was drawn under.

    Returns:
        AgentBirth: The birth, ready for the population's admission path.
    """
    carried = (
        MappingProxyType(dict(population.learned_state(parents[0])))
        if inheritance is Inheritance.LAMARCKIAN
        else MappingProxyType({})
    )
    return AgentBirth(
        agent_id=agent_id,
        genome=offspring.genome,
        learned_state=carried,
        feature_schema_version=versions.feature_schema_version,
        model_version=versions.model_version,
        parents=tuple(parents),
        born_at=born_at,
        mutation_scale=offspring.mutation_scale,
        spawn_seed=spawn_seed,
    )
