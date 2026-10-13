"""What a feature switched on at birth starts from, under each inheritance arm.

A parent walks with one feature switched off. The decision never reads that
feature, so the parent's learned weight for it is moved by nothing but
forgetting: it only decays from the prior it started at. Its offspring is then
spawned through the population port with that feature's mask flipped on, and
admitted under each arm:

- **Lamarckian.** The offspring carries its parent's whole learned state, so it
  starts the feature from the parent's decayed weight, bit for bit.
- **Baldwinian.** It carries nothing, so it starts the feature from its own
  ``weight.`` gene.

The two arms therefore start a newly switched-on feature from different values.

**Nothing here is market data.** The decision points are the declared
arithmetic stream in ``tests/support/level_shift_stream.py``, with a second,
switched-off feature added by declared arithmetic: an INVALID MUTATION of that
stream, labelled as one, and evidence of nothing about any market.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from liq.evolution.ecology import Genome, agent, config, genome_schema, inheritance
from liq.evolution.ecology.adapters import ArrayGenomePopulation
from liq.evolution.ecology.learning import WEIGHT_PREFIX as LEARNED_WEIGHT_PREFIX
from tests.support import level_shift_stream as stream

#: The second feature, added to the declared stream by arithmetic, and the
#: period of the series it carries.
SWITCHED_OFF = "slow"
SWITCHED_OFF_PERIOD = 11

#: The prior the parent carries for the switched-off feature: inside the
#: declared weight bound, and away from zero so its decay shows.
PRIOR = 1.0

#: One forgetting factor, fast enough that the decay is visible after the walk.
FORGETTING = 0.95

#: How many decision points the parent walks before it breeds.
WALKED = 60

#: Seeds a flipped-on offspring is looked for under.
SEEDS = range(500)

PARENT = "parent"
DESCRIPTOR_SCHEMA_VERSION = "mask-flip-descriptors-1"


def _windows() -> tuple[agent.BarWindow, ...]:
    """The declared stream with the switched-off feature added.

    INVALID MUTATION of the declared stream: one feature is added; nothing else
    changes.
    """
    return tuple(
        replace(
            window,
            features=MappingProxyType(
                {
                    stream.INSTRUMENT: MappingProxyType(
                        {
                            stream.FEATURE: stream.feature_at(index),
                            SWITCHED_OFF: float(index % SWITCHED_OFF_PERIOD)
                            - float(SWITCHED_OFF_PERIOD // 2),
                        }
                    )
                }
            ),
        )
        for index, window in enumerate(stream.windows(WALKED + 1))
    )


def _parent() -> agent.AgentBirth:
    genes = {
        f"{agent.MASK_PREFIX}{stream.FEATURE}": 1.0,
        f"{agent.WEIGHT_PREFIX}{stream.FEATURE}": stream.PRIOR_AT,
        f"{agent.MASK_PREFIX}{SWITCHED_OFF}": 0.0,
        f"{agent.WEIGHT_PREFIX}{SWITCHED_OFF}": PRIOR,
        agent.ENTRY_THRESHOLD: stream.ENTRY_AT,
        agent.FORGETTING_FACTOR: FORGETTING,
    }
    return agent.AgentBirth(
        agent_id=PARENT,
        genome=Genome(
            genes=MappingProxyType(genes),
            schema_version=genome_schema.SCHEMA_VERSION,
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=stream.FEATURE_SCHEMA_VERSION,
        model_version=stream.MODEL_VERSION,
    )


def _walked() -> tuple[agent.PopulationState, agent.BarWindow]:
    """The parent, walked over the stream with every outcome shown."""
    state = agent.PopulationState.founded(
        (_parent(),), learning=config.LearningConfig()
    )
    windows = _windows()
    for index, window in enumerate(windows[:-1]):
        state.step(window, stream.INSTRUMENT)
        state.observe(windows[index + 1].as_of, {PARENT: stream.outcome_at(index)})
    return state, windows[-1]


def _learned_weight(state: agent.PopulationState, agent_id: str) -> float:
    return state.learned_state(agent_id)[f"{LEARNED_WEIGHT_PREFIX}{SWITCHED_OFF}"]


def _flipped_on(population: ArrayGenomePopulation) -> tuple[int, object]:
    """The first seed whose offspring switches the feature on, and the offspring."""
    for seed in SEEDS:
        offspring = population.spawn(parents=(PARENT,), seed=seed)
        if offspring.genome.genes[f"{agent.MASK_PREFIX}{SWITCHED_OFF}"] == 1.0:
            return seed, offspring
    raise AssertionError(f"no offspring among {len(SEEDS)} seeds switched it on")


def _born(arm: config.Inheritance) -> tuple[float, float, float]:
    """The parent's decayed weight, the child's gene, and what the child starts from."""
    state, at = _walked()
    population = ArrayGenomePopulation(
        state=state, descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION
    )
    seed, offspring = _flipped_on(population)
    decayed = _learned_weight(state, PARENT)
    child = f"child-{arm.value}"
    state.admit(
        inheritance.birth_of(
            agent_id=child,
            offspring=offspring,  # type: ignore[arg-type]
            parents=(PARENT,),
            population=population,
            versions=state.versions(PARENT),
            inheritance=arm,
            born_at=at.as_of,
            spawn_seed=seed,
        )
    )
    gene = offspring.genome.genes[f"{agent.WEIGHT_PREFIX}{SWITCHED_OFF}"]  # type: ignore[attr-defined]
    return decayed, gene, _learned_weight(state, child)


def test_the_switched_off_weight_only_decays_from_its_prior() -> None:
    """The premise: the parent's weight for the feature moved, toward zero only."""
    state, _ = _walked()
    decayed = _learned_weight(state, PARENT)

    assert 0.0 < abs(decayed) < abs(PRIOR)
    assert (decayed > 0.0) == (PRIOR > 0.0)


def test_a_lamarckian_offspring_starts_the_feature_from_its_parents_decayed_weight() -> (
    None
):
    """Bit for bit the parent's learned weight, not the child's own gene."""
    decayed, gene, started = _born(config.Inheritance.LAMARCKIAN)

    assert started == decayed
    assert started != gene


def test_a_baldwinian_offspring_starts_the_feature_from_its_own_gene() -> None:
    """Exactly its own ``weight.`` gene, not anything its parent learned."""
    decayed, gene, started = _born(config.Inheritance.BALDWINIAN)

    assert started == gene
    assert started != decayed
