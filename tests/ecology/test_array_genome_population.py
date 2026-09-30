"""What the array-genome population does beyond the contract every population meets.

The port's own promises are asserted once, in `tests/support/contract/`, against
this adapter and the stand-in together. What is left for here is what is
particular to this one: the two descriptors it reads off the store, the refusal
of a birth with nobody to inherit from, and the fact that an offspring's every
gene is a value some parent actually carried rather than one this adapter
invented.
"""

from __future__ import annotations

import pytest

from liq.evolution.ecology.adapters import ArrayGenomePopulation
from liq.evolution.ecology.adapters.array_genome import (
    EXPOSURE_WANTED,
    FEATURES_READ,
)
from liq.evolution.ecology.adapters.null import NullBarSource
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    FULL_EXPOSURE,
    MASK_PREFIX,
    WEIGHT_PREFIX,
    AgentBirth,
    PopulationState,
)
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.learning import WEIGHT_PREFIX as LEARNED_WEIGHT_PREFIX
from liq.evolution.ecology.types import Genome

GENE_SCHEMA_VERSION = "adapter-genes-1"
FEATURE_SCHEMA_VERSION = "adapter-features-1"
MODEL_VERSION = "adapter-model-1"
DESCRIPTOR_SCHEMA_VERSION = "adapter-descriptors-1"

LEVEL = "level"
STEP = "step"

#: How fast every agent here discards what it learned. One value, inside the
#: declared range, because nothing in this module is a claim about the gene.
FORGETS_AT = 0.97

#: How many decision points are walked before the wish below is read. The
#: reading is standardised against prior-bar statistics, and at the first
#: decision point there are none — so an agent asked for a wish before any bar
#: is behind it wants nothing whatever it has learned.
WARMED_UP_AFTER = 5


def _birth(agent_id: str, *, genes: dict[str, float]) -> AgentBirth:
    """One agent entering the population, carrying exactly ``genes``."""
    return AgentBirth(
        agent_id=agent_id,
        genome=Genome(genes=genes, schema_version=GENE_SCHEMA_VERSION),
        learned_state={},
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def _reading_two_features(agent_id: str, *, level_on: float) -> AgentBirth:
    """An agent switching the level feature on or off and always reading the step."""
    return _birth(
        agent_id,
        genes={
            f"{MASK_PREFIX}{LEVEL}": level_on,
            f"{WEIGHT_PREFIX}{LEVEL}": 0.0,
            f"{MASK_PREFIX}{STEP}": 1.0,
            f"{WEIGHT_PREFIX}{STEP}": 0.0,
            ENTRY_THRESHOLD: 0.0,
            FORGETTING_FACTOR: FORGETS_AT,
        },
    )


def _population(*births: AgentBirth) -> ArrayGenomePopulation:
    """The population, over a store holding ``births``."""
    return ArrayGenomePopulation(
        state=PopulationState.founded(births, learning=LearningConfig()),
        descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION,
    )


def test_a_descriptor_says_what_an_agent_does_not_how_well_it_does_it() -> None:
    """How much of what it could read it reads, and whether it wants to be in.

    The second descriptor is read after the population has been stepped, and
    the agent is given a learned weight to be stepped with, because the wish is
    formed from what the agent has learned: an agent at its cold start reads
    nothing off any view and would report the same coordinate whatever the tape
    did.
    """
    population = _population(
        _reading_two_features("greedy", level_on=1.0),
        _reading_two_features("sparing", level_on=0.0),
    )

    assert population.descriptor("greedy").values[FEATURES_READ] == 1.0
    assert population.descriptor("sparing").values[FEATURES_READ] == 0.5
    assert population.descriptor("greedy").values[EXPOSURE_WANTED] == 0.0
    assert population.descriptor("greedy").schema_version == DESCRIPTOR_SCHEMA_VERSION

    windows = tuple(
        NullBarSource(feature_schema_version=FEATURE_SCHEMA_VERSION).windows()
    )
    instrument = next(iter(windows[0].bars))
    for warming in windows[:WARMED_UP_AFTER]:
        population.state.step(warming, instrument)
    for agent_id in population.agent_ids():
        population.state.learn(
            agent_id,
            {f"{LEARNED_WEIGHT_PREFIX}{name}": 1.0 for name in (LEVEL, STEP)},
        )
    population.state.step(windows[WARMED_UP_AFTER], instrument)

    assert population.descriptor("greedy").values[EXPOSURE_WANTED] == FULL_EXPOSURE


def test_an_agent_that_reads_nothing_sits_at_the_origin_of_that_axis() -> None:
    """A genome with no features to switch on reads none of them, not all of them."""
    population = _population(
        _birth("blank", genes={ENTRY_THRESHOLD: 0.0, FORGETTING_FACTOR: FORGETS_AT})
    )

    assert population.descriptor("blank").values[FEATURES_READ] == 0.0


def test_every_gene_an_offspring_carries_is_a_value_some_parent_carried() -> None:
    """Recombination takes; it does not average, perturb or invent."""
    population = _population(
        _reading_two_features("first", level_on=1.0),
        _reading_two_features("second", level_on=0.0),
    )

    offspring = population.spawn(parents=("first", "second"), seed=17)

    for name, value in offspring.genes.items():
        assert value in {
            population.genome("first").genes[name],
            population.genome("second").genes[name],
        }


def test_a_birth_of_nobody_is_refused_rather_than_invented() -> None:
    """An offspring with no parents would be an initialisation, not an inheritance."""
    population = _population(_reading_two_features("only", level_on=1.0))

    with pytest.raises(ValueError, match="at least one parent"):
        population.spawn(parents=(), seed=17)
