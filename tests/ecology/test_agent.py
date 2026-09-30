"""How one agent, addressed by name, behaves against the population it is.

The evidence that a wish is a reproducible function of a genome, a learned
state and a broadcast lives elsewhere: the genome and reproducibility clauses
beside the tape in the consumer repository, the learned-state clause under
`tests/facts/test_intent.py`. What is checked here is the facade's own
contract — that it is a population of one and not a second decision rule, that
what it is shown reaches what it wants, and that the refusals it inherits are
the population's own rather than a second set written over a single genome.
"""

from __future__ import annotations

import pickle

import pytest

import liq.evolution.ecology.agent as agent_module
from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    FULL_EXPOSURE,
    MASK_PREFIX,
    SWITCHED_ON_AT,
    WEIGHT_PREFIX,
    Agent,
    AgentBornUnderAnotherVocabulary,
    OutcomeFromTheSameBar,
)
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.learning import WEIGHT_PREFIX as LEARNED_WEIGHT_PREFIX
from liq.evolution.ecology.types import BarWindow, Genome

AGENT_ID = "ramp-agent"
GENE_SCHEMA_VERSION = "ramp-genes-1"
FEATURE_SCHEMA_VERSION = "ramp-features-1"
MODEL_VERSION = "ramp-model-1"

#: The two features the declared ramp broadcasts.
LEVEL = "level"
STEP = "step"

#: How fast the agent here discards what it learned. One value, inside the
#: declared range; nothing in this module is a claim about the gene.
FORGETS_AT = 0.97

#: How many of the ramp's decision points are stepped before a wish is read.
#: The reading is standardised against prior-bar statistics and one bar has no
#: spread, so an agent asked before that wants nothing whatever it has learned.
WARMED_UP_AFTER = 5

#: A bar no standardised reading of the ramp reaches.
BEYOND_ANY_READING = 1e12


def test_agent_module_keeps_its_public_api_and_class_identity() -> None:
    """Internal extractions do not change the established import surface."""
    assert agent_module.__all__ == [
        "ENTRY_THRESHOLD",
        "FORGETTING_FACTOR",
        "FULL_EXPOSURE",
        "MASK_PREFIX",
        "STORAGE_DTYPE",
        "SWITCHED_ON_AT",
        "WEIGHT_PREFIX",
        "Agent",
        "AgentBirth",
        "AgentBornUnderAnotherVocabulary",
        "AgentSnapshot",
        "AgentVersions",
        "BoundReached",
        "ForgettingFactorOutsideItsRange",
        "Lineage",
        "NothingWasShown",
        "OutcomeFromTheSameBar",
        "PopulationDoesNotLearn",
        "PopulationSnapshot",
        "PopulationState",
        "PopulationStep",
        "StorageReport",
    ]
    assert all(hasattr(agent_module, name) for name in agent_module.__all__)
    assert agent_module.Agent.__module__ == agent_module.__name__
    assert agent_module.PopulationState.__module__ == agent_module.__name__
    assert all(
        getattr(agent_module, name).__module__ == agent_module.__name__
        for name in agent_module.__all__
        if name
        in {
            "Agent",
            "AgentBirth",
            "AgentBornUnderAnotherVocabulary",
            "AgentSnapshot",
            "AgentVersions",
            "BoundReached",
            "ForgettingFactorOutsideItsRange",
            "Lineage",
            "NothingWasShown",
            "OutcomeFromTheSameBar",
            "PopulationDoesNotLearn",
            "PopulationSnapshot",
            "PopulationState",
            "PopulationStep",
            "StorageReport",
        }
    )
    versions = agent_module.AgentVersions(
        feature_schema_version="features-v1",
        model_version="model-v1",
    )
    assert pickle.loads(pickle.dumps(versions)) == versions


def _windows() -> tuple[BarWindow, ...]:
    """Every decision point the declared placeholder ramp offers."""
    return tuple(NullBarSource(feature_schema_version=FEATURE_SCHEMA_VERSION).windows())


def _instrument(window: BarWindow) -> str:
    """The single instrument the ramp carries a bar for."""
    return next(iter(window.bars))


def _agent(**genes: float) -> Agent:
    """An agent carrying exactly the genes named, at its configured cold start."""
    return Agent.founded(
        agent_id=AGENT_ID,
        genome=Genome(genes=dict(genes), schema_version=GENE_SCHEMA_VERSION),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
        learning=LearningConfig(),
    )


def _reading(entry: float) -> Agent:
    """An agent that reads the ramp's level, with its entry gene at ``entry``."""
    return _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": 1.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: entry,
            FORGETTING_FACTOR: FORGETS_AT,
        }
    )


def _warmed(agent: Agent, *, leaning: float) -> tuple[BarWindow, ...]:
    """Step ``agent`` over the ramp's opening bars and give it a weight to read."""
    windows = _windows()
    instrument = _instrument(windows[0])
    for window in windows[:WARMED_UP_AFTER]:
        agent.intend(window, instrument)
    agent.state.learn(AGENT_ID, {f"{LEARNED_WEIGHT_PREFIX}{LEVEL}": leaning})
    return windows


def test_a_reading_above_the_entry_gene_wants_the_whole_account() -> None:
    """Clearing the gene is wanted in full; there is nothing between that and flat."""
    agent = _reading(entry=0.0)
    windows = _warmed(agent, leaning=1.0)

    wish = agent.intend(windows[WARMED_UP_AFTER], _instrument(windows[0]))

    assert wish.target_exposure == FULL_EXPOSURE
    assert wish.agent_id == AGENT_ID
    assert wish.as_of == windows[WARMED_UP_AFTER].as_of


def test_a_reading_that_does_not_clear_the_entry_gene_wants_nothing() -> None:
    """The gene is a bar to be cleared, not a target to be approached."""
    agent = _reading(entry=BEYOND_ANY_READING)
    windows = _warmed(agent, leaning=1.0)

    wish = agent.intend(windows[WARMED_UP_AFTER], _instrument(windows[0]))

    assert wish.target_exposure == 0.0


def test_a_weight_the_agent_has_not_learned_yet_reads_as_nothing() -> None:
    """At its cold start an agent reads nothing off any view, and says so.

    The declared cold start is zero, so the reading is exactly zero however
    large the scaled feature is — which is a deterministic, observable starting
    behaviour rather than an accident of an uninitialised array.
    """
    agent = _reading(entry=0.0)
    windows = _windows()

    wishes = tuple(
        agent.intend(window, _instrument(windows[0])).target_exposure
        for window in windows[: WARMED_UP_AFTER + 1]
    )

    assert set(wishes) == {0.0}
    assert set(agent.learned_weights().values()) == {0.0}


def test_a_feature_switched_off_is_not_read_however_much_it_was_learned() -> None:
    """A mask below the cut takes its feature out of the reading entirely."""
    agent = _agent(
        **{
            f"{MASK_PREFIX}{LEVEL}": SWITCHED_ON_AT / 2.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: 0.0,
            FORGETTING_FACTOR: FORGETS_AT,
        }
    )
    windows = _warmed(agent, leaning=BEYOND_ANY_READING)

    wish = agent.intend(windows[WARMED_UP_AFTER], _instrument(windows[0]))

    assert wish.target_exposure == 0.0


def test_an_instrument_the_decision_point_carries_no_view_for_reads_as_nothing() -> (
    None
):
    """An absent view is a reading of nothing, never a reach for one elsewhere."""
    agent = _reading(entry=-1.0)
    windows = _warmed(agent, leaning=1.0)

    wish = agent.intend(
        windows[WARMED_UP_AFTER], "AN_INSTRUMENT_THIS_POINT_DOES_NOT_CARRY"
    )

    assert wish.target_exposure == FULL_EXPOSURE
    assert wish.instrument == "AN_INSTRUMENT_THIS_POINT_DOES_NOT_CARRY"


def test_a_genome_with_no_entry_gene_cannot_be_born() -> None:
    """Where the bar sits is inherited; there is no reading that defaults it."""
    with pytest.raises(ValueError, match=ENTRY_THRESHOLD):
        _agent(
            **{
                f"{MASK_PREFIX}{LEVEL}": 1.0,
                f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
                FORGETTING_FACTOR: FORGETS_AT,
            }
        )


def test_a_feature_switched_on_without_a_weight_cannot_be_born() -> None:
    """How much a feature counts is inherited too, and is never defaulted."""
    with pytest.raises(ValueError, match=f"{WEIGHT_PREFIX}{LEVEL}"):
        _agent(
            **{
                f"{MASK_PREFIX}{LEVEL}": 1.0,
                ENTRY_THRESHOLD: 0.0,
                FORGETTING_FACTOR: FORGETS_AT,
            }
        )


def test_an_agent_is_not_stepped_under_a_vocabulary_it_was_not_born_under() -> None:
    """A gene means what its vocabulary says; under another it means nothing."""
    from dataclasses import replace

    agent = _reading(entry=0.0)
    elsewhere = replace(_windows()[0], feature_schema_version="another-vocabulary-1")

    with pytest.raises(AgentBornUnderAnotherVocabulary, match=AGENT_ID):
        agent.intend(elsewhere, _instrument(elsewhere))


def test_an_outcome_stamped_at_the_reading_it_follows_is_refused() -> None:
    """What holding a reading earned is not known at the instant it was formed."""
    agent = _reading(entry=0.0)
    windows = _windows()
    instrument = _instrument(windows[0])
    agent.intend(windows[0], instrument)

    with pytest.raises(OutcomeFromTheSameBar):
        agent.observe(windows[0].as_of, 0.01)

    assert agent.observe(windows[1].as_of, 0.01) == ()


def test_what_an_agent_learned_moves_and_is_reported_by_feature() -> None:
    """The weights the wish reads are the weights an outcome writes."""
    agent = _reading(entry=0.0)
    windows = _windows()
    instrument = _instrument(windows[0])

    for index in range(WARMED_UP_AFTER):
        agent.intend(windows[index], instrument)
        agent.observe(windows[index + 1].as_of, 0.01 * float(index + 1))

    learned = agent.learned_weights()
    assert set(learned) == {LEVEL}
    assert learned[LEVEL] != 0.0
