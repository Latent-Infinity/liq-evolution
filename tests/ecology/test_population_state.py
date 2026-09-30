"""How the population behaves on the genomes and views this repository can build.

The evidence that the two parts of an agent come apart, and that a snapshot
preserves what an agent goes on to do, lives under `tests/facts/`. What is
checked here is everything around that: the layout the genes are held in, the
set of populations that are refused rather than silently misread, the arithmetic
agreeing with the one-agent rule it is the wide form of, and the ceiling the
outcome history keeps.

Nothing here is market data. The decision points are the declared arithmetic
ramp; every number computed from them is a number about the ramp.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FULL_EXPOSURE,
    MASK_PREFIX,
    WEIGHT_PREFIX,
    Agent,
    AgentBirth,
    AgentBornUnderAnotherVocabulary,
    AgentSnapshot,
    Lineage,
    PopulationSnapshot,
    PopulationState,
)
from liq.evolution.ecology.config import LearningConfig, PopulationConfig
from liq.evolution.ecology.learning import WEIGHT_PREFIX as LEARNED_WEIGHT_PREFIX
from liq.evolution.ecology.types import BarWindow, Genome

GENE_SCHEMA_VERSION = "population-genes-1"
FEATURE_SCHEMA_VERSION = "population-features-1"
MODEL_VERSION = "population-model-1"

#: The two features the declared ramp broadcasts.
LEVEL = "level"
STEP = "step"

#: One of the columns the update declares, written directly by the checks
#: that are about what an agent can learn rather than about how it learns.
#: A learning population declares exactly the update's columns at birth, so
#: there is no other name to write.
OBSERVED = f"{LEARNED_WEIGHT_PREFIX}{STEP}"

#: A gene the step reads only through the update, carried by every genome here
#: because a population that learns refuses one that does not carry it.
FORGETTING = "forgetting_factor"

#: How many of the ramp's decision points are stepped before a wish is read.
#: Two would do — the standardiser needs a spread and one bar has none — and a
#: few more keeps the reading away from the boundary.
WARMED_UP_AFTER = 5

#: A learned weight far enough either side of nothing that the reading it
#: produces clears, or fails to clear, an entry gene of zero without the check
#: depending on the size of the scaled feature.
LEANS_LONG = 1.0

#: An entry gene no standardised reading of this ramp can reach, in either
#: direction. A sentinel, so which branch a row takes is decided by the gene.
BEYOND_ANY_READING = 1e12


def _genes(*, step_weight: float = 0.0, entry: float = 0.0) -> dict[str, float]:
    """A readable genome over both ramp features, plus one gene nothing steps on."""
    return {
        f"{MASK_PREFIX}{LEVEL}": 0.0,
        f"{WEIGHT_PREFIX}{LEVEL}": 0.0,
        f"{MASK_PREFIX}{STEP}": 1.0,
        f"{WEIGHT_PREFIX}{STEP}": step_weight,
        ENTRY_THRESHOLD: entry,
        FORGETTING: 0.97,
    }


def _birth(
    agent_id: str, *, step_weight: float = 0.0, entry: float = 0.0
) -> AgentBirth:
    """One agent entering the population."""
    return AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=_genes(step_weight=step_weight, entry=entry),
            schema_version=GENE_SCHEMA_VERSION,
        ),
        learned_state={},
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def _founded(*births: AgentBirth, history_capacity: int = 4) -> PopulationState:
    """A population of ``births``, or of one agent where none is named.

    The ceiling arrives as the configured key it now is, rather than as a
    number handed to the constructor: what an outcome history drops is a
    declaration a run records, not a choice a call site makes. The learning
    configuration arrives for a harder reason — the decision rule reads what an
    agent has learned, so a population built without one has no weight for it
    to read and refuses to be stepped at all.
    """
    return PopulationState.founded(
        births or (_birth("only"),),
        population=PopulationConfig(history_capacity=history_capacity),
        learning=LearningConfig(),
    )


def _warmed(state: PopulationState, *, leaning: float) -> tuple[BarWindow, ...]:
    """Step ``state`` over the ramp's opening bars and give it a weight to read.

    Two things have to be true before a wish means anything, and neither is
    true at a population's first decision point: the online standardiser needs
    at least two distinct bars behind it before it has a spread to scale by,
    and the agent needs a learned weight to multiply the scaled reading with.
    Both are arranged here rather than waited for, because what these checks
    are about is the shape of the wish and not the estimator's convergence.
    """
    windows = tuple(
        NullBarSource(feature_schema_version=FEATURE_SCHEMA_VERSION).windows()
    )
    instrument = _instrument(windows[0])
    for window in windows[:WARMED_UP_AFTER]:
        state.step(window, instrument)
    for agent_id in state.agent_ids():
        state.learn(
            agent_id,
            {f"{LEARNED_WEIGHT_PREFIX}{name}": leaning for name in (LEVEL, STEP)},
        )
    return windows


def _window(index: int = 0) -> BarWindow:
    """One decision point of the declared ramp, stamped with this vocabulary."""
    source = NullBarSource(feature_schema_version=FEATURE_SCHEMA_VERSION)
    return tuple(source.windows())[index]


def _instrument(window: BarWindow) -> str:
    """The single instrument the ramp carries a bar for."""
    return next(iter(window.bars))


def test_the_genes_are_held_as_two_aligned_blocks_and_whatever_else() -> None:
    """Masks first, weights in the same order, everything else after them."""
    state = _founded()

    assert state.feature_names == (LEVEL, STEP)
    assert state.gene_names == (
        f"{MASK_PREFIX}{LEVEL}",
        f"{MASK_PREFIX}{STEP}",
        f"{WEIGHT_PREFIX}{LEVEL}",
        f"{WEIGHT_PREFIX}{STEP}",
        ENTRY_THRESHOLD,
        FORGETTING,
    )


def test_a_gene_the_step_does_not_read_is_still_carried() -> None:
    """A genome hands back everything it inherited, not just the read columns."""
    state = _founded()

    assert state.genome("only").genes[FORGETTING] == 0.97
    assert state.genome("only").schema_version == GENE_SCHEMA_VERSION


def test_one_agent_addressed_by_name_is_the_population_and_not_a_second_rule() -> None:
    """The facade computes nothing; it hands back the population's own wish.

    There was a second implementation of the decision rule, written over a
    single genome, and it is gone: two forms of one rule agree until the same
    sum is added in two orders. What is asserted here is the replacement's
    contract — the agent's wish is the wish the population formed for that
    agent, object for object, and the agent's learned state is the population's
    row rather than a copy that could fall behind it.
    """
    genome = Genome(genes=_genes(), schema_version=GENE_SCHEMA_VERSION)
    sole = Agent.founded(
        agent_id="only",
        genome=genome,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
        learning=LearningConfig(),
    )
    windows = _warmed(sole.state, leaning=LEANS_LONG)
    instrument = _instrument(windows[0])

    wished = sole.intend(windows[WARMED_UP_AFTER], instrument)

    assert wished.agent_id == "only"
    assert wished.as_of == windows[WARMED_UP_AFTER].as_of
    assert wished.target_exposure == sole.state.intended("only")
    assert sole.genome() == genome
    assert dict(sole.learned_weights()) == dict(sole.state.learned_weights("only"))


def test_what_each_agent_wanted_is_readable_apart_from_what_it_holds() -> None:
    """A wish and a position are different numbers and are never the same field."""
    state = _founded()
    windows = _warmed(state, leaning=LEANS_LONG)
    window = windows[WARMED_UP_AFTER]

    step = state.step(window, _instrument(window))

    assert step.held == (0.0,)
    assert state.intended("only") == FULL_EXPOSURE
    assert state.realised("only") == 0.0

    state.hold("only", 0.25)
    assert state.realised("only") == 0.25
    assert state.intended("only") == FULL_EXPOSURE
    assert state.step(windows[WARMED_UP_AFTER + 1], _instrument(window)).held == (0.25,)


def test_the_wishes_can_be_read_as_the_value_type_the_ecology_speaks() -> None:
    """The step is arrays; a caller that wants intents asks for them."""
    state = _founded()
    windows = _warmed(state, leaning=LEANS_LONG)
    window = windows[WARMED_UP_AFTER]

    intents = state.step(window, _instrument(window)).intents()

    assert [intent.agent_id for intent in intents] == ["only"]
    assert intents[0].as_of == window.as_of
    assert intents[0].instrument == _instrument(window)
    assert intents[0].target_exposure == FULL_EXPOSURE


def test_a_reading_that_only_reaches_the_entry_gene_does_not_clear_it() -> None:
    """The gene is a bar to be cleared, and clearing it is strict.

    A learned weight of nothing produces a reading of exactly nothing whatever
    the view says, which is the one reading whose relation to a gene of zero
    can be stated without arithmetic. Either side of the boundary is asserted,
    because a rule comparing with ``>=`` passes the first of these on its own.
    """
    on_the_gene = _founded(_birth("only", entry=0.0))
    windows = _warmed(on_the_gene, leaning=0.0)
    instrument = _instrument(windows[0])
    assert on_the_gene.step(windows[WARMED_UP_AFTER], instrument).wanted == (0.0,)

    just_below = _founded(_birth("only", entry=-1e-9))
    _warmed(just_below, leaning=0.0)
    assert just_below.step(windows[WARMED_UP_AFTER], instrument).wanted == (
        FULL_EXPOSURE,
    )


def test_an_instrument_the_decision_point_carries_no_view_for_reads_as_nothing() -> (
    None
):
    """An absent view is a reading of nothing, never a reach for one elsewhere."""
    window = _window(index=5)
    state = _founded(_birth("only", entry=-1.0))

    step = state.step(window, "AN_INSTRUMENT_THIS_POINT_DOES_NOT_CARRY")

    assert step.wanted == (FULL_EXPOSURE,)


def test_an_agent_is_not_stepped_under_a_vocabulary_it_was_not_born_under() -> None:
    """A gene means what its vocabulary says; under another it means nothing."""
    state = _founded()
    elsewhere = replace(_window(), feature_schema_version="another-vocabulary-1")

    with pytest.raises(AgentBornUnderAnotherVocabulary, match="only"):
        state.step(elsewhere, _instrument(elsewhere))


def test_the_outcome_history_keeps_its_ceiling_and_drops_the_oldest() -> None:
    """A bounded history is a loss, stated: what is retained is what is handed back."""
    state = _founded(history_capacity=3)

    for outcome in (1.0, 2.0, 3.0):
        state.record_outcome("only", outcome)
    assert state.history("only") == (1.0, 2.0, 3.0)

    state.record_outcome("only", 4.0)
    assert state.history("only") == (2.0, 3.0, 4.0)


def test_what_an_agent_can_learn_is_declared_at_its_birth() -> None:
    """A learned name arriving later is a value nothing chose to give it."""
    state = _founded()

    state.learn("only", {OBSERVED: 3.0})
    assert dict(state.learned_state("only"))[OBSERVED] == 3.0

    with pytest.raises(KeyError, match="not born able to learn"):
        state.learn("only", {"something_else": 1.0})


def test_an_agent_nobody_is_named_after_is_refused_by_name() -> None:
    """A lookup that missed says who is alive rather than raising a bare key."""
    state = _founded()

    with pytest.raises(KeyError, match="is not alive in this population"):
        state.genome("nobody")


def test_the_population_reports_the_shape_it_is_actually_held_in() -> None:
    """The claim that a population is arrays is readable rather than asserted in prose."""
    state = _founded(history_capacity=7)

    storage = state.storage()

    assert storage.agents == 1
    assert storage.gene_columns == 6
    assert storage.feature_columns == 2
    # Exactly the columns the update declares, and nothing a birth added: a
    # weight, a drift and an energy per feature, the standardiser's two moments
    # per feature, and the one mass they are divided by.
    assert storage.learned_columns == 5 * 2 + 1
    assert storage.history_capacity == 7
    assert storage.dtype == "float64"
    assert storage.contiguous


def test_ten_thousand_agents_over_sixty_four_features_step_as_one_product() -> None:
    """The size the ecology is meant to run at, stepped as arithmetic not as a loop.

    Both branches of the wish are exercised, and the entry genes are put far
    enough either side of anything the reading can reach that which branch each
    row takes is decided by the gene rather than by the size of a scaled
    feature. What is under test here is that ten thousand rows are answered as
    one product, not what any of them concluded.
    """
    agents = 10_000
    features = tuple(f"feature_{index:02d}" for index in range(64))
    genes = {
        **{f"{MASK_PREFIX}{name}": 1.0 for name in features},
        **{f"{WEIGHT_PREFIX}{name}": 0.0 for name in features},
        ENTRY_THRESHOLD: 0.0,
        FORGETTING: 0.97,
    }
    state = PopulationState.founded(
        tuple(
            AgentBirth(
                agent_id=f"agent-{index}",
                genome=Genome(
                    genes={
                        **genes,
                        ENTRY_THRESHOLD: -BEYOND_ANY_READING
                        if index % 2 == 0
                        else BEYOND_ANY_READING,
                    },
                    schema_version=GENE_SCHEMA_VERSION,
                ),
                learned_state={},
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                model_version=MODEL_VERSION,
            )
            for index in range(agents)
        ),
        learning=LearningConfig(),
    )
    window = BarWindow(
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        segment_id="one",
        segment_role="train",
        bars={},
        features={"ANY": dict.fromkeys(features, 1.0 / 64.0)},
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )

    storage = state.storage()
    step = state.step(window, "ANY")

    assert storage.agents == agents
    assert storage.feature_columns == 64
    assert storage.contiguous
    assert np.array_equal(
        np.asarray(step.wanted),
        np.where(np.arange(agents) % 2 == 0, FULL_EXPOSURE, 0.0),
    )


def test_a_founder_has_no_parents_and_a_birth_instant_is_stated_in_utc() -> None:
    """Lineage is not heritable, and a birth instant that names no zone is refused."""
    assert _founded().lineage("only") == Lineage(
        agent_id="only", parents=(), born_at=None
    )

    elsewhere = datetime(2000, 1, 1, 12, tzinfo=timezone(timedelta(hours=5)))
    born = Lineage(agent_id="child", parents=("only",), born_at=elsewhere)
    assert born.born_at == datetime(2000, 1, 1, 7, tzinfo=UTC)

    with pytest.raises(ValueError, match="timezone-aware"):
        Lineage(agent_id="child", parents=(), born_at=datetime(2000, 1, 1))  # noqa: DTZ001


def test_what_an_agent_was_born_under_is_recorded_per_agent() -> None:
    """Both versions, kept beside the agent rather than assumed for the run."""
    versions = _founded().versions("only")

    assert versions.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert versions.model_version == MODEL_VERSION


def _snapshot_of(state: PopulationState) -> PopulationSnapshot:
    """The population as it stands, for a test that is about to spoil it."""
    return state.snapshot()


def _held(state: PopulationState, **changes: object) -> AgentSnapshot:
    """The population's single agent, with ``changes`` applied to it."""
    return replace(_snapshot_of(state).agents[0], **changes)


@pytest.mark.parametrize(
    ("changes", "refused"),
    [
        (
            {"genome": Genome(genes=_genes(), schema_version="another-schema-1")},
            "one population reads one gene vocabulary",
        ),
        (
            {
                "genome": Genome(
                    genes={**_genes(), "an_extra_gene": 1.0},
                    schema_version=GENE_SCHEMA_VERSION,
                )
            },
            "has no column",
        ),
        ({"learned_state": {"another_thing": 0.0}}, "what is learnable is a column"),
        ({"history": (1.0, 2.0, 3.0, 4.0, 5.0)}, "being restored under a ceiling"),
    ],
)
def test_an_agent_that_does_not_fit_the_layout_is_refused(
    changes: dict[str, object], refused: str
) -> None:
    """A column means the same thing for everyone, or it is not a column."""
    state = _founded()
    second = replace(_held(state, **changes), agent_id="second")
    spoiled = PopulationSnapshot(
        agents=(_snapshot_of(state).agents[0], second), history_capacity=4
    )

    with pytest.raises(ValueError, match=refused):
        PopulationState.restored(spoiled)


def test_a_population_that_is_not_one_is_refused_rather_than_held() -> None:
    """Nobody alive, two agents with one name, and a history that retains nothing."""
    with pytest.raises(ValueError, match="not a population"):
        PopulationState.founded(())

    with pytest.raises(ValueError, match="share an identity"):
        PopulationState.founded((_birth("same"), _birth("same")))

    with pytest.raises(ValueError, match="retains nothing"):
        PopulationState.founded(
            (_birth("only"),), population=PopulationConfig(history_capacity=0)
        )

    # The ceiling is checked twice over, and both are needed: the configuration
    # refuses a run that declares nothing is kept, and the store refuses a
    # snapshot that says so, because a snapshot is a value anybody can build
    # and a restore does not pass through the configuration that made one.
    with pytest.raises(ValueError, match="retains nothing"):
        PopulationState.restored(replace(_snapshot_of(_founded()), history_capacity=0))


def test_a_genome_the_layout_cannot_be_read_off_is_refused() -> None:
    """A switched-on feature with no weight, and no entry gene, are both defects."""
    unweighted = dict(_genes())
    del unweighted[f"{WEIGHT_PREFIX}{STEP}"]
    with pytest.raises(ValueError, match="no weight"):
        PopulationState.founded(
            (
                AgentBirth(
                    agent_id="only",
                    genome=Genome(genes=unweighted, schema_version=GENE_SCHEMA_VERSION),
                    learned_state={},
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    model_version=MODEL_VERSION,
                ),
            )
        )

    entryless = dict(_genes())
    del entryless[ENTRY_THRESHOLD]
    with pytest.raises(ValueError, match=ENTRY_THRESHOLD):
        PopulationState.founded(
            (
                AgentBirth(
                    agent_id="only",
                    genome=Genome(genes=entryless, schema_version=GENE_SCHEMA_VERSION),
                    learned_state={},
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    model_version=MODEL_VERSION,
                ),
            )
        )
