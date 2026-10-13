"""What the array-genome population does beyond the contract every population meets.

The port's own promises are asserted once, in `tests/support/contract/`, against
this adapter and the stand-in together. What is left for here is what is
particular to this one: the two descriptors it reads off the store, the refusal
of a birth with nobody to inherit from, that births can carry a prior outside
the span its founders' priors cover, what it refuses to hand to the platform's
archive, and how it chooses parents when some or all of the living carry no
figure.

**The declared four axes.** Built with a descriptor declaration and its bins,
the record is the platform's archive on the four ecology axes: each agent is
described by the ecology's descriptor module over what the store holds, an entry
is binned on those axes by the archive's own bounds interface and reported back
through the port's coverage, and an entry in any other vocabulary, the
provisional one included, is refused. The same population is held to the whole
port contract over agents walked on the declared stream S1 until each holds more
than the descriptor window. Built without a declaration, the record keeps the
two axes above unchanged. No population, declared or not, is built under the
provisional vocabulary.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import descriptors, genome_schema
from liq.evolution.ecology.adapters import (
    ArrayGenomePopulation,
    NullExecutionSimulator,
    NullRiskSizer,
)
from liq.evolution.ecology.adapters.array_genome import (
    EXPOSURE_WANTED,
    FEATURES_READ,
    design_founders,
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
from liq.evolution.ecology.agent_genes import MUTATION_SCALE
from liq.evolution.ecology.config import EcologyConfig, LearningConfig, PopulationConfig
from liq.evolution.ecology.learning import WEIGHT_PREFIX as LEARNED_WEIGHT_PREFIX
from liq.evolution.ecology.types import ArchiveEntry, Descriptor, Genome
from liq.gp import ConfigurationError, GPConfig
from tests.support import turnover_stream as streams
from tests.support.contract import test_agent_population as contract

GENE_SCHEMA_VERSION = genome_schema.SCHEMA_VERSION
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


#: The interval the founders' priors span, and how many founders span it,
#: evenly and end to end. Declared genes, not measurements.
FOUNDER_PRIOR_SPAN = (-0.5, 0.5)
FOUNDERS = 4

#: How many births the prior's mobility is asked over.
MOBILITY_BIRTHS = 1_000


def _founders_spanning_the_prior() -> ArrayGenomePopulation:
    """Founders whose ``weight.`` priors cover :data:`FOUNDER_PRIOR_SPAN` exactly."""
    low, high = FOUNDER_PRIOR_SPAN
    founders = []
    for index in range(FOUNDERS):
        prior = low + (high - low) * index / (FOUNDERS - 1)
        genes = _two_features_off()
        genes.update({f"{WEIGHT_PREFIX}{name}": prior for name in (LEVEL, STEP)})
        founders.append(_birth(f"founder-{index}", genes=genes))
    return _population(*founders)


def test_births_can_move_a_prior_outside_the_founders_span() -> None:
    """Variation can move the prior: some birth leaves the span founders cover.

    Selection-free: each birth's two parents are drawn uniformly from the
    founders by the birth's own seed, so nothing but variation can carry a
    prior anywhere.
    """
    population = _founders_spanning_the_prior()
    founders = population.agent_ids()
    low, high = FOUNDER_PRIOR_SPAN
    left = 0
    for seed in range(MOBILITY_BIRTHS):
        parents = tuple(random.Random(seed).choices(founders, k=2))
        genes = population.spawn(parents=parents, seed=seed).genome.genes
        left += any(
            not low <= genes[f"{WEIGHT_PREFIX}{name}"] <= high for name in (LEVEL, STEP)
        )

    assert left > 0, (
        f"0 of {MOBILITY_BIRTHS} births carried a prior outside {FOUNDER_PRIOR_SPAN}"
    )


def test_a_birth_of_nobody_is_refused_rather_than_invented() -> None:
    """An offspring with no parents would be an initialisation, not an inheritance."""
    population = _population(_reading_two_features("only", level_on=1.0))

    with pytest.raises(ValueError, match="at least one parent"):
        population.spawn(parents=(), seed=17)


def test_a_birth_of_more_than_two_is_refused() -> None:
    """The declared recombination takes one or two parents."""
    population = _founders_spanning_the_prior()

    with pytest.raises(ValueError, match="one or two parents"):
        population.spawn(parents=population.agent_ids()[:3], seed=17)


def test_a_gene_with_no_declared_range_is_refused() -> None:
    """No operator can vary a gene inside a range nobody declared."""
    genes = _two_features_off()
    genes["undeclared"] = 0.0
    population = _population(_birth("odd", genes=genes))

    with pytest.raises(ValueError, match="no declared range"):
        population.spawn(parents=("odd",), seed=17)


def test_a_parent_without_a_scale_gene_is_read_at_the_founding_scale() -> None:
    """The scale in force is reported; the child's vocabulary stays its parents'."""
    population = _population(_birth("plain", genes=_two_features_off()))

    offspring = population.spawn(parents=("plain",), seed=17)

    assert MUTATION_SCALE not in offspring.genome.genes
    assert offspring.mutation_scale is not None
    assert MUTATION_SCALE not in offspring.boundary_firings


def test_a_mask_travels_with_the_nearer_parents_weight() -> None:
    """Each mask comes from the parent whose weight the child's is nearer.

    The parents' masks and weights are opposite, so wherever the child's weight
    is a parent's unmutated value it names the parent its mask should have come
    from. Mutation flips a mask now and then, so the rule is held as a rate: it
    explains most masks, and taking every mask from the first parent explains
    far fewer.
    """
    first, second = _two_features_off(), _two_features_off()
    for name in (LEVEL, STEP):
        second[f"{MASK_PREFIX}{name}"] = 1.0
        first[f"{WEIGHT_PREFIX}{name}"] = -10.0
        second[f"{WEIGHT_PREFIX}{name}"] = 10.0
    population = _population(
        _birth("first", genes=first), _birth("second", genes=second)
    )
    nearer = from_first = checked = 0
    for seed in range(400):
        genes = population.spawn(parents=("first", "second"), seed=seed).genome.genes
        for name in (LEVEL, STEP):
            weight = genes[f"{WEIGHT_PREFIX}{name}"]
            if abs(weight) != 10.0:
                continue
            mask = genes[f"{MASK_PREFIX}{name}"]
            checked += 1
            nearer += mask == (1.0 if weight > 0.0 else 0.0)
            from_first += mask == 0.0

    assert checked > 400
    assert nearer / checked > 0.7
    assert nearer - from_first > 0.2 * checked


def test_a_parent_carrying_a_gene_outside_its_range_is_refused() -> None:
    """A founder declared past a bound is refused at its first birth, not carried.

    Nothing is varied around it: the operators work only inside the declared
    ranges, so a parent outside one is named — gene, value and range — and no
    offspring is returned.
    """
    learning = LearningConfig()
    outside = 0.9 * learning.weight_norm_bound
    genes = _two_features_off()
    genes[f"{WEIGHT_PREFIX}{LEVEL}"] = outside
    population = _population(_birth("heavy", genes=genes))
    bound = genome_schema.weight_bound(learning, 2)
    assert outside > bound

    with pytest.raises(ValueError, match="outside its declared range") as refused:
        population.spawn(parents=("heavy",), seed=17)
    message = str(refused.value)
    assert f"{WEIGHT_PREFIX}{LEVEL}" in message
    assert repr(outside) in message
    assert repr(bound) in message


def test_a_second_parent_outside_its_range_is_refused_too() -> None:
    """Either parent is held to the ranges, not only the first."""
    learning = LearningConfig()
    inside = _two_features_off()
    outside = _two_features_off()
    outside[ENTRY_THRESHOLD] = -2.0 * genome_schema.entry_bound(learning)
    population = _population(
        _birth("inside", genes=inside), _birth("outside", genes=outside)
    )

    with pytest.raises(ValueError, match=ENTRY_THRESHOLD):
        population.spawn(parents=("inside", "outside"), seed=17)


# ---------------------------------------------------------------------------
# The genome schema a birth is held to.
# ---------------------------------------------------------------------------

#: A genome schema this adapter does not declare ranges for.
ANOTHER_GENOME_SCHEMA = "another-genome-schema"


def _designed(*, schema_version: str) -> ArrayGenomePopulation:
    """Four designed founders, their genomes labelled ``schema_version``.

    The genes are the design's, so every one lies inside the declared ranges;
    only the label differs between the two populations built from it.
    """
    births = [
        AgentBirth(
            agent_id=birth.agent_id,
            genome=Genome(genes=birth.genome.genes, schema_version=schema_version),
            learned_state=birth.learned_state,
            feature_schema_version=birth.feature_schema_version,
            model_version=birth.model_version,
        )
        for birth in design_founders(
            features=(LEVEL, STEP),
            learning=LearningConfig(),
            count=FOUNDERS,
            seed=3,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_version=MODEL_VERSION,
        )
    ]
    return _population(*births)


@pytest.mark.parametrize("parents", [1, 2])
def test_a_parent_under_another_genome_schema_is_refused(parents: int) -> None:
    """A genome is varied only inside the ranges its own version declares.

    The adapter holds every gene to :data:`genome_schema.SCHEMA_VERSION`'s
    ranges, so a parent labelled under another declaration is refused before
    anything is varied, and the refusal names the parent, the version it
    carries and the one it would have been held to.
    """
    population = _designed(schema_version=ANOTHER_GENOME_SCHEMA)
    chosen = population.agent_ids()[:parents]

    with pytest.raises(ValueError) as refused:
        population.spawn(parents=chosen, seed=17)
    message = str(refused.value)
    assert repr(chosen[0]) in message
    assert repr(ANOTHER_GENOME_SCHEMA) in message
    assert repr(genome_schema.SCHEMA_VERSION) in message


@pytest.mark.parametrize("parents", [1, 2])
def test_a_child_carries_the_genome_schema_it_was_held_to(parents: int) -> None:
    """The child is labelled with the declaration whose ranges it was varied in."""
    population = _designed(schema_version=genome_schema.SCHEMA_VERSION)
    chosen = population.agent_ids()[:parents]

    offspring = population.spawn(parents=chosen, seed=17)

    assert offspring.genome.schema_version == genome_schema.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# What is handed to the platform's archive, and what is refused.
# ---------------------------------------------------------------------------

#: When the probe entries below were offered. Any UTC instant; nothing reads it.
OFFERED_AT = datetime(2000, 1, 1, tzinfo=UTC)


def _offered(
    population: ArrayGenomePopulation,
    agent_id: str,
    *,
    values: dict[str, float] | None = None,
    schema_version: str = DESCRIPTOR_SCHEMA_VERSION,
    objectives: dict[str, float] | None = None,
) -> ArchiveEntry:
    """An entry for ``agent_id``, with its descriptor or figures replaced."""
    held = population.descriptor(agent_id)
    return ArchiveEntry(
        agent_id=agent_id,
        genome=population.genome(agent_id),
        descriptor=Descriptor(
            values=MappingProxyType(dict(held.values) if values is None else values),
            schema_version=schema_version,
        ),
        objectives=MappingProxyType(
            {"score": 0.0} if objectives is None else objectives
        ),
        recorded_at=OFFERED_AT,
    )


@pytest.mark.parametrize(
    "coordinate", [math.nextafter(1.0, 2.0), -0.25, math.nan, math.inf]
)
def test_a_coordinate_the_archive_cannot_bin_is_an_exclusion(
    coordinate: float,
) -> None:
    """Outside the unit interval or not a number: excluded, never clipped or binned."""
    population = _population(_reading_two_features("only", level_on=1.0))
    entry = _offered(
        population,
        "only",
        values={FEATURES_READ: coordinate, EXPOSURE_WANTED: 0.0},
    )

    assert population.record(entry) is False

    assert population.recorded() == ()
    assert population.excluded() == (entry,)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "another-vocabulary"}, "cannot be placed"),
        ({"values": {FEATURES_READ: 0.5}}, "described on exactly"),
        ({"objectives": {"score": 0.0, "other": 1.0}}, "direction"),
    ],
)
def test_an_entry_the_record_cannot_read_is_refused_as_a_defect(
    change: dict[str, object], message: str
) -> None:
    """Another vocabulary, other axes or another figure count is the caller's error."""
    population = _population(_reading_two_features("only", level_on=1.0))
    entry = _offered(population, "only", **change)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        population.record(entry)

    assert population.recorded() == ()
    assert population.excluded() == ()


def test_the_archive_has_one_cell_per_value_each_descriptor_takes() -> None:
    """Agents reading different shares of their features never share a cell."""
    population = _population(
        _reading_two_features("greedy", level_on=1.0),
        _reading_two_features("sparing", level_on=0.0),
        _birth("blank-but-for-two", genes=_two_features_off()),
    )

    for agent_id in population.agent_ids():
        assert population.record(_offered(population, agent_id))

    assert population.archive.filled_bins == len(population.agent_ids())
    assert len(population.recorded()) == len(population.agent_ids())


def _two_features_off() -> dict[str, float]:
    """The two-feature genome reading neither feature."""
    return {
        f"{MASK_PREFIX}{LEVEL}": 0.0,
        f"{WEIGHT_PREFIX}{LEVEL}": 0.0,
        f"{MASK_PREFIX}{STEP}": 0.0,
        f"{WEIGHT_PREFIX}{STEP}": 0.0,
        ENTRY_THRESHOLD: 0.0,
        FORGETTING_FACTOR: FORGETS_AT,
    }


# ---------------------------------------------------------------------------
# Parents when some, or all, of the living carry no figure.
# ---------------------------------------------------------------------------

#: Seeds a selection is asked under, so "never chosen" is not one draw's luck.
SELECTION_SEEDS = tuple(range(20))

#: How many parents each case asks for: more than the living, so a selection
#: has to choose some agents more than once.
PARENTS_WANTED = 25

#: How many living agents carry no figure in the ordinary case.
UNASSESSED = 3


def _living(count: int) -> ArrayGenomePopulation:
    """A population of ``count`` agents differing in nothing a selection reads."""
    return _population(
        *(
            _reading_two_features(f"agent-{index}", level_on=1.0)
            for index in range(count)
        )
    )


def test_the_fewest_candidates_is_the_platform_selections_own_floor() -> None:
    """The declared floor is the smallest population the platform can configure."""
    population = _living(1)
    floor = population.minimum_candidates

    assert isinstance(floor, int)
    GPConfig(population_size=floor, elitism_count=floor - 1, tournament_size=2)
    with pytest.raises(ConfigurationError, match="population_size"):
        GPConfig(population_size=floor - 1, elitism_count=floor - 2, tournament_size=2)


def test_an_agent_without_a_figure_is_never_chosen() -> None:
    """Only agents carrying a figure are ranked, so only they are returned."""
    population = _living(_living(1).minimum_candidates + UNASSESSED)
    agent_ids = population.agent_ids()
    unassessed = set(agent_ids[:UNASSESSED])
    fitness = {
        agent_id: None if agent_id in unassessed else float(position)
        for position, agent_id in enumerate(agent_ids)
    }

    for seed in SELECTION_SEEDS:
        parents = population.select_parents(
            fitness=fitness, count=PARENTS_WANTED, seed=seed
        )
        assert len(parents) == PARENTS_WANTED
        assert not unassessed & set(parents)


def test_with_no_figure_at_all_parents_are_drawn_from_the_living() -> None:
    """A blind selection draws from every living agent, reproducibly for the seed.

    Below the platform's floor too: no ordering is configured when nothing is
    ranked, so a blind draw needs no configuration that could refuse it.
    """
    for living in (3, _living(1).minimum_candidates + UNASSESSED):
        population = _living(living)
        agent_ids = population.agent_ids()
        blind = dict.fromkeys(agent_ids)

        drawn = {
            seed: population.select_parents(
                fitness=blind, count=PARENTS_WANTED, seed=seed
            )
            for seed in SELECTION_SEEDS
        }

        for seed, parents in drawn.items():
            assert len(parents) == PARENTS_WANTED
            assert set(parents) <= set(agent_ids)
            assert (
                population.select_parents(
                    fitness=blind, count=PARENTS_WANTED, seed=seed
                )
                == parents
            )
        assert len(set(drawn.values())) > 1, "every seed drew the same parents"
        assert set().union(*drawn.values()) == set(agent_ids)


def test_some_figures_but_fewer_than_the_floor_is_refused() -> None:
    """Too few to rank, too many to be blind: the caller must decide."""
    population = _living(_living(1).minimum_candidates + UNASSESSED)
    agent_ids = population.agent_ids()
    few = population.minimum_candidates - 1
    fitness = {
        agent_id: float(position) if position < few else None
        for position, agent_id in enumerate(agent_ids)
    }

    with pytest.raises(ValueError):
        population.select_parents(fitness=fitness, count=PARENTS_WANTED, seed=17)


# ---------------------------------------------------------------------------
# Founders designed across the declared ranges.
# ---------------------------------------------------------------------------


def test_designed_founders_are_admitted_and_start_from_their_priors() -> None:
    """Bits for masks, the founding scale, nothing learned, and ``founded()`` takes them."""
    learning = LearningConfig()
    births = design_founders(
        features=(LEVEL, STEP),
        learning=learning,
        count=12,
        seed=3,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )
    state = PopulationState.founded(births, learning=learning)

    assert state.agent_ids() == tuple(f"founder-{index:02d}" for index in range(12))
    for birth in births:
        genes = birth.genome.genes
        assert {genes[f"{MASK_PREFIX}{name}"] for name in (LEVEL, STEP)} <= {0.0, 1.0}
        assert genes[MUTATION_SCALE] == genome_schema.FOUNDING_SCALE
        assert birth.learned_state == {}
        assert birth.genome.schema_version == genome_schema.SCHEMA_VERSION
        assert birth.feature_schema_version == FEATURE_SCHEMA_VERSION
        assert birth.model_version == MODEL_VERSION
        prior = state.learned_weights(birth.agent_id)
        assert prior[LEVEL] == genes[f"{WEIGHT_PREFIX}{LEVEL}"]


def test_a_design_is_reproducible_for_its_seed_and_moves_with_it() -> None:
    """The same seed designs the same founders; another seed designs others."""

    def genes(seed: int) -> list[dict[str, float]]:
        return [
            dict(birth.genome.genes)
            for birth in design_founders(
                features=(LEVEL,), learning=LearningConfig(), count=4, seed=seed
            )
        ]

    assert genes(5) == genes(5)
    assert genes(5) != genes(6)


def test_a_design_of_nobody_is_refused() -> None:
    """There is no initial population of no founders."""
    with pytest.raises(ValueError):
        design_founders(features=(LEVEL,), learning=LearningConfig(), count=0, seed=0)


# ---------------------------------------------------------------------------
# The declared four axes
# ---------------------------------------------------------------------------

#: The descriptor window: the fitness window a run would tie it to.
DESCRIPTOR_WINDOW = 8

#: How many decision points of S1 a declared population is walked over before
#: it is handed out: past the window, so every founder holds more outcomes than
#: its descriptor reads.
WALKED_FOR = 3 * DESCRIPTOR_WINDOW

#: The bins of the holding horizon, turnover and beta axes, as the
#: descriptor-parameter decision record states them
#: (``liq-docs/plans/agent-ecology-adr-descriptor-parameters.md``).
DECIDED_BINS = (3, 16, 9)

#: The one declared asset class of the one traded instrument.
ASSET_CLASS = "equity"

#: The mandate the declaration derives its caps from. The gross bound is the
#: null sizer's own, so the accounts were walked under the bound the axes are
#: normalised by.
DECLARED_MANDATE = descriptors.Mandate(
    max_position=1.0,
    max_gross_exposure=NullRiskSizer().max_gross_exposure,
    max_weight_delta=2.0,
    max_turnover=1.0,
)


def _declaration() -> descriptors.DescriptorDeclaration:
    return descriptors.DescriptorDeclaration(
        window=DESCRIPTOR_WINDOW,
        history_capacity=streams.HISTORY_CAPACITY,
        asset_classes=(ASSET_CLASS,),
        universe={streams.INSTRUMENT: ASSET_CLASS},
        mandate=DECLARED_MANDATE,
    )


def _founded_on_s1(agent_count: int) -> PopulationState:
    return PopulationState.founded(
        streams.founders(agent_count),
        population=PopulationConfig(history_capacity=streams.HISTORY_CAPACITY),
        learning=LearningConfig(),
    )


def _declared(state: PopulationState) -> ArrayGenomePopulation:
    return ArrayGenomePopulation(
        state=state,
        descriptor_schema_version=descriptors.VOCABULARY,
        descriptors=_declaration(),
        descriptor_bins=DECIDED_BINS,
    )


def _walked(state: PopulationState) -> PopulationState:
    """``state`` walked over S1 with no turnover, its histories written by its accounts."""
    accounts = ecology.population_accounts.PopulationAccounts(
        state,
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=1.0,
    )
    accounts.run(
        streams.Watched(streams.priced(count=WALKED_FOR), state),
        EcologyConfig(run_id="declared-descriptors"),
    )
    return state


def a_declared_array_genome_population(
    agent_count: int = contract.ARRAY_AGENTS,
) -> ArrayGenomePopulation:
    """``agent_count`` founders walked on S1, held on the declared four axes."""
    population = _declared(_walked(_founded_on_s1(agent_count)))
    reading = descriptors.read(population.state, _declaration())
    assert reading.undescribed == {}, "a walked founder is short of the window"
    assert all(not each.excluded for each in reading.described.values()), (
        "a walked founder carries an axis the record cannot bin, so the contract "
        "would be asked about an exclusion rather than a description"
    )
    return population


def an_undescribed_declared_population() -> tuple[ArrayGenomePopulation, str]:
    """A declared population straight from founding: nobody holds an outcome yet."""
    population = _declared(_founded_on_s1(contract.ARRAY_AGENTS))
    return population, population.agent_ids()[0]


class TestDeclaredArrayGenomePopulation(contract.AgentPopulationContract):
    """The array-genome population on the declared four axes, held to the whole contract."""

    adapter_factory = staticmethod(a_declared_array_genome_population)
    undescribed_factory = staticmethod(an_undescribed_declared_population)


def _bin(value: float, cells: int) -> int:
    """The cell a unit-interval value falls in among ``cells`` equal ones, the top edge in the last."""
    return min(int(value * cells), cells - 1)


def test_the_four_axes_are_binned_and_reported_through_the_platform_archive() -> None:
    """Each entry lands in the cell its four values name, and coverage says so.

    The archive is the platform's, built on the declared axes, bounds and bins;
    nothing here bins anything itself except to state the expected cell.
    """
    population = a_declared_array_genome_population(agent_count=14)
    bins = _declaration().archive_bins(
        holding_horizon=DECIDED_BINS[0], turnover=DECIDED_BINS[1], beta=DECIDED_BINS[2]
    )
    assert population.archive.n_dims == len(descriptors.AXES)
    assert population.archive.bins_per_dim == bins
    assert population.archive.descriptor_bounds == ((0.0, 1.0),) * len(bins)

    for position, agent_id in enumerate(population.agent_ids()):
        population.record(
            _offered(
                population,
                agent_id,
                schema_version=descriptors.VOCABULARY,
                objectives={"deflated": float(position)},
            )
        )

    held = population.recorded()
    assert len(held) >= 2, "the walk described too few distinct behaviours to bin"
    expected = [
        tuple(
            _bin(entry.descriptor.values[axis], cells)
            for axis, cells in zip(descriptors.AXES, bins, strict=True)
        )
        for entry in held
    ]
    placed = [tuple(raw["bin"]) for raw in population.archive.to_dict()["entries"]]
    assert placed == expected
    coverage = population.coverage()
    assert coverage.axes == descriptors.AXES
    assert coverage.total == math.prod(bins)
    assert coverage.filled == len(set(expected))
    assert coverage.histograms == tuple(
        tuple(sum(1 for cell in expected if cell[dim] == index) for index in range(n))
        for dim, n in enumerate(bins)
    )


def test_the_declared_descriptor_is_the_descriptor_modules_reading() -> None:
    """Four axes in their declared order, under the declared vocabulary, nothing added."""
    population = a_declared_array_genome_population()
    for agent_id in population.agent_ids():
        held = population.descriptor(agent_id)
        reading = descriptors.describe(population.state, _declaration(), agent_id)
        assert tuple(held.values) == descriptors.AXES
        assert dict(held.values) == dict(reading.values)
        assert held.schema_version == descriptors.VOCABULARY


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": descriptors.PROVISIONAL_VOCABULARY}, "cannot be placed"),
        ({"schema_version": DESCRIPTOR_SCHEMA_VERSION}, "cannot be placed"),
        (
            {"values": {FEATURES_READ: 0.5, EXPOSURE_WANTED: 0.0}},
            "described on exactly",
        ),
    ],
)
def test_the_declared_record_refuses_an_entry_in_another_vocabulary(
    change: dict[str, object], message: str
) -> None:
    """No coordinate of the provisional vocabulary, or any other, enters the record."""
    population = a_declared_array_genome_population()
    agent_id = population.agent_ids()[0]
    entry = _offered(
        population,
        agent_id,
        **{"schema_version": descriptors.VOCABULARY, **change},  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        population.record(entry)

    assert population.recorded() == ()
    assert population.excluded() == ()


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"descriptors": None, "descriptor_bins": DECIDED_BINS}, "both or neither"),
        ({"descriptor_bins": None}, "both or neither"),
        ({"descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION}, "vocabulary"),
    ],
)
def test_a_declaration_is_whole_and_names_the_vocabulary_it_computes(
    declared: dict[str, object], message: str
) -> None:
    """A declaration without its bins, bins without one, or a second vocabulary is refused."""
    arguments: dict[str, object] = {
        "state": _founded_on_s1(contract.ARRAY_AGENTS),
        "descriptor_schema_version": descriptors.VOCABULARY,
        "descriptors": _declaration(),
        "descriptor_bins": DECIDED_BINS,
        **declared,
    }
    with pytest.raises(ValueError, match=message):
        ArrayGenomePopulation(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("declared", [False, True], ids=["undeclared", "declared"])
def test_no_population_is_built_under_the_provisional_vocabulary(
    declared: bool,
) -> None:
    """The provisional vocabulary is refused at construction, with or without a declaration.

    One vocabulary per population, and the provisional one is no longer any
    population's: a record built under it would hold coordinates that are never
    carried into a record of the declared axes.
    """
    arguments: dict[str, object] = {
        "state": _founded_on_s1(contract.ARRAY_AGENTS),
        "descriptor_schema_version": descriptors.PROVISIONAL_VOCABULARY,
    }
    if declared:
        arguments.update(descriptors=_declaration(), descriptor_bins=DECIDED_BINS)
    with pytest.raises(ValueError, match="is the provisional vocabulary"):
        ArrayGenomePopulation(**arguments)  # type: ignore[arg-type]


def test_an_undeclared_record_keeps_its_two_axes() -> None:
    """Built with no declaration, the record and its coverage stay on the two axes."""
    population = _population(_reading_two_features("only", level_on=1.0))
    assert population.archive.n_dims == 2
    assert population.coverage().axes == (FEATURES_READ, EXPOSURE_WANTED)
    assert population.descriptor("only").schema_version == DESCRIPTOR_SCHEMA_VERSION
