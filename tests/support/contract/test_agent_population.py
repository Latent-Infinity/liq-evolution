"""The contract every population is run against.

Tier 2. Two separations are the whole of what is checked: an agent's heritable
part is reachable apart from what it has learned, so a birth can carry one
without the other; and the diversity record decides what it keeps, so the
ecology offers an entry and reads the answer rather than placing or evicting
anything itself.

Which variation a birth applies is the adapter's business. What is contracted is
that a birth produces one offspring genome in the parents' own vocabulary, that
the same parents and seed produce it again, that spawning does not by itself
change who is alive, and that the offspring is not a copy: it differs from each
parent in at least one object gene, whether it had one parent, two that differ
or the same agent twice. One copy is outside that claim: a copy of the second of
two different parents that carry a bit-equal ``weight.<f>`` at a feature whose
``mask.<f>`` differs, which the decided mask-inheritance tie rule produces. A
copy of the first parent, or a copy from parents with no such tie, is still a
breach, and two parents with no tie are asked for as a shape of their own. A birth also reports the variation it applied — the mutation
scale in force, and how often the boundary rule fired on each gene — because a
birth whose variation cannot be read back cannot be checked against the rule
that was declared for it.

Choosing parents is the population's too, through the platform's own selection.
What is contracted is that a call returns exactly the number of parents asked
for, drawn only from the living, reproducibly for a seed, from a fitness keyed by
the living population as it is at that call — never sized from a configuration
the population may since have drifted away from.

A figure may be absent: an agent the deflator could not assess is handed in as
``None``. Such an agent is never chosen while enough agents carry a figure, and
when none does parents are drawn uniformly from the living. Some figures, but
fewer than the population declares it can choose among, are refused, because
whether that trigger is blind is the caller's to decide. The population does not
report blindness back: the caller made it so by handing in no figure at all.

The record reports what it holds and what it would not bin. Its coverage — the
cells filled, the cells there are, and per axis how many entries sit in each
cell — is read from the record itself, so it agrees with the entries handed back.
An entry whose coordinate the record cannot bin is kept aside as an exclusion,
changing nothing else. A record persisted by a run restores into a fresh
population of that run as the same entries and the same coverage, and a payload
naming another run, another payload version, another vocabulary or other cells,
or one handed to a record that already holds something, is refused rather than
loaded. An agent a population cannot yet describe is refused with
:class:`~liq.evolution.ecology.descriptors.DescriptorUnavailable` and given no
coordinate; a registration that holds no such agent says so rather than passing
over nothing.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import ClassVar

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import (
    AgentId,
    AgentPopulation,
    ArchiveEntry,
    Descriptor,
    Genome,
    Offspring,
    genome_schema,
)
from liq.evolution.ecology.adapters import ArrayGenomePopulation, NullAgentPopulation
from liq.evolution.ecology.adapters.array_genome import design_founders
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    MASK_PREFIX,
    WEIGHT_PREFIX,
    AgentBirth,
    PopulationState,
)
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.descriptors import DescriptorUnavailable
from liq.evolution.ecology.errors import EcologyError

RECORDED_AT = datetime(2000, 1, 1, tzinfo=UTC)

# Declared placeholders, not anybody's population: three agents differing in one
# gene each, over two named features, so the contract below has more than one
# parent to breed from and more than one behaviour to describe.
ARRAY_AGENTS = 3
ARRAY_FEATURES = ("first_feature", "second_feature")
ARRAY_GENE_SCHEMA = genome_schema.SCHEMA_VERSION
ARRAY_FEATURE_SCHEMA = "contract-features-1"
ARRAY_MODEL = "contract-model-1"
ARRAY_DESCRIPTOR_SCHEMA = "contract-descriptors-1"
ARRAY_LEARNED = "observations"


def an_array_genome_population(
    agent_count: int = ARRAY_AGENTS,
) -> ArrayGenomePopulation:
    """A small array-genome population, built fresh, holding nothing yet."""
    span = max(agent_count - 1, 1)
    births = tuple(
        AgentBirth(
            agent_id=f"array-agent-{index}",
            genome=Genome(
                genes=MappingProxyType(
                    {
                        **{
                            f"{MASK_PREFIX}{name}": float(position <= index)
                            for position, name in enumerate(ARRAY_FEATURES)
                        },
                        **{f"{WEIGHT_PREFIX}{name}": 0.0 for name in ARRAY_FEATURES},
                        ENTRY_THRESHOLD: 1.0 - index / span,
                    }
                ),
                schema_version=ARRAY_GENE_SCHEMA,
            ),
            learned_state=MappingProxyType({ARRAY_LEARNED: float(index)}),
            feature_schema_version=ARRAY_FEATURE_SCHEMA,
            model_version=ARRAY_MODEL,
        )
        for index in range(agent_count)
    )
    return ArrayGenomePopulation(
        state=PopulationState.founded(births),
        descriptor_schema_version=ARRAY_DESCRIPTOR_SCHEMA,
    )


# Declared probe objectives, not measurements: one named figure offered twice
# unchanged and then higher, so a record is asked to keep, to refuse and to
# reconsider without being told which answer is right.
OFFERED_SCORES = (0.0, 0.0, 1.0)

# The size a run was configured for, and two living populations that have
# drifted either side of it — the shapes under which a selection sized from the
# configuration returns quietly too few parents, or indexes past the fitnesses
# it was handed (probe record §1.5, H-1 and H-2).
CONFIGURED_SIZE = 50
DRIFTED_SIZES = (47, 52)

# Seeds a "never chosen" case is asserted across, so one lucky draw cannot pass
# it. Declared draws, not tuned ones.
SELECTION_SEEDS = (3, 17, 29, 41, 53)

# How many living agents a fitness mapping leaves out, or names that are not
# alive, when a case asks whether selection notices.
MISMATCHED_BY = 2

# Births asked for per parent shape when the guard below asks whether variation
# was applied: the variation probe's own N, under which three platform operators
# were measured returning their input unchanged 200 times out of 200.
VARIATION_SEEDS = range(200)

# The genes that parameterise the variation itself rather than the agent. A
# difference confined to them is not a variation of the agent, so they are left
# out of the comparison. One: the self-adaptive mutation scale the operator
# decision selected (``liq-docs/plans/agent-ecology-adr-evolution-operators.md``,
# factor M). Held as a literal here so that reclassifying a gene is an edit to
# evidence.
STRATEGY_GENES: frozenset[str] = frozenset({"mutation_scale"})

# The ways a birth's parents can stand: one parent, recombined with itself; two
# parents whose genomes differ, as the adapter's own population holds them; the
# same agent offered twice, which a population converging under selection offers
# more and more often; and two parents that differ with no weight tie at a
# differing mask, so that no copy of either is admitted.
ONE_PARENT = "one_parent"
TWO_DISTINCT_PARENTS = "two_distinct_parents"
TWO_IDENTICAL_PARENTS = "two_identical_parents"
TWO_UNTIED_PARENTS = "two_untied_parents"
PARENT_SHAPES = (ONE_PARENT, TWO_DISTINCT_PARENTS, TWO_IDENTICAL_PARENTS)

# The shapes the guard asks about: every way the adapter's own population can
# stand, and the untied pair, which may be asked of another population.
GUARD_SHAPES = (*PARENT_SHAPES, TWO_UNTIED_PARENTS)


def declared_fitness(agent_ids: tuple[str, ...]) -> dict[str, float]:
    """A probe fitness keyed by ``agent_ids``: each agent's position, nothing more.

    A declared ordering, not a measurement — it gives selection something to
    prefer without claiming any agent earned it.
    """
    return {agent_id: float(position) for position, agent_id in enumerate(agent_ids)}


def figures_for_the_first(
    agent_ids: tuple[str, ...], carrying: int
) -> dict[str, float | None]:
    """The declared ordering for the first ``carrying`` agents; ``None`` after.

    The agents left without a figure are the ones a selection would prefer if it
    read their positions, so a case that chose one could not do so by accident.
    """
    return {
        agent_id: float(position) if position < carrying else None
        for position, agent_id in enumerate(agent_ids)
    }


def _bits(value: float) -> bytes:
    """The bytes of ``value`` as a double: equal exactly when the doubles are bit-equal."""
    return struct.pack("<d", value)


def weight_tie_at_a_differing_mask(first: Genome, second: Genome) -> bool:
    """Whether two genomes carry a bit-equal ``weight.<f>`` at some ``mask.<f>`` they differ in.

    Read from the genes by the mask and weight prefixes alone, so a genome
    carrying neither has no such tie.
    """
    for name, mask in first.genes.items():
        if not name.startswith(MASK_PREFIX):
            continue
        weight = f"{WEIGHT_PREFIX}{name[len(MASK_PREFIX) :]}"
        if (
            name in second.genes
            and weight in first.genes
            and weight in second.genes
            and mask != second.genes[name]
            and _bits(first.genes[weight]) == _bits(second.genes[weight])
        ):
            return True
    return False


def parents_of_shape(population: AgentPopulation, shape: str) -> tuple[str, ...]:
    """The parents a birth of ``shape`` is asked for, from the first two agents.

    Raises:
        AssertionError: If the first two agents carry the same genome, so a
            "two distinct parents" birth would be the identical-parent shape
            under another name; or, for the untied shape, if they carry a
            weight tie at a mask they differ in.
    """
    first, second = population.agent_ids()[:2]
    assert population.genome(first) != population.genome(second), (
        "the first two agents carry one genome, so no birth here has two "
        "distinct parents"
    )
    if shape == TWO_UNTIED_PARENTS:
        assert not weight_tie_at_a_differing_mask(
            population.genome(first), population.genome(second)
        ), (
            "the first two agents carry a weight tie at a mask they differ in, so "
            "they are not the untied shape"
        )
    return {
        ONE_PARENT: (first,),
        TWO_DISTINCT_PARENTS: (first, second),
        TWO_IDENTICAL_PARENTS: (first, first),
        TWO_UNTIED_PARENTS: (first, second),
    }[shape]


def copies_of_each_parent(
    population: AgentPopulation, parents: tuple[str, ...]
) -> tuple[int, int]:
    """How many of :data:`VARIATION_SEEDS` births copy the first parent, and the last.

    A birth copies a parent when its genome, as returned — after every boundary
    rule has acted — carries that parent's value for every object gene. With
    one parent, or the same agent twice, the first and the last are one genome.
    """
    first, last = (population.genome(parents[0]), population.genome(parents[-1]))
    copies = [0, 0]
    for seed in VARIATION_SEEDS:
        genes = population.spawn(parents=parents, seed=seed).genome.genes
        objects = set(genes) - STRATEGY_GENES
        for place, parent in enumerate((first, last)):
            if all(genes[name] == parent.genes.get(name) for name in objects):
                copies[place] += 1
    return copies[0], copies[1]


def offer(population: AgentPopulation, agent_id: str, score: float) -> ArchiveEntry:
    """Return an entry describing ``agent_id`` as the population currently has it."""
    return ArchiveEntry(
        agent_id=agent_id,
        genome=population.genome(agent_id),
        descriptor=population.descriptor(agent_id),
        objectives=MappingProxyType({"score": score}),
        recorded_at=RECORDED_AT,
    )


#: The run a persisted record is written under, and another run it must not be
#: restored into. Declared names, not anybody's run.
PERSISTING_RUN = "contract-run"
ANOTHER_RUN = "another-contract-run"

#: Coordinates no record bins: past each end of the unit interval, and not a
#: number at all.
UNBINNABLE = (1.5, -0.5, math.nan)


def unbinnable(entry: ArchiveEntry, coordinate: float) -> ArchiveEntry:
    """``entry`` with every descriptor value moved to ``coordinate``, nothing else changed."""
    return replace(
        entry,
        descriptor=replace(
            entry.descriptor,
            values=MappingProxyType(dict.fromkeys(entry.descriptor.values, coordinate)),
        ),
    )


def offered_by_each(population: AgentPopulation) -> list[ArchiveEntry]:
    """Every declared score offered for every living agent; the entries offered, in order."""
    offered = []
    for agent_id in population.agent_ids():
        for score in OFFERED_SCORES:
            entry = offer(population, agent_id, score)
            population.record(entry)
            offered.append(entry)
    return offered


def coverage_problems(population: AgentPopulation) -> list[str]:
    """What stops ``population``'s coverage describing the entries it hands back."""
    coverage = population.coverage()
    held = population.recorded()
    problems: list[str] = []
    if not isinstance(coverage, ecology.RecordCoverage):  # ty: ignore[unresolved-attribute]
        return [f"coverage is a {type(coverage).__qualname__}, not a RecordCoverage"]
    if len(coverage.histograms) != len(coverage.axes):
        problems.append(
            f"{len(coverage.histograms)} histograms for {len(coverage.axes)} axes"
        )
    if coverage.total != math.prod(len(counts) for counts in coverage.histograms):
        problems.append(
            f"{coverage.total} cells reported, the axes' cells multiply to "
            f"{math.prod(len(counts) for counts in coverage.histograms)}"
        )
    for axis, counts in zip(coverage.axes, coverage.histograms, strict=False):
        if sum(counts) != len(held):
            problems.append(
                f"{axis!r} places {sum(counts)} entries, the record holds {len(held)}"
            )
    if held and not 1 <= coverage.filled <= min(len(held), coverage.total):
        problems.append(
            f"{coverage.filled} cells filled by {len(held)} entries out of "
            f"{coverage.total}"
        )
    if not held and coverage.filled != 0:
        problems.append(f"{coverage.filled} cells filled by an empty record")
    for entry in held:
        if set(entry.descriptor.values) != set(coverage.axes):
            problems.append(
                f"an entry on {sorted(entry.descriptor.values)} is reported on "
                f"{sorted(coverage.axes)}"
            )
    return problems


class AgentPopulationContract:
    """What a population promises, whichever variation and record it holds.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyAgentPopulation(AgentPopulationContract):
            adapter_factory = staticmethod(MyAgentPopulation)

    The factory returns a fresh population for each test, so a record filled by
    one test is never read by another. Called with no argument it builds the
    adapter's own small population; called with ``agent_count`` it builds one
    holding exactly that many agents, which is how the selection cases put a
    population either side of the size a run was configured for.
    """

    adapter_factory: ClassVar[Callable[..., AgentPopulation]]

    #: Builds the population the untied shape is asked of, whose first two
    #: agents differ and carry no weight tie at a mask they differ in. ``None``
    #: asks it of the adapter's own population.
    untied_factory: ClassVar[Callable[[], AgentPopulation] | None] = None

    #: Builds a population and names one of its living agents that it cannot
    #: yet describe. ``None`` declares that this registration holds no such
    #: agent, and the case that asks about one says so.
    undescribed_factory: ClassVar[
        Callable[[], tuple[AgentPopulation, AgentId]] | None
    ] = None

    @pytest.fixture
    def adapter(self) -> AgentPopulation:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    def sized(self, agent_count: int) -> AgentPopulation:
        """The adapter under test, built fresh holding ``agent_count`` agents."""
        return type(self).adapter_factory(agent_count=agent_count)

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
        """The ecology asks for a child, not for an operator; it gets one genome.

        The genome is the child's heritable part; what the variation did to make
        it travels beside it and is contracted below.
        """
        parents = adapter.agent_ids()[:2]
        lineage = [adapter.genome(agent_id) for agent_id in parents]
        offspring = adapter.spawn(parents=parents, seed=17).genome
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

    def test_a_birth_reports_the_variation_it_applied(
        self, adapter: AgentPopulation
    ) -> None:
        """A birth hands back the child with the scale in force and every firing.

        The scale is absent only where no adaptive scale is in force; where one
        is, it is a positive finite number. Firings are counted for every gene
        the child carries — zero where the boundary rule did not act — so an
        absent count can never be mistaken for a rule that did not fire.
        """
        parents = adapter.agent_ids()[:2]
        offspring = adapter.spawn(parents=parents, seed=17)
        assert isinstance(offspring, Offspring)
        assert isinstance(offspring.genome, Genome)
        assert offspring.genome.schema_version == (
            adapter.genome(parents[0]).schema_version
        )
        scale = offspring.mutation_scale
        assert scale is None or (math.isfinite(scale) and scale > 0.0)
        assert set(offspring.boundary_firings) == set(offspring.genome.genes)
        for count in offspring.boundary_firings.values():
            assert isinstance(count, int)
            assert count >= 0

    @pytest.mark.parametrize("shape", GUARD_SHAPES)
    def test_offspring_differs_from_each_parent(
        self, adapter: AgentPopulation, shape: str
    ) -> None:
        """A birth is not a copy, whichever way its parents stand.

        Over :data:`VARIATION_SEEDS` births, no offspring carries the first
        parent's value for every object gene, and none the second parent's —
        except where two different parents carry a bit-equal ``weight.<f>`` at
        a feature whose ``mask.<f>`` differs, the one copy the mask-inheritance
        tie rule makes. Identical parents are a shape of their own because
        recombination alone returns their genome unchanged, and a converged
        population offers them constantly; untied parents are a shape of their
        own so that a copy of the second parent is always asked about.
        """
        factory = type(self).untied_factory
        population = factory() if shape == TWO_UNTIED_PARENTS and factory else adapter
        parents = parents_of_shape(population, shape)
        first, second = copies_of_each_parent(population, parents)
        tied = len(set(parents)) == 2 and weight_tie_at_a_differing_mask(
            population.genome(parents[0]), population.genome(parents[1])
        )
        assert first == 0 and (second == 0 or tied), (
            f"{first} of {len(VARIATION_SEEDS)} offspring of {shape} copied the "
            f"first parent and {second} the second (a copy of the second "
            f"{'is' if tied else 'is not'} admitted by a weight tie at a "
            "differing mask): variation was not applied"
        )

    @pytest.mark.parametrize("living", DRIFTED_SIZES)
    def test_selection_returns_exactly_the_parents_asked_for_from_the_living(
        self, living: int
    ) -> None:
        """A population drifted from its configured size still gets what it asks.

        The request is the configured size; the population is not. Every parent
        returned is alive, and the count is the request, not whatever a
        configuration built for a different population would have produced.
        """
        population = self.sized(living)
        select_parents = population.select_parents
        agent_ids = population.agent_ids()
        assert len(agent_ids) == living
        parents = select_parents(
            fitness=declared_fitness(agent_ids), count=CONFIGURED_SIZE, seed=17
        )
        assert isinstance(parents, tuple)
        assert len(parents) == CONFIGURED_SIZE
        assert set(parents) <= set(agent_ids)

    def test_selection_is_reproducible(self) -> None:
        """The same fitness, count and seed choose the same parents in order."""
        population = self.sized(DRIFTED_SIZES[0])
        select_parents = population.select_parents
        fitness = declared_fitness(population.agent_ids())
        assert select_parents(
            fitness=fitness, count=CONFIGURED_SIZE, seed=17
        ) == select_parents(fitness=fitness, count=CONFIGURED_SIZE, seed=17)

    def test_selection_does_not_by_itself_change_who_is_alive(self) -> None:
        """Choosing parents is not a birth or a death; the living are unchanged."""
        population = self.sized(DRIFTED_SIZES[1])
        select_parents = population.select_parents
        before = population.agent_ids()
        select_parents(fitness=declared_fitness(before), count=2, seed=17)
        assert population.agent_ids() == before

    def test_selection_refuses_a_fitness_not_keyed_by_the_living(self) -> None:
        """A fitness for a different population is refused, not indexed past.

        Two shapes: the living outnumber the fitnesses (probe H-2, where an
        engine sized by the fitnesses indexes past them), and the fitnesses name
        an agent that is not alive. Either is a caller holding a stale view of
        the population, which is a defect to surface rather than a choice to
        make on the caller's behalf.
        """
        population = self.sized(DRIFTED_SIZES[1])
        select_parents = population.select_parents
        agent_ids = population.agent_ids()
        with pytest.raises(ValueError, match="living population"):
            select_parents(
                fitness=declared_fitness(agent_ids[:-MISMATCHED_BY]),
                count=CONFIGURED_SIZE,
                seed=17,
            )
        stale = {
            **declared_fitness(agent_ids),
            **{f"not-alive-{index}": 0.0 for index in range(MISMATCHED_BY)},
        }
        with pytest.raises(ValueError, match="living population"):
            select_parents(fitness=stale, count=CONFIGURED_SIZE, seed=17)

    def test_selection_refuses_a_fitness_it_cannot_order(self) -> None:
        """A non-finite fitness has no place in an ordering, so it is refused.

        Any comparison with a missing number is false, so a selection that
        accepted one would choose by the order agents happen to be held in and
        report it as a choice by fitness.
        """
        population = self.sized(DRIFTED_SIZES[0])
        select_parents = population.select_parents
        fitness = declared_fitness(population.agent_ids())
        fitness[population.agent_ids()[0]] = math.nan
        with pytest.raises(ValueError, match="finite"):
            select_parents(fitness=fitness, count=CONFIGURED_SIZE, seed=17)

    def test_the_fewest_candidates_selection_can_choose_among_is_declared(
        self, adapter: AgentPopulation
    ) -> None:
        """A whole number of agents, at least one."""
        minimum = adapter.minimum_candidates
        assert isinstance(minimum, int)
        assert not isinstance(minimum, bool)
        assert minimum >= 1

    @pytest.mark.parametrize("seed", SELECTION_SEEDS)
    def test_an_agent_without_a_figure_is_never_chosen(self, seed: int) -> None:
        """With enough figures present, only the agents carrying one breed.

        Exactly the declared minimum carry a figure, so the case sits at the
        floor, and the agents without one are those a reader of positions would
        have preferred.
        """
        population = self.sized(DRIFTED_SIZES[0])
        agent_ids = population.agent_ids()
        fitness = figures_for_the_first(agent_ids, population.minimum_candidates)
        parents = population.select_parents(
            fitness=fitness, count=CONFIGURED_SIZE, seed=seed
        )
        assert len(parents) == CONFIGURED_SIZE
        assert all(fitness[parent] is not None for parent in parents)

    def test_with_no_figure_at_all_parents_are_drawn_from_the_living(self) -> None:
        """A blind call still returns exactly what was asked, reproducibly."""
        population = self.sized(DRIFTED_SIZES[0])
        agent_ids = population.agent_ids()
        blind: dict[str, float | None] = dict.fromkeys(agent_ids)
        parents = population.select_parents(
            fitness=blind, count=CONFIGURED_SIZE, seed=17
        )
        assert isinstance(parents, tuple)
        assert len(parents) == CONFIGURED_SIZE
        assert set(parents) <= set(agent_ids)
        assert parents == population.select_parents(
            fitness=blind, count=CONFIGURED_SIZE, seed=17
        )

    def test_some_figures_but_too_few_to_choose_among_are_refused(self) -> None:
        """Whether a trigger is blind is the caller's call, never made here.

        Every count of figures from one up to one below the declared minimum is
        refused. A population whose minimum is one has no such count, and the
        case says so rather than passing over nothing.
        """
        population = self.sized(DRIFTED_SIZES[0])
        minimum = population.minimum_candidates
        if minimum < 2:
            pytest.skip("no count of figures is both present and below one")
        for carrying in range(1, minimum):
            with pytest.raises(ValueError, match="minimum"):
                population.select_parents(
                    fitness=figures_for_the_first(population.agent_ids(), carrying),
                    count=CONFIGURED_SIZE,
                    seed=17,
                )

    def test_selection_of_no_parents_is_refused(self) -> None:
        """A request for nothing is not a selection; it is refused, not answered."""
        population = self.sized(DRIFTED_SIZES[0])
        select_parents = population.select_parents
        with pytest.raises(ValueError, match="at least one parent"):
            select_parents(
                fitness=declared_fitness(population.agent_ids()), count=0, seed=17
            )

    def test_an_empty_record_reports_no_cell_filled(
        self, adapter: AgentPopulation
    ) -> None:
        """A record nobody has offered anything holds nothing and covers nothing."""
        coverage = adapter.coverage()
        assert isinstance(coverage, ecology.RecordCoverage)  # ty: ignore[unresolved-attribute]
        assert coverage.filled == 0
        assert coverage.total >= 1
        assert all(count == 0 for counts in coverage.histograms for count in counts)
        assert coverage_problems(adapter) == []

    def test_coverage_is_read_from_the_record(self, adapter: AgentPopulation) -> None:
        """Cells filled and per-axis counts agree with the entries handed back."""
        offered_by_each(adapter)
        assert adapter.recorded() != ()
        assert coverage_problems(adapter) == []

    @pytest.mark.parametrize("coordinate", UNBINNABLE)
    def test_an_entry_the_record_cannot_bin_is_excluded_and_reported(
        self, adapter: AgentPopulation, coordinate: float
    ) -> None:
        """Refused rather than binned anywhere, kept aside, and nothing else changed."""
        assert adapter.excluded() == ()
        adapter.record(offer(adapter, adapter.agent_ids()[0], 0.0))
        held, covered = adapter.recorded(), adapter.coverage()
        entry = unbinnable(offer(adapter, adapter.agent_ids()[-1], 1.0), coordinate)
        assert adapter.record(entry) is False
        assert adapter.recorded() == held
        assert adapter.coverage() == covered
        assert adapter.excluded() == (entry,)

    def test_a_persisted_record_restores_into_its_own_run(
        self, adapter: AgentPopulation
    ) -> None:
        """The same entries, in the same order, and the same coverage, in a fresh population."""
        offered_by_each(adapter)
        payload = adapter.persisted(PERSISTING_RUN)
        assert isinstance(payload, ecology.RecordPayload)  # ty: ignore[unresolved-attribute]
        assert payload.run_id == PERSISTING_RUN
        fresh = type(self).adapter_factory()
        assert fresh.recorded() == ()
        fresh.restore(payload, PERSISTING_RUN)
        assert fresh.recorded() == adapter.recorded()
        assert fresh.coverage() == adapter.coverage()

    def test_persisting_leaves_the_record_as_it_was(
        self, adapter: AgentPopulation
    ) -> None:
        """Writing a payload is a read of the record, not a change to it."""
        offered_by_each(adapter)
        held, covered = adapter.recorded(), adapter.coverage()
        adapter.persisted(PERSISTING_RUN)
        assert adapter.recorded() == held
        assert adapter.coverage() == covered

    @pytest.mark.parametrize(
        ("changed", "refused_for"),
        [
            ({"run_id": ANOTHER_RUN}, "run"),
            ({"payload_version": "another-payload-version"}, "version"),
            ({"descriptor_schema_version": "another-descriptors-1"}, "vocabulary"),
            ({"axes": ("another_axis",)}, "axes"),
            ({"bins": (1,)}, "bins"),
            ({"bin_capacity": 2}, "capacity"),
        ],
    )
    def test_a_foreign_payload_is_refused_rather_than_loaded(
        self,
        adapter: AgentPopulation,
        changed: dict[str, object],
        refused_for: str,
    ) -> None:
        """A payload whose envelope does not name this run's record loads nothing."""
        offered_by_each(adapter)
        foreign = replace(adapter.persisted(PERSISTING_RUN), **changed)
        fresh = type(self).adapter_factory()
        with pytest.raises(ValueError, match=refused_for):
            fresh.restore(foreign, PERSISTING_RUN)
        assert fresh.recorded() == ()

    def test_a_payload_is_never_restored_into_another_run(
        self, adapter: AgentPopulation
    ) -> None:
        """The envelope names the run that wrote it; another run is refused."""
        offered_by_each(adapter)
        payload = adapter.persisted(PERSISTING_RUN)
        fresh = type(self).adapter_factory()
        with pytest.raises(ValueError, match="run"):
            fresh.restore(payload, ANOTHER_RUN)
        assert fresh.recorded() == ()

    def test_a_restore_into_a_record_that_holds_something_is_refused(
        self, adapter: AgentPopulation
    ) -> None:
        """Restoring replays a whole record; it is never merged into another."""
        offered_by_each(adapter)
        payload = adapter.persisted(PERSISTING_RUN)
        held = adapter.recorded()
        with pytest.raises(ValueError, match="empty"):
            adapter.restore(payload, PERSISTING_RUN)
        assert adapter.recorded() == held

    def test_an_undescribed_agent_is_refused_and_given_no_coordinate(self) -> None:
        """An agent short of what its descriptor reads has no place, and none is made up."""
        factory = type(self).undescribed_factory
        if factory is None:
            pytest.skip("this registration holds no agent it cannot describe")
        population, agent_id = factory()
        assert agent_id in population.agent_ids()
        with pytest.raises(DescriptorUnavailable):
            population.descriptor(agent_id)


class TestNullAgentPopulation(AgentPopulationContract):
    """The stand-in, held to the same contract as any provider-backed population."""

    adapter_factory = staticmethod(NullAgentPopulation)


@dataclass
class _Cloning(NullAgentPopulation):
    """A wrong build: every birth hands back its first parent's genome."""

    def spawn(self, *, parents: Sequence[str], seed: int) -> Offspring:
        genome = self.genome(parents[0])
        return Offspring(
            genome=genome,
            mutation_scale=None,
            boundary_firings=MappingProxyType(dict.fromkeys(genome.genes, 0)),
        )


class _HeldToTheContract(AgentPopulationContract):
    """The contract, holding the cloning wrong build; never collected itself."""

    adapter_factory = staticmethod(_Cloning)


@pytest.mark.parametrize("shape", GUARD_SHAPES)
def test_the_offspring_differs_from_each_parent_guard_catches_a_clone(
    shape: str,
) -> None:
    """The guard fails a population that clones, at every one of its seeds."""
    with pytest.raises(AssertionError, match=f"^{len(VARIATION_SEEDS)} of"):
        _HeldToTheContract().test_offspring_differs_from_each_parent(_Cloning(), shape)


def test_offspring_differs_from_each_parent_in_the_genes_the_schema_calls_objects() -> (
    None
):
    """The strategy genes the guard leaves out are the ones the schema declares.

    The guard reads :data:`STRATEGY_GENES`, a literal here; the schema's own set
    must equal it, so reclassifying an object gene as a strategy gene — which
    would weaken the guard without touching it — is an edit to this evidence.
    """
    assert ecology.genome_schema.STRATEGY_GENES == STRATEGY_GENES  # type: ignore[attr-defined]


# The array-genome population's declared floor, read from the adapter so the
# case below is one below whatever the platform's configuration requires.
ARRAY_FLOOR = an_array_genome_population().minimum_candidates


# Two founders designed across the declared ranges, chosen before any birth of
# theirs was counted: a declared seed of the platform's design primitive over
# the contract's two features. Their weight genes differ at every feature, so no
# mask they differ in is tied — asserted where they are used.
UNTIED_SEED = 0


def an_untied_array_genome_population() -> ArrayGenomePopulation:
    """Two designed founders whose ``weight.`` genes differ at every feature.

    The only place in this contract a weight gene is moved off 0.0: the tied
    population :func:`an_array_genome_population` builds is never replaced.
    """
    births = design_founders(
        features=ARRAY_FEATURES, learning=LearningConfig(), count=2, seed=UNTIED_SEED
    )
    first, second = (birth.genome.genes for birth in births)
    assert all(
        first[f"{WEIGHT_PREFIX}{name}"] != second[f"{WEIGHT_PREFIX}{name}"]
        for name in ARRAY_FEATURES
    ), "the designed founders share a weight, so they are not untied"
    return ArrayGenomePopulation(
        state=PopulationState.founded(births),
        descriptor_schema_version=ARRAY_DESCRIPTOR_SCHEMA,
    )


class TestArrayGenomePopulation(AgentPopulationContract):
    """The array-genome population, held to the contract the stand-in is held to.

    Two adapters differing in mechanism — one a dictionary of genomes, one a
    matrix of genes — run the same suite, which is the only way the suite
    demonstrates it is a contract rather than a description of one implementation.
    """

    adapter_factory = staticmethod(an_array_genome_population)
    untied_factory = staticmethod(an_untied_array_genome_population)

    def test_a_population_too_small_for_the_platforms_selection_is_refused(
        self,
    ) -> None:
        """Too few agents carrying a figure are refused before the platform is asked.

        The size is not restated here: it is one below the floor the adapter
        declares from the platform's own configuration. Every agent carries a
        figure, so the call is not blind, and the refusal is the port's shared
        one rather than the platform's error, which can no longer be reached.
        """
        population = an_array_genome_population(agent_count=ARRAY_FLOOR - 1)
        fitness = declared_fitness(population.agent_ids())
        with pytest.raises(ValueError, match="minimum") as refused:
            population.select_parents(fitness=fitness, count=2, seed=17)
        assert not isinstance(refused.value, EcologyError)


@dataclass
class _CloningTheSecond(ArrayGenomePopulation):
    """A wrong build: every birth hands back its last parent's genome."""

    def spawn(self, *, parents: Sequence[str], seed: int) -> Offspring:
        genome = self.genome(parents[-1])
        return Offspring(
            genome=genome,
            mutation_scale=None,
            boundary_firings=MappingProxyType(dict.fromkeys(genome.genes, 0)),
        )


def _cloning_the_second(population: ArrayGenomePopulation) -> _CloningTheSecond:
    return _CloningTheSecond(
        state=population.state,
        descriptor_schema_version=population.descriptor_schema_version,
    )


class _SecondHeldToTheContract(AgentPopulationContract):
    """The array registration, holding the second-parent cloner; never collected."""

    adapter_factory = staticmethod(
        lambda: _cloning_the_second(an_array_genome_population())
    )
    untied_factory = staticmethod(
        lambda: _cloning_the_second(an_untied_array_genome_population())
    )


def test_the_offspring_differs_from_each_parent_guard_catches_a_second_parent_cloner() -> (
    None
):
    """A build that always copies the second parent fails where nothing is tied.

    At the tied shape its every birth is the one copy the tie rule admits, so
    the guard cannot fail it there; the untied shape is what catches it.
    """
    contract = _SecondHeldToTheContract()
    adapter = _SecondHeldToTheContract.adapter_factory()
    contract.test_offspring_differs_from_each_parent(adapter, TWO_DISTINCT_PARENTS)
    with pytest.raises(
        AssertionError,
        match=f"^0 of {len(VARIATION_SEEDS)} offspring of {TWO_UNTIED_PARENTS} "
        f"copied the first parent and {len(VARIATION_SEEDS)} the second",
    ):
        contract.test_offspring_differs_from_each_parent(adapter, TWO_UNTIED_PARENTS)


def test_the_untied_shape_defaults_to_the_adapters_own_first_two_agents() -> None:
    """With no untied factory registered, the adapter's own first two are asked."""
    population = NullAgentPopulation()
    assert (
        parents_of_shape(population, TWO_UNTIED_PARENTS) == (population.agent_ids()[:2])
    )


def test_the_untied_shape_refuses_a_tied_pair() -> None:
    """The untied shape's precondition fails on the tied contract population."""
    with pytest.raises(AssertionError, match="not the untied shape"):
        parents_of_shape(an_array_genome_population(), TWO_UNTIED_PARENTS)


def test_a_weight_tie_is_read_bit_for_bit_and_only_at_a_differing_mask() -> None:
    """Equal weights at equal masks are no tie; 0.0 and -0.0 are not bit-equal."""

    def genome(mask: float, weight: float) -> Genome:
        return Genome(
            genes=MappingProxyType(
                {f"{MASK_PREFIX}f": mask, f"{WEIGHT_PREFIX}f": weight}
            ),
            schema_version=ARRAY_GENE_SCHEMA,
        )

    assert weight_tie_at_a_differing_mask(genome(0.0, 0.0), genome(1.0, 0.0))
    assert not weight_tie_at_a_differing_mask(genome(1.0, 0.0), genome(1.0, 0.0))
    assert not weight_tie_at_a_differing_mask(genome(0.0, 0.0), genome(1.0, -0.0))
    assert not weight_tie_at_a_differing_mask(genome(0.0, 0.0), genome(1.0, 0.5))
    no_weights = Genome(
        genes=MappingProxyType({f"{MASK_PREFIX}f": 0.0}),
        schema_version=ARRAY_GENE_SCHEMA,
    )
    assert not weight_tie_at_a_differing_mask(no_weights, genome(1.0, 0.0))


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ({"histograms": ()}, "histograms for 1 axes"),
        ({"histograms": ((0, -1),)}, "negative"),
        ({"filled": 3}, "cells filled"),
        ({"filled": -1}, "cells filled"),
    ],
)
def test_a_coverage_that_cannot_describe_a_record_is_refused(
    shape: dict[str, object], message: str
) -> None:
    """One histogram per axis, no negative count, and between none and all cells filled."""
    declared: dict[str, object] = {
        "axes": ("activity",),
        "filled": 1,
        "total": 2,
        "histograms": ((1, 0),),
        **shape,
    }
    with pytest.raises(ValueError, match=message):
        ecology.RecordCoverage(**declared)  # ty: ignore[unresolved-attribute]
