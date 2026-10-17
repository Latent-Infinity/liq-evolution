"""Keeping the population, over the arrays the population is actually held in.

This is the capability "hold the living agents and the record of what behaviours
have been seen", satisfied over :class:`~liq.evolution.ecology.agent.PopulationState`.
It owns no state of its own beyond the record below: every question about an
agent is answered by the store, so there is one representation of a population
and not two that have to be kept in step.

**Variation and selection are the operator decision's**
(``liq-docs/plans/agent-ecology-adr-evolution-operators.md``, "Successor-3
result"), applied through the platform's primitives and under the ranges and
rules :mod:`~liq.evolution.ecology.genome_schema` declares.

*Variation.* On one generator, ``numpy.random.default_rng(seed)``, in order:

1. *Scale.* The child's mutation scale is a self-adaptive log-normal step from
   its parents' ``mutation_scale`` genes (a parent carrying none is read at the
   schema's founding scale), at a learning rate set by the number of genes
   mutated — every gene but the scale gene — and reflected in log space at the
   ends of its range. One scale serves every gene, the forgetting factor
   included.
2. *Recombination.* Uniform: each gene other than a mask is taken from one
   parent or the other. One parent, or the same agent twice, recombines with
   itself.
3. *Masks.* Each ``mask.<f>`` is inherited, without a draw, from the parent
   whose ``weight.<f>`` is nearer the child's recombined ``weight.<f>`` — the
   first parent on a tie — so a mask travels with the weight it switches.
4. *Mutation.* Each gene is selected with probability one over their number,
   one at least. A selected mask is flipped; any other selected gene moves by
   the child's scale times a standard normal draw in the unit coordinates of
   its range, and is reflected back inside if the step left it.

The child carries the scale it was drawn at wherever its parents carry the
gene, and the birth reports that scale and, per gene, whether a reflection
acted on it. A parent carrying a gene outside its declared range — a founder
declared there by hand — is refused when it is first asked to breed, before
anything is drawn: the operators work only inside the declared ranges, and
parents inside them never produce a gene outside one, so a gene outside its
range could only be carried on from a genome that already breached it.

*Founders.* :func:`design_founders` designs an initial population across the
declared ranges: a scrambled Sobol' design from the platform's design primitive,
mapped onto every mask, weight, the entry threshold and the forgetting factor.
Founders carry the founding scale and nothing learned, so they enter through
``founded()`` and start from their own priors.

*Selection.* Parents are chosen by the platform's own tournament selection, at
the decided tournament size of two, reused unchanged and reached from nowhere
else in the ecology. That selection was written for a generational population
whose size is fixed by its configuration; a population here turns over
continuously and drifts from any size recorded anywhere. So the configuration is
built afresh on every call from the agents it is to rank, and sized to return
exactly the parents asked for.

*An agent without a figure.* Only agents handed a figure are ranked, so an
agent handed none is never chosen while enough others carry one. The fewest
that can be ranked is the platform configuration's own population floor, which
is declared here as :attr:`ArrayGenomePopulation.minimum_candidates` rather
than restated anywhere else. When nobody carries a figure the call is blind:
parents are drawn uniformly from the living with the seeded generator the
tournament would have used. That draw is this adapter's own ``choice`` over the
living rather than the platform's tournament at size one, because a blind
draw ranks nothing and the platform's configuration refuses a population below
its floor, which a blind draw must still serve. Whether a call was blind is
known to its caller, which decided it by handing no figures; it is not
reported back from here.

*The record.* The diversity record is the platform's own quality-diversity
archive, composed as it is: never forked, shadowed or subclassed, and held as
:attr:`ArrayGenomePopulation.archive` so that whoever asks can see it is that
archive and not something shaped like it. Every entry offered is handed to it,
and every entry reported is read back from it; nothing here keeps a list beside
it. Which cell an entry falls in, what it displaces and what it is displaced by
are the archive's decisions, so the rules this adapter adds are only about what
it hands over:

* *When an entry is offered* is the caller's decision; this adapter places
  nothing on its own initiative.
* *Axes.* One per descriptor this population reads, both declared on the unit
  interval, with one cell for every value the descriptor can take — the
  descriptors' own resolution, not a binning anybody tuned. A cell holds one
  entry, the archive's own default.
* *Directions.* The figures an entry carries are read in name order against
  the directions of the platform's canonical fitness configuration, in the
  vocabulary of the platform's single direction schema. Neither is restated
  here.
* *Refused rather than binned.* A descriptor outside the unit interval, or one
  that is not a number at all, is refused by the archive's own bounds check;
  here that refusal becomes an exclusion — the entry is not kept, is kept
  aside for whoever reports exclusions, and nothing else changes. An entry in
  another descriptor vocabulary, or carrying a different number of figures
  than there are directions, is a caller's defect and raises.
* *Eviction never kills a living agent, and a death never removes an entry.*
  The archive holds entries, which are records of agents, and the population
  holds agents. Neither is consulted when the other changes.

*The descriptors.* Two figures, both read off what the agent does rather than
how well it does it, both normalised to the unit interval, and both stamped with
a vocabulary the caller names. They are provisional in the same way as the two
above: the descriptor set this ecology will actually niche on is a separate
piece of work, and when it arrives it bumps the schema version rather than
silently changing what an old archive's coordinates meant.

*The declared descriptors.* Built with a
:class:`~liq.evolution.ecology.descriptors.DescriptorDeclaration` and the bins of
its three behavioural axes — both or neither, the bins always the caller's and
never defaulted here — the population is described by the ecology's descriptor
module instead, :mod:`~liq.evolution.ecology.descriptors`:

* *Axes.* The module's four, in its order, each on the unit interval, binned at
  one cell per declared asset class and at the counts handed in for the rest.
* *Descriptor.* The module's reading of the agent over what the store holds,
  under the module's vocabulary. An agent the store holds too little of is
  refused with :class:`~liq.evolution.ecology.descriptors.DescriptorUnavailable`
  and given no coordinate; a value the module marks excluded reaches the record
  unaltered and is excluded there, never clamped.
* *Vocabulary.* The record refuses any entry not described in the module's
  vocabulary, the provisional one included, and the population refuses to be
  built under any vocabulary but that one. The vocabulary is read from the
  module, never restated here.

Without a declaration the two descriptors above are read, unchanged. Either
way, the provisional vocabulary those two were first stamped with is refused at
construction: no population is built under it.

*What the record reports.* Its coverage is the platform archive's own report of
its cells; the entries it would not bin are the exclusions kept aside above.
*Persistence* is :mod:`~liq.evolution.ecology.adapters.archive_codec`'s: the
archive's entries encoded inside an envelope naming the run, restored only into
an empty record of that run by replaying them through the archive's insertion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np

from liq.evolution.config import FitnessConfig
from liq.evolution.ecology import descriptors as declared_axes
from liq.evolution.ecology import genome_schema
from liq.evolution.ecology.adapters import archive_codec
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    MASK_PREFIX,
    SWITCHED_ON_AT,
    WEIGHT_PREFIX,
    AgentBirth,
    PopulationState,
)
from liq.evolution.ecology.agent_genes import MUTATION_SCALE, _gene_layout
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.errors import EcologyError, translating
from liq.evolution.ecology.ports import refuse_an_unanswerable_selection
from liq.evolution.ecology.types import (
    AgentId,
    ArchiveEntry,
    Descriptor,
    Genome,
    LivingOccupancy,
    Offspring,
    RecordCoverage,
    RecordPayload,
)
from liq.evolution.fitness.evaluation_schema import ObjectiveDirection
from liq.gp import ConfigurationError, FitnessResult, GPConfig, Program
from liq.gp.evolution import array_design, array_operators
from liq.gp.evolution.qd_archive import QDArchive
from liq.gp.evolution.selection import tournament_select

__all__ = [
    "EXPOSURE_WANTED",
    "FEATURES_READ",
    "UNDECLARED_VERSION",
    "ArrayGenomePopulation",
    "SelectionUnavailable",
    "design_founders",
]

#: Descriptor name: how much of what it could read an agent switches on. A
#: behaviour, not a quality — a parsimonious agent and a greedy one are
#: different, and neither is thereby better.
FEATURES_READ = "features_read"

#: Descriptor name: whether the agent currently wants to be in the market. The
#: rule wants all of its account or none of it, so this is already on the unit
#: interval by construction rather than by clipping.
EXPOSURE_WANTED = "exposure_wanted"


#: The axes the record bins on, in the order the archive's coordinates take
#: them. Both descriptors are on the unit interval by construction.
DESCRIPTOR_AXES = (FEATURES_READ, EXPOSURE_WANTED)
UNIT_INTERVAL = (0.0, 1.0)

#: Cells per axis for the wish to be in the market: it wants all of its
#: account or none of it, so it takes two values and each has its own cell.
EXPOSURE_CELLS = 2


#: The smallest population the platform's selection can be configured for:
#: ``liq.gp`` refuses ``GPConfig(population_size=...)`` below it
#: (``liq-gp/src/liq/gp/config.py``, "population_size must be >= 10"; probe
#: record §1.5, H-3). Declared once here as the population's
#: ``minimum_candidates`` (``plans/oracle-v3-unassessed-fitness-2026-10-01.md``
#: §3.3) and pinned by a test that builds the configuration at it and is
#: refused one below it.
SMALLEST_SELECTABLE_POPULATION = 10

#: The tournament size the operator decision selected (S1). It is also the
#: fewest competitors the platform's tournament configuration admits.
FEWEST_COMPETITORS = 2


#: The vocabulary versions a designed founder is born under where its caller
#: names none: a design read for its genes alone. A run founding on a design
#: names both.
UNDECLARED_VERSION = "undeclared"


def design_founders(
    *,
    features: Sequence[str],
    learning: LearningConfig,
    count: int,
    seed: int,
    feature_schema_version: str = UNDECLARED_VERSION,
    model_version: str = UNDECLARED_VERSION,
) -> tuple[AgentBirth, ...]:
    """``count`` founders designed across the declared gene ranges.

    A scrambled Sobol' design over the 2F + 2 designed genes — every mask, then
    every weight, the entry threshold and the forgetting factor, in the schema's
    layout — drawn from ``numpy.random.default_rng(seed)`` through the
    platform's design primitive, and mapped onto the ranges
    :mod:`~liq.evolution.ecology.genome_schema` declares: a mask is on where its
    coordinate is at least one half, and every other gene is its range's low end
    plus the coordinate times its width. Every founder carries the founding
    mutation scale and nothing learned, so it enters through ``founded()`` and
    starts from its own priors.

    Raises:
        ValueError: If ``count`` is below one.
    """
    names = tuple(features)
    ranges = genome_schema.declared(features=names, learning=learning)
    masks = tuple(f"{MASK_PREFIX}{name}" for name in names)
    columns = (
        *masks,
        *(f"{WEIGHT_PREFIX}{name}" for name in names),
        ENTRY_THRESHOLD,
        FORGETTING_FACTOR,
    )
    design = array_design.sobol_design(count, len(columns), np.random.default_rng(seed))
    width = len(str(count - 1))
    births = []
    for index, row in enumerate(design):
        genes = {
            name: (
                float(unit >= 0.5)
                if name in masks
                else ranges[name].low
                + float(unit) * (ranges[name].high - ranges[name].low)
            )
            for name, unit in zip(columns, row, strict=True)
        }
        genes[MUTATION_SCALE] = genome_schema.FOUNDING_SCALE
        births.append(
            AgentBirth(
                agent_id=f"founder-{index:0{width}d}",
                genome=Genome(
                    genes=MappingProxyType(genes),
                    schema_version=genome_schema.SCHEMA_VERSION,
                ),
                learned_state=MappingProxyType({}),
                feature_schema_version=feature_schema_version,
                model_version=model_version,
            )
        )
    return tuple(births)


def _refuse_a_gene_outside_its_range(
    agent_id: AgentId, genome: Genome, ranges: Mapping[str, genome_schema.GeneRange]
) -> None:
    """Refuse a parent carrying any declared gene outside its declared range."""
    for name, value in sorted(genome.genes.items()):
        declared = ranges.get(name)
        if declared is not None and not declared.low <= value <= declared.high:
            raise ValueError(
                f"{agent_id!r} carries {name} = {value!r}, outside its declared "
                f"range [{declared.low!r}, {declared.high!r}]; no operator varies "
                "a gene from outside its range"
            )


class SelectionUnavailable(EcologyError):
    """The platform's selection could not be configured for the living population.

    Raised, with the platform's own error as ``__cause__``, where the population
    has fallen below the smallest size that selection can be configured for. It
    is a condition a turning-over population can reach, not a caller's defect,
    which is why it is an ecology error rather than an argument error.
    """


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
            safely and would be compared anyway. With a declaration, it must be
            the descriptor module's own.
        descriptors: What the declared four axes are computed under, or
            ``None`` for the two descriptors read off the genome and the wish.
        descriptor_bins: The bins of the holding-horizon, turnover and beta
            axes, in that order, with a declaration and only with one.

    Raises:
        ValueError: If the vocabulary is the provisional one, with or without a
            declaration; if a declaration comes without its bins or bins
            without a declaration; if a declared population names a
            vocabulary other than the descriptor module's; or if its
            declaration keeps a different history per agent than its store
            does.
    """

    state: PopulationState
    descriptor_schema_version: str
    descriptors: declared_axes.DescriptorDeclaration | None = None
    descriptor_bins: tuple[int, int, int] | None = None

    archive: QDArchive = field(init=False, repr=False)
    _excluded: list[ArchiveEntry] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compose the platform's archive over this population's descriptors."""
        if self.descriptor_schema_version == declared_axes.PROVISIONAL_VOCABULARY:
            raise ValueError(
                f"{declared_axes.PROVISIONAL_VOCABULARY!r} is the provisional "
                "vocabulary; no population is built under it, declared or not, so "
                "no coordinate of it is ever kept in a record"
            )
        if (self.descriptors is None) != (self.descriptor_bins is None):
            raise ValueError(
                "a descriptor declaration and its bins are given both or neither: "
                "a declaration without bins has no cells to bin on, and bins "
                "without a declaration bin axes nobody computes"
            )
        directions: list[ObjectiveDirection] = list(
            FitnessConfig().objective_directions
        )
        if self.descriptors is None or self.descriptor_bins is None:
            self.archive = QDArchive(
                n_dims=len(DESCRIPTOR_AXES),
                bins_per_dim=(len(self.state.feature_names) + 1, EXPOSURE_CELLS),
                descriptor_bounds=(UNIT_INTERVAL,) * len(DESCRIPTOR_AXES),
                objective_directions=directions,
            )
            return
        if self.descriptor_schema_version != declared_axes.VOCABULARY:
            raise ValueError(
                f"a population described under the declared axes keeps its "
                f"record in vocabulary {declared_axes.VOCABULARY!r}, not "
                f"{self.descriptor_schema_version!r}"
            )
        if self.descriptors.history_capacity != self.state.history_capacity:
            raise ValueError(
                f"the descriptor declaration's history_capacity "
                f"{self.descriptors.history_capacity} is not the "
                f"{self.state.history_capacity} outcomes the store keeps per "
                "agent; a window read past what the store keeps would leave "
                "every agent undescribed without saying why"
            )
        horizon, turnover, beta = self.descriptor_bins
        self.archive = QDArchive(
            n_dims=len(declared_axes.AXES),
            bins_per_dim=self.descriptors.archive_bins(
                holding_horizon=horizon, turnover=turnover, beta=beta
            ),
            descriptor_bounds=(UNIT_INTERVAL,) * len(declared_axes.AXES),
            objective_directions=directions,
        )

    @property
    def descriptor_declaration(self) -> declared_axes.DescriptorDeclaration | None:
        """The declaration this population was built with, the same object, or ``None``."""
        return self.descriptors

    @property
    def _axes(self) -> tuple[str, ...]:
        """The axes the record bins on, in the order its coordinates take them."""
        return DESCRIPTOR_AXES if self.descriptors is None else declared_axes.AXES

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
        """Return where one agent currently sits in behaviour space.

        Raises:
            DescriptorUnavailable: With a declaration, if the store holds fewer
                outcomes for the agent than its descriptor reads.
        """
        if self.descriptors is not None:
            reading = declared_axes.describe(self.state, self.descriptors, agent_id)
            return Descriptor(
                values=MappingProxyType(
                    {axis: reading.values[axis] for axis in declared_axes.AXES}
                ),
                schema_version=declared_axes.VOCABULARY,
            )
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

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        """Return one offspring of ``parents``, with the variation it applied.

        Raises:
            ValueError: If no parents were named — a birth of nobody would have
                to invent an entire genome, which is a different operation from
                inheritance and is not this one — or more than two; if a
                parent's genome is under a schema version other than
                :data:`genome_schema.SCHEMA_VERSION`, whose ranges every gene
                is held to, which is named with the parent and both versions;
                if a gene has no declared range; or if a parent carries a gene
                outside its declared range, which is named with its value and
                range. The store holds one vocabulary of genes, so parents
                never differ in which genes they carry.
        """
        if not parents:
            raise ValueError(
                "a birth needs at least one parent: an offspring of nobody would "
                "be an initialisation, not an inheritance"
            )
        if len(parents) > 2:
            raise ValueError(
                f"a birth recombines one or two parents, not {len(parents)}"
            )
        lineage = [self.state.genome(agent_id) for agent_id in parents]
        for agent_id, genome in zip(parents, lineage, strict=True):
            if genome.schema_version != genome_schema.SCHEMA_VERSION:
                raise ValueError(
                    f"{agent_id!r} carries genome schema {genome.schema_version!r}, "
                    "but spawn holds every genome to "
                    f"{genome_schema.SCHEMA_VERSION!r}'s ranges"
                )
        names, features = _gene_layout(lineage[0])
        learning = self.state.learning or LearningConfig()
        ranges = genome_schema.declared(features=features, learning=learning)
        varied = tuple(name for name in names if name != MUTATION_SCALE)
        undeclared = [name for name in varied if name not in ranges]
        if undeclared:
            raise ValueError(
                f"{undeclared} have no declared range, so no operator can vary "
                "them inside one"
            )
        for agent_id, genome in zip(parents, lineage, strict=True):
            _refuse_a_gene_outside_its_range(agent_id, genome, ranges)
        first, second = (lineage[0], lineage[-1])
        p1 = np.array([first.genes[name] for name in varied], dtype=np.float64)
        p2 = np.array([second.genes[name] for name in varied], dtype=np.float64)
        low = np.array([ranges[name].low for name in varied], dtype=np.float64)
        high = np.array([ranges[name].high for name in varied], dtype=np.float64)
        is_mask = np.array([name.startswith(MASK_PREFIX) for name in varied])

        rng = np.random.default_rng(seed)
        step = array_operators.lognormal_scale_step(
            first.genes.get(MUTATION_SCALE, genome_schema.FOUNDING_SCALE),
            second.genes.get(MUTATION_SCALE, genome_schema.FOUNDING_SCALE),
            len(varied),
            rng,
            low=genome_schema.SCALE_MINIMUM,
            high=genome_schema.SCALE_MAXIMUM,
        )
        child = p1.copy()
        child[~is_mask] = array_operators.uniform_crossover(
            p1[~is_mask], p2[~is_mask], rng
        )
        for name in features:
            mask = varied.index(f"{MASK_PREFIX}{name}")
            weight = varied.index(f"{WEIGHT_PREFIX}{name}")
            nearer_second = abs(child[weight] - p2[weight]) < abs(
                child[weight] - p1[weight]
            )
            child[mask] = p2[mask] if nearer_second else p1[mask]
        mutation = array_operators.gaussian_mutation(
            child,
            low,
            high,
            step.scale,
            rng,
            bits=is_mask,
        )
        genes = {
            name: float(value)
            for name, value in zip(varied, mutation.values, strict=True)
        }
        firings = {
            name: int(fired) for name, fired in zip(varied, mutation.fired, strict=True)
        }
        if MUTATION_SCALE in first.genes:
            genes[MUTATION_SCALE] = step.scale
            firings[MUTATION_SCALE] = int(step.fired)
        return Offspring(
            genome=Genome(
                genes=MappingProxyType(genes),
                schema_version=genome_schema.SCHEMA_VERSION,
            ),
            mutation_scale=step.scale,
            boundary_firings=MappingProxyType(firings),
        )

    @property
    def minimum_candidates(self) -> int:
        """The fewest agents carrying a figure this population can rank."""
        return SMALLEST_SELECTABLE_POPULATION

    def select_parents(
        self, *, fitness: Mapping[AgentId, float | None], count: int, seed: int
    ) -> tuple[AgentId, ...]:
        """Return ``count`` living parents, ranked by figure or, with none, blind.

        The agents carrying a figure are ranked by the platform's tournament
        selection, its configuration rebuilt over them on every call. The
        platform's selection returns as many parents as its configuration's
        population size less its elite count, so each call is sized to return
        what is still owed, in rounds of at most one per ranked agent. With no
        figure at all, parents are drawn uniformly from the living.

        Raises:
            ValueError: If the request is not one the port's contract can
                answer — see
                :func:`~liq.evolution.ecology.ports.refuse_an_unanswerable_selection`,
                which also refuses some figures but fewer than
                :attr:`minimum_candidates`.
            SelectionUnavailable: If the platform's selection cannot be
                configured for the agents it is to rank.
        """
        living = self.state.agent_ids()
        refuse_an_unanswerable_selection(
            fitness=fitness,
            living=living,
            count=count,
            minimum=self.minimum_candidates,
        )
        rng = np.random.default_rng(seed)
        ranked = [
            (agent_id, figure)
            for agent_id in living
            if (figure := fitness[agent_id]) is not None
        ]
        if not ranked:
            drawn = rng.integers(len(living), size=count)
            return tuple(living[int(position)] for position in drawn)
        # The platform's selection is typed over programs and reads nothing of
        # them: it orders positions by the figures beside them and hands back
        # the members at the positions it chose, so identities pass through it.
        members = cast("list[Program]", [agent_id for agent_id, _ in ranked])
        figures = [FitnessResult(objectives=(figure,)) for _, figure in ranked]
        chosen: list[AgentId] = []
        while len(chosen) < count:
            owed = min(count - len(chosen), len(ranked))
            with translating(
                ConfigurationError,
                into=SelectionUnavailable,
                action=f"configuring selection for {len(ranked)} ranked agents",
            ):
                config = GPConfig(
                    population_size=len(ranked),
                    elitism_count=len(ranked) - owed,
                    tournament_size=FEWEST_COMPETITORS,
                )
            selected = tournament_select(members, figures, config, rng)
            chosen.extend(cast("list[AgentId]", selected))
        return tuple(chosen)

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the platform's archive; report whether it was kept.

        Raises:
            ValueError: If the entry's descriptor is in another vocabulary or
                names other axes, or if it carries a different number of
                figures than the archive has directions. An entry out of the
                unit interval, or not a number, is not raised: it is excluded.
        """
        if entry.descriptor.schema_version != self.descriptor_schema_version:
            raise ValueError(
                f"an entry described in {entry.descriptor.schema_version!r} "
                f"cannot be placed in a record kept in "
                f"{self.descriptor_schema_version!r}"
            )
        axes = self._axes
        if set(entry.descriptor.values) != set(axes):
            raise ValueError(
                f"an entry must be described on exactly {axes}, not "
                f"{tuple(sorted(entry.descriptor.values))}"
            )
        if len(entry.objectives) != len(self.archive.objective_directions):
            raise ValueError(
                f"an entry carrying {len(entry.objectives)} figure(s) cannot be "
                f"compared under {len(self.archive.objective_directions)} "
                "direction(s)"
            )
        try:
            return self.archive.insert(
                entry,
                tuple(entry.objectives[name] for name in sorted(entry.objectives)),
                tuple(entry.descriptor.values[axis] for axis in axes),
            )
        except ValueError:
            # The archive refuses a coordinate outside its declared bounds,
            # NaN included, before it places anything: an exclusion.
            self._excluded.append(entry)
            return False

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the platform's archive currently holds."""
        return tuple(cast("list[ArchiveEntry]", self.archive.elites()))

    def excluded(self) -> tuple[ArchiveEntry, ...]:
        """Return every entry refused for a coordinate the archive cannot bin."""
        return tuple(self._excluded)

    def occupancy(self) -> LivingOccupancy:
        """Return the cells the living occupy, binned by the platform archive's own rule.

        Each living agent's descriptor is placed with the archive's public
        ``cell_of``, which inserts nothing; the record's entries are never read.
        """
        axes = self._axes
        cells: set[tuple[int, ...]] = set()
        undescribed = excluded = 0
        for agent_id in self.agent_ids():
            try:
                descriptor = self.descriptor(agent_id)
            except declared_axes.DescriptorUnavailable:
                undescribed += 1
                continue
            try:
                cells.add(
                    self.archive.cell_of(
                        tuple(descriptor.values[axis] for axis in axes)
                    )
                )
            except ValueError:
                # Outside the archive's bounds, NaN included: in no cell.
                excluded += 1
        return LivingOccupancy(
            axes=axes,
            occupied=len(cells),
            total=self.archive.total_bins,
            described=len(self.agent_ids()) - undescribed - excluded,
            undescribed=undescribed,
            excluded=excluded,
        )

    def coverage(self) -> RecordCoverage:
        """Return the cells the platform's archive fills, from its own report."""
        report = self.archive.coverage_report()
        return RecordCoverage(
            axes=self._axes,
            filled=cast("int", report["filled_bins"]),
            total=cast("int", report["total_bins"]),
            histograms=tuple(
                tuple(counts)
                for counts in cast("list[list[int]]", report["dimension_histograms"])
            ),
        )

    def persisted(self, run_id: str) -> RecordPayload:
        """Return the platform's archive as a payload written by ``run_id``."""
        return archive_codec.persisted(
            self.archive,
            run_id=run_id,
            descriptor_schema_version=self.descriptor_schema_version,
            axes=self._axes,
        )

    def restore(self, payload: RecordPayload, run_id: str) -> None:
        """Replay ``payload`` into the empty platform archive, for restart of ``run_id``.

        Raises:
            ValueError: If the payload is not a record of this run in this
                record's vocabulary, axes, bins and capacity, or if the archive
                already holds an entry.
        """
        archive_codec.restore(
            self.archive,
            payload,
            run_id=run_id,
            descriptor_schema_version=self.descriptor_schema_version,
            axes=self._axes,
        )
