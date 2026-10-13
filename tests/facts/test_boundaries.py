"""The ecology's import boundaries, asserted.

Evidence for two claims: that nothing but a port carries a provider into the
code that decides behaviour, and that no broker or execution client is reachable
from this package at all — by import or by dependency edge.

The scanning lives in `tests/support/boundary_scan.py`. Only the assertions and
the failure messages live here, so that a change to *what is scanned* and a
change to *what is claimed* are separate edits to separate files.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
import textwrap
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

from liq.evolution.ecology import ArchiveEntry, Offspring
from liq.evolution.ecology import descriptors as declared_axes
from liq.evolution.ecology.adapters import NullAgentPopulation, array_genome
from liq.evolution.ecology.config import (
    EcologyConfig,
    LearningConfig,
    PopulationConfig,
)
from liq.evolution.ecology.types import AgentId
from liq.evolution.fitness import evaluation_schema
from liq.gp.evolution import selection as platform_selection
from liq.gp.evolution.qd_archive import QDArchive
from tests.support import boundary_scan, variation_flow
from tests.support.contract.test_agent_population import (
    CONFIGURED_SIZE,
    DRIFTED_SIZES,
    PARENT_SHAPES,
    VARIATION_SEEDS,
    an_array_genome_population,
    declared_fitness,
    offer,
    parents_of_shape,
)


def test_domain_reaches_no_provider_package() -> None:
    """Domain and use-case code speaks value types and ports, nothing else."""
    violations = boundary_scan.first_party_imports_outside_the_ports()
    assert violations == [], (
        f"a provider package reached code that decides behaviour: {violations}. "
        "What is needed arrives through one of the five ports; if no port "
        "expresses it, widen a port or write an adapter, rather than importing "
        "the library here."
    )


def test_no_broker_reachable() -> None:
    """No order-submitting library is importable or installed under this tree.

    Both halves are asserted together because the claim is one claim. A source
    scan alone would miss a broker arriving as somebody else's dependency, and
    a graph walk alone would miss an import of something already installed for
    an unrelated reason.
    """
    imported = boundary_scan.broker_imports()
    assert imported == [], (
        f"a broker or execution client is imported by the ecology: {imported}. "
        "This system produces a weight vector in an artifact and nothing that "
        "can act on it; that has to hold by construction, not by configuration."
    )

    reachable = boundary_scan.brokers_in_dependency_graph()
    assert reachable == [], (
        f"a broker or execution client is installed under this package: "
        f"{reachable}. Nothing imports it today, which is exactly how the next "
        "person finds it already there and uses it."
    )


def test_only_the_population_adapter_reaches_the_evolution_engine() -> None:
    """The ecology composes with one evolution engine and never grows another."""
    violations = boundary_scan.gp_evolution_imports_outside_the_population_adapter()
    assert violations == [], (
        f"the evolution engine is reached outside the population adapter: "
        f"{violations}. Selection, mutation and archive primitives are consumed "
        "through that one module; a second importer is a second engine."
    )


def test_only_the_execution_adapter_reaches_the_execution_model() -> None:
    """Fill mechanics are translated in one module and modelled in none."""
    violations = boundary_scan.execution_model_imports_outside_the_execution_adapter()
    assert violations == [], (
        f"the execution model is reached outside the execution adapter: "
        f"{violations}. Whether an order fills, at what price and at what "
        "charge is asked of the simulator through that one module; a second "
        "importer is an execution assumption spreading into code that decides."
    )


def test_no_archive_is_defined_here() -> None:
    """The archive is composed as-is, so there is nothing here to define."""
    violations = boundary_scan.archive_classes_defined_here()
    assert violations == [], (
        f"an archive is defined inside the ecology: {violations}. The existing "
        "archive is consumed without forking, shadowing or subclassing it."
    )


def test_the_scan_is_not_vacuous() -> None:
    """Guard against every assertion above passing over an empty file list."""
    if not boundary_scan.ECOLOGY_ROOT.is_dir():
        assert boundary_scan.ecology_files() == []
        assert boundary_scan.decision_making_files() == []
        return
    assert boundary_scan.ecology_files()
    assert boundary_scan.decision_making_files()


def test_the_dependency_walk_is_not_vacuous() -> None:
    """A walk that resolved nothing would report no broker just as confidently."""
    reachable = boundary_scan.reachable_distributions()
    assert boundary_scan.DISTRIBUTION in reachable
    assert "liq-core" in reachable, (
        "the dependency walk did not reach a known first-order dependency, so "
        "it is resolving nothing and proving nothing"
    )


def test_the_detector_would_catch_a_violation() -> None:
    """The predicates are exercised, not merely defined.

    Without this, a mistake in `covers` would make every assertion above pass
    for as long as the mistake survived.
    """
    assert boundary_scan.covers("liq.sim", "liq.sim")
    assert boundary_scan.covers("liq.sim.fills", "liq.sim")
    assert not boundary_scan.covers("liq.simulation", "liq.sim")
    assert not boundary_scan.covers("liq.core", "liq.sim")

    forbidden = (
        "liq.sim",
        "liq.gp",
        "liq.risk",
        "liq.data",
        "liq.features",
        "liq.datasets",
        "liq.runner",
        "liq.validation",
    )
    for module in forbidden:
        assert boundary_scan.covers(module, "liq")
        assert not boundary_scan.covered_by_any(
            module, boundary_scan.ALLOWED_FIRST_PARTY
        )
    assert boundary_scan.covered_by_any(
        "liq.core.bars", boundary_scan.ALLOWED_FIRST_PARTY
    )
    assert boundary_scan.covered_by_any(
        "liq.evolution.ecology.types", boundary_scan.ALLOWED_FIRST_PARTY
    )

    assert boundary_scan.covered_by_any(
        "liq.live", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert boundary_scan.covered_by_any(
        "liq.trade.runner", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert boundary_scan.covered_by_any(
        "ccxt.binance", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert not boundary_scan.covered_by_any(
        "polars", boundary_scan.BROKER_IMPORT_PREFIXES
    )


def test_adapters_are_exempt_from_containment_but_not_from_the_broker_rule() -> None:
    """The two scans have deliberately different scopes; pin the difference."""
    assert boundary_scan.ADAPTER_DIRECTORY not in boundary_scan.ALLOWED_FIRST_PARTY
    if not boundary_scan.ECOLOGY_ROOT.is_dir():
        return
    adapter_tree = boundary_scan.ECOLOGY_ROOT / boundary_scan.ADAPTER_DIRECTORY
    scanned_everywhere = set(boundary_scan.ecology_files())
    contained = set(boundary_scan.decision_making_files())
    assert all(not path.is_relative_to(adapter_tree) for path in contained)
    assert contained <= scanned_everywhere


# ---------------------------------------------------------------------------
# One evolution engine: the platform's machinery is composed, never redefined.
#
# The static scans above and below cannot see a loop with its own tournament or
# an adapter holding the archive while writing to a list beside it. So the
# archive and the selection are also *traced*: the platform's object or callable
# is intercepted where the adapter looks it up, and what it received and handed
# back is compared with what the adapter reports. Equal outputs are not
# evidence — a faithful copy of a platform primitive returns the same answer —
# so only an intercepted call counts. Every wrong build the anti-divergence
# ruling names that falls to this module is seeded below and must be caught.
# ---------------------------------------------------------------------------


def _offers(population: object) -> list[ArchiveEntry]:
    """Offers that place, re-offer and evict, over the contract's three agents.

    The second and third agents read the same features and want the same
    exposure, so they share a cell: the third agent's higher figure dominates
    the second's, and the platform's archive must evict it. A record that
    never evicts — a list beside the archive — then disagrees with the archive.
    """
    first, second, third = population.agent_ids()  # type: ignore[attr-defined]
    return [
        offer(population, first, 0.0),  # type: ignore[arg-type]
        offer(population, second, 0.0),  # type: ignore[arg-type]
        offer(population, second, 0.0),  # type: ignore[arg-type]
        offer(population, third, 1.0),  # type: ignore[arg-type]
    ]


def archive_trace_problems(population: object) -> list[str]:
    """What stops ``population``'s diversity record being the platform archive.

    Three clauses, each observed rather than reported by the code under test:
    the record *is* a ``QDArchive`` by identity; every entry offered reaches
    that instance's ``insert``, intercepted on the instance the adapter looks
    it up on; and every entry reported is supplied by that instance's
    ``elites``, again intercepted, and equals what the archive holds.

    Raises:
        AttributeError: If ``population`` holds no ``archive`` at all.
    """
    archive = population.archive  # type: ignore[attr-defined]
    if type(archive) is not QDArchive:
        return [
            f"the diversity record is a {type(archive).__qualname__}, not the "
            "platform's QDArchive"
        ]
    problems: list[str] = []
    offers = _offers(population)
    with (
        mock.patch.object(archive, "insert", wraps=archive.insert) as inserted,
        mock.patch.object(archive, "elites", wraps=archive.elites) as supplied,
    ):
        for entry in offers:
            population.record(entry)  # type: ignore[attr-defined]
        reported = population.recorded()  # type: ignore[attr-defined]
        received = [
            call.args[0] if call.args else call.kwargs["individual"]
            for call in inserted.call_args_list
        ]
        if received != offers:
            problems.append(
                f"{len(offers)} entries offered, {len(received)} reached the "
                "archive's insert in that order"
            )
        if supplied.call_count == 0:
            problems.append("the entries reported were not read from the archive")
    held = tuple(archive.elites())
    if reported != held:
        problems.append(
            f"{len(reported)} entries reported, the archive holds {len(held)}"
        )
    return problems


def selection_trace_problems(population: object) -> list[str]:
    """What stops ``population``'s parents being the platform selection's.

    The platform's tournament selection is intercepted at the name the adapter
    looks it up by, so a call that reaches it is observed and a copy is not.
    Every parent returned must be one an intercepted call returned, in order.
    """
    platform = array_genome.tournament_select
    returned: list[AgentId] = []

    def intercepted(*args: object, **kwargs: object) -> list[object]:
        chosen = platform(*args, **kwargs)  # type: ignore[arg-type]
        returned.extend(typing.cast("Sequence[AgentId]", chosen))
        return chosen

    agent_ids = population.agent_ids()  # type: ignore[attr-defined]
    with mock.patch.object(array_genome, "tournament_select", intercepted):
        parents = population.select_parents(  # type: ignore[attr-defined]
            fitness=declared_fitness(agent_ids), count=CONFIGURED_SIZE, seed=17
        )
    if not returned:
        return ["no parent was chosen by the platform's selection"]
    if parents != tuple(returned):
        return [
            f"{len(parents)} parents returned, {len(returned)} came from the "
            "platform's selection"
        ]
    return []


def test_the_diversity_record_is_the_platform_archive() -> None:
    """The record is the platform's QDArchive, and every entry passes through it."""
    population = an_array_genome_population()
    assert type(population.archive) is QDArchive  # type: ignore[attr-defined]
    assert archive_trace_problems(population) == []
    directions = population.archive.objective_directions  # type: ignore[attr-defined]
    assert directions
    assert set(directions) <= set(typing.get_args(evaluation_schema.ObjectiveDirection))


def test_selection_reaches_the_platform_selection() -> None:
    """Every parent is one the platform's selection returned, observed at its call."""
    population = an_array_genome_population(agent_count=DRIFTED_SIZES[0])
    assert selection_trace_problems(population) == []


def test_no_configuration_key_counts_generations() -> None:
    """No key reachable from the ecology's configuration names a generation count.

    Lexical, and stated as such: a schedule under another name is not caught
    here (see the scan's docstring).
    """
    assert boundary_scan.generation_count_keys(EcologyConfig) == []
    walked = boundary_scan.configuration_classes_walked(EcologyConfig)
    assert {
        EcologyConfig.__name__,
        PopulationConfig.__name__,
        LearningConfig.__name__,
    } <= set(walked), "the walk did not reach the nested configuration"


def test_objective_directions_come_from_the_single_schema() -> None:
    """No direction is spelled, converted or sourced inside the ecology."""
    violations = boundary_scan.direction_definitions_here()
    assert violations == [], (
        f"an objective direction is defined inside the ecology: {violations}. "
        "Directions and their loss form are the platform's single schema's, "
        f"{boundary_scan.DIRECTION_SCHEMA_MODULE}."
    )


# --- Seeded wrong builds: each must be caught. ------------------------------


@dataclass(frozen=True)
class _Seeded:
    """A nested configuration holding a seeded generation count."""

    n_generations: int = 1
    regeneration_rate: float = 0.0


@dataclass(frozen=True)
class _SeededAgain:
    """A second, reached only through a tuple and an optional."""

    max_epochs: int = 1


@dataclass(frozen=True)
class _SeededRoot:
    """A root configuration whose violations are nested, not on its surface."""

    run_id: str
    nested: _Seeded | None = None
    more: tuple[_SeededAgain, ...] = ()
    replay: bool = False


def test_the_generation_key_detector_catches_a_seeded_key() -> None:
    """Seeded keys at three depths are caught; a key merely containing the word is not."""
    assert boundary_scan.generation_count_keys(_SeededRoot) == [
        "_SeededRoot.more.max_epochs",
        "_SeededRoot.nested.n_generations",
        "_SeededRoot.replay",
    ]
    with pytest.raises(TypeError, match="not a configuration class"):
        boundary_scan.generation_count_keys(dict)


@pytest.mark.parametrize(
    ("source", "caught"),
    [
        ('Direction = typing.Literal["maximize", "minimize"]\n', 2),
        ('BETTER = "maximize"\n', 1),
        ('def worse(d):\n    return d == "minimize"\n', 1),
        ("def as_loss_form(value):\n    return -value\n", 1),
        ("from liq.gp.evolution.qd_archive import ObjectiveDirection\n", 1),
        ('"""Higher is better: "maximize", as the schema spells it."""\n', 0),
        (
            "from liq.evolution.fitness.evaluation_schema import "
            "ObjectiveDirection, to_loss_form\n",
            0,
        ),
    ],
)
def test_the_direction_detector_catches_a_seeded_direction(
    tmp_path: Path, source: str, caught: int
) -> None:
    """A direction spelled, converted or imported from elsewhere is caught."""
    module = tmp_path / "seeded.py"
    module.write_text(textwrap.dedent(source), encoding="utf-8")
    found = boundary_scan.direction_definitions_in([module], root=tmp_path)
    assert len(found) == caught, found


class _Delegating:
    """A wrong build: the real adapter's agents, under another record."""

    def __init__(self) -> None:
        self.real = an_array_genome_population()

    def agent_ids(self) -> tuple[AgentId, ...]:
        return self.real.agent_ids()

    def genome(self, agent_id: AgentId) -> object:
        return self.real.genome(agent_id)

    def descriptor(self, agent_id: AgentId) -> object:
        return self.real.descriptor(agent_id)


def _a_platform_archive() -> QDArchive:
    """A QDArchive a wrong build holds, over the adapter's two unit axes."""
    return QDArchive(
        n_dims=2,
        bins_per_dim=3,
        descriptor_bounds=((0.0, 1.0), (0.0, 1.0)),
        objective_directions=["maximize"],
    )


class _SideListOnly(_Delegating):
    """Holds a QDArchive but offers to, and reports from, a list beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.archive = _a_platform_archive()
        self._kept: list[ArchiveEntry] = []

    def record(self, entry: ArchiveEntry) -> bool:
        if entry in self._kept:
            return False
        self._kept.append(entry)
        return True

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._kept)


class _MirroredSideList(_SideListOnly):
    """Offers every entry to the archive, but reports from its own list."""

    def record(self, entry: ArchiveEntry) -> bool:
        self.archive.insert(entry, (0.0,), (0.0, 0.0))
        return super().record(entry)


class DiversityRecord:
    """A second archive under a name the class-name scan does not match."""

    def __init__(self) -> None:
        self.objective_directions = ["maximize"]
        self._cells: dict[tuple[float, ...], ArchiveEntry] = {}

    def insert(self, entry: ArchiveEntry, score: float) -> bool:
        cell = tuple(sorted(entry.descriptor.values.values()))
        held = self._cells.get(cell)
        if held is not None and sum(held.objectives.values()) >= score:
            return False
        self._cells[cell] = entry
        return True

    def elites(self) -> list[ArchiveEntry]:
        return list(self._cells.values())


class _HeldDiversityRecord(_Delegating):
    """Keeps its record in a ``DiversityRecord`` with cell eviction."""

    def __init__(self) -> None:
        super().__init__()
        self.archive = DiversityRecord()

    def record(self, entry: ArchiveEntry) -> bool:
        return self.archive.insert(entry, sum(entry.objectives.values()))

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self.archive.elites())


@pytest.mark.parametrize(
    "wrong_build", [_SideListOnly, _MirroredSideList, _HeldDiversityRecord]
)
def test_the_archive_trace_catches_a_second_record(wrong_build: type) -> None:
    """A side list, a mirrored side list and a renamed archive are each caught.

    ``DiversityRecord`` passes the class-name scan by construction — which is
    why the record is also held to identity and traced.
    """
    assert boundary_scan.ARCHIVE_CLASS_SUFFIX not in DiversityRecord.__name__
    assert archive_trace_problems(wrong_build()) != []


def test_the_archive_trace_catches_the_null_populations_record() -> None:
    """The stand-in's dict archive is never the platform's, whatever it is called."""
    with pytest.raises(AttributeError):
        archive_trace_problems(NullAgentPopulation())


class _InlineTopK(_Delegating):
    """Chooses its own parents: the top of the fitness, never the platform's."""

    def select_parents(
        self, *, fitness: Mapping[AgentId, float], count: int, seed: int
    ) -> tuple[AgentId, ...]:
        ranked = sorted(fitness, key=fitness.__getitem__, reverse=True)
        return tuple(ranked[index % len(ranked)] for index in range(count))


class _FaithfulCopy(_Delegating):
    """Runs a byte-identical copy of the platform's selection, not the platform's.

    Its parents equal the real adapter's for the same seed, which is exactly
    why equal outputs are not evidence: only an intercepted call counts.
    """

    def select_parents(
        self, *, fitness: Mapping[AgentId, float], count: int, seed: int
    ) -> tuple[AgentId, ...]:
        original = platform_selection.tournament_select
        copied = types.FunctionType(
            original.__code__,
            original.__globals__,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        with mock.patch.object(array_genome, "tournament_select", copied):
            return self.real.select_parents(fitness=fitness, count=count, seed=seed)


@pytest.mark.parametrize("wrong_build", [_InlineTopK, _FaithfulCopy])
def test_the_selection_trace_catches_a_selection_written_here(
    wrong_build: type,
) -> None:
    """Inline top-k and a faithful copy of the platform's selection are caught."""
    population = wrong_build()
    population.real = an_array_genome_population(agent_count=DRIFTED_SIZES[0])
    assert selection_trace_problems(population) != []


def test_a_faithful_copy_returns_what_the_platform_returns() -> None:
    """The copy above is caught by interception alone; its output is identical."""
    copy = _FaithfulCopy()
    copy.real = an_array_genome_population(agent_count=DRIFTED_SIZES[0])
    fitness = declared_fitness(copy.real.agent_ids())
    assert copy.select_parents(
        fitness=fitness, count=CONFIGURED_SIZE, seed=17
    ) == copy.real.select_parents(fitness=fitness, count=CONFIGURED_SIZE, seed=17)


# ---------------------------------------------------------------------------
# Variation: every birth reaches the platform's array-genome primitives.
#
# Traced like selection, and for the same reason: a faithful copy of a platform
# primitive produces the platform's offspring, so only a call intercepted where
# the adapter looks the primitive up counts. The adapter may hold a primitive by
# name or hold the platform's module and reach into it; both are intercepted.
# ---------------------------------------------------------------------------

#: The module the platform's array-genome variation primitives are defined in.
PLATFORM_VARIATION = "liq.gp.evolution.array_operators"


def _defined_in_the_platform_variation(value: object) -> bool:
    """Whether ``value`` is a callable the platform's variation module defines."""
    return callable(value) and (
        getattr(value, "__module__", None) == PLATFORM_VARIATION
    )


def variation_trace_problems(population: object) -> list[str]:
    """What stops every birth of ``population`` reaching the platform's variation.

    Every primitive defined in :data:`PLATFORM_VARIATION` is wrapped at the
    name the adapter module looks it up by — a name it holds directly, or an
    attribute of the platform module it holds — and each birth of every parent
    shape, at every one of :data:`VARIATION_SEEDS`, must make at least one
    intercepted call.
    """
    calls = [0]

    def counting(primitive: Callable[..., object]) -> Callable[..., object]:
        def intercepted(*args: object, **kwargs: object) -> object:
            calls[0] += 1
            return primitive(*args, **kwargs)

        return intercepted

    births = 0
    silent = 0
    with contextlib.ExitStack() as stack:
        for name, value in list(vars(array_genome).items()):
            if isinstance(value, types.ModuleType):
                if value.__name__ == PLATFORM_VARIATION:
                    for attribute, primitive in list(vars(value).items()):
                        if _defined_in_the_platform_variation(primitive):
                            stack.enter_context(
                                mock.patch.object(value, attribute, counting(primitive))
                            )
            elif _defined_in_the_platform_variation(value):
                stack.enter_context(
                    mock.patch.object(array_genome, name, counting(value))  # type: ignore[arg-type]
                )
        for shape in PARENT_SHAPES:
            parents = parents_of_shape(population, shape)  # type: ignore[arg-type]
            for seed in VARIATION_SEEDS:
                before = calls[0]
                population.spawn(parents=parents, seed=seed)  # type: ignore[attr-defined]
                births += 1
                silent += calls[0] == before
    if silent:
        return [f"{silent} of {births} births made no call into {PLATFORM_VARIATION}"]
    return []


def test_variation_reaches_the_platform_variation() -> None:
    """Every birth calls the platform's variation, observed where it is looked up."""
    problems = variation_trace_problems(an_array_genome_population())
    assert not problems, problems[0]


def _a_platform_primitive() -> Callable[..., float]:
    """A stand-in primitive that says it is defined in the platform's module."""

    def perturbed(value: float, seed: int) -> float:
        return value + (seed + 1) * 1.0e-3

    perturbed.__module__ = PLATFORM_VARIATION
    return perturbed


def _varied(real: object, parents: Sequence[AgentId], perturb: Callable) -> Offspring:
    """The first parent's genome with every gene passed through ``perturb``.

    Built from the parent rather than from the real adapter's birth, so the only
    call a stand-in makes is the one it is written to make.
    """
    parent = real.genome(parents[0])  # type: ignore[attr-defined]
    genes = {name: perturb(value, 0) for name, value in parent.genes.items()}
    return Offspring(
        genome=dataclasses.replace(parent, genes=types.MappingProxyType(genes)),
        mutation_scale=None,
        boundary_firings=types.MappingProxyType(dict.fromkeys(genes, 0)),
    )


class _ThroughAName(_Delegating):
    """Calls a platform primitive the adapter module holds by name."""

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        return _varied(self.real, parents, array_genome.perturbed)  # type: ignore[attr-defined]


class _ThroughTheModule(_Delegating):
    """Calls a platform primitive through the platform module the adapter holds."""

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        module = array_genome.array_operators  # type: ignore[attr-defined]
        return _varied(self.real, parents, module.perturbed)


class _InlineCopy(_Delegating):
    """Varies every gene with a copy of the primitive written here."""

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        def perturbed(value: float, seed: int) -> float:
            return value + (seed + 1) * 1.0e-3

        return _varied(self.real, parents, perturbed)


def test_the_variation_trace_counts_a_call_where_it_is_looked_up() -> None:
    """A primitive held by name and one reached through its module both count."""
    platform_module = types.ModuleType(PLATFORM_VARIATION)
    platform_module.perturbed = _a_platform_primitive()  # type: ignore[attr-defined]
    with (
        mock.patch.object(
            array_genome, "perturbed", _a_platform_primitive(), create=True
        ),
        mock.patch.object(
            array_genome, "array_operators", platform_module, create=True
        ),
    ):
        assert variation_trace_problems(_ThroughAName()) == []
        assert variation_trace_problems(_ThroughTheModule()) == []


def test_the_variation_trace_catches_a_copy_written_here() -> None:
    """A copy of the primitive is caught, though the platform's is in reach."""
    births = len(PARENT_SHAPES) * len(VARIATION_SEEDS)
    silent = [f"{births} of {births} births made no call into {PLATFORM_VARIATION}"]
    with mock.patch.object(
        array_genome, "perturbed", _a_platform_primitive(), create=True
    ):
        assert variation_trace_problems(_InlineCopy()) == silent


# ---------------------------------------------------------------------------
# Every birth's genome is the platform's variation output, traced as data.
#
# The trace above counts calls, so a birth that calls a primitive and then
# overwrites, clamps or redraws what it returned passes it. Here every value a
# birth hands a primitive and every value it hands back is recorded
# (``tests/support/variation_flow.py``), and the genome returned is held to them
# by identity alone: no draw order and no arithmetic.
# ---------------------------------------------------------------------------


def test_each_birth_is_the_platform_variation_output() -> None:
    """Each birth's genome is what the platform's three primitives made of its parents.

    For every parent shape at every seed, over the contract's population and
    over designed founders carrying every gene class: the scale step,
    recombination and mutation are each called once, as the decided operator
    calls them for every shape (one parent, or the same agent twice, recombines
    with itself; it skips nothing); the genes handed over are the genes the
    schema declares varied; mutation's input on non-mask genes is
    recombination's output; mutation's scale is the step's; and every varied
    gene returned is mutation's output, the scale gene the step's scale.
    """
    births = 0
    problems: list[str] = []
    for population in (
        an_array_genome_population(),
        variation_flow.every_gene_class_population(),
    ):
        with variation_flow.traced() as spawned:
            for shape in PARENT_SHAPES:
                parents = parents_of_shape(population, shape)
                for seed in VARIATION_SEEDS:
                    population.spawn(parents=parents, seed=seed)
        births += len(spawned)
        problems.extend(
            problem
            for birth in spawned
            for problem in variation_flow.flow_problems(birth)
        )
    assert births == 2 * len(PARENT_SHAPES) * len(VARIATION_SEEDS)
    assert not problems, (
        f"{problems[0]}; {len(problems)} departures from the platform's output "
        f"over {births} births"
    )


# ---------------------------------------------------------------------------
# Descriptors add no second engine and no hidden provider reach.
#
# Four static detectors, each proven non-vacuous by a seed planted in a copy of
# the real source tree, never in the tree itself:
#
# 1. Transitive reach: no decision-making module reaches the simulator, the
#    evolution engine, the GP stream's descriptor module or its QD orchestrator
#    through any chain of first-party imports, not only directly.
# 2. One clip: the descriptor module holds the ecology's only clip, and nothing
#    the ecology reaches names the GP stream's clamp helpers.
# 3. Record calls: no module but the loop calls the population port's
#    ``record``.
# 4. The descriptor vocabulary is read from the descriptor module, never
#    restated as a literal.
# ---------------------------------------------------------------------------

#: The module the seeded transitive reach goes through.
GP_STREAM_DESCRIPTORS = "liq.evolution.fitness.behavior_descriptors"


def _seeded_tree(tmp_path: Path, relative: str, source: str) -> Path:
    """A copy of the real source tree with ``source`` appended to one ecology module.

    Returns the copy's ``src`` root. The seed is planted in the copy only, so
    the detector is shown catching it without the real tree ever holding it.
    """
    src_root = tmp_path / "src"
    shutil.copytree(
        boundary_scan.SRC_ROOT,
        src_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    module = src_root.joinpath(*boundary_scan.ECOLOGY_PACKAGE.split("."), relative)
    with module.open("a", encoding="utf-8") as handle:
        handle.write("\n" + textwrap.dedent(source))
    return src_root


def _ecology_root(src_root: Path) -> Path:
    return src_root.joinpath(*boundary_scan.ECOLOGY_PACKAGE.split("."))


def test_no_decision_making_module_reaches_a_provider_transitively() -> None:
    """No chain of first-party imports carries a provider into code that decides.

    A static closure over module files. ``liq/evolution/__init__.py`` loads
    the simulator at run time whatever the ecology imports; that is the GP
    stream's package initialiser, not an ecology dependency, and a green scan
    here does not say the process never loads the simulator. Two more run-time
    reaches are outside it for the same reason: ``ecology/adapters/__init__.py``
    re-exports ``LiqSimExecutionAdapter``, so importing any ``adapters``
    submodule executes ``adapters.liq_sim``; and ``adapters/array_genome.py``'s
    import of ``liq.evolution.fitness.evaluation_schema`` executes
    ``liq/evolution/fitness/__init__.py``, which loads ``liq.sim`` and the GP
    stream's descriptor module. Both are ancestor initialisers, not a module's
    own dependency, and neither is repaired here.
    """
    violations = boundary_scan.decision_making_transitive_reach()
    assert violations == [], (
        f"a provider is reached through a chain of first-party imports: "
        f"{violations}. A module that imports a module that imports the "
        "provider has the provider as surely as one importing it directly."
    )
    reached = boundary_scan.transitive_closure(boundary_scan.decision_making_files())
    assert boundary_scan.ECOLOGY_ROOT / "descriptors.py" in reached


def test_the_transitive_reach_catches_a_seeded_descriptor_import(
    tmp_path: Path,
) -> None:
    """The GP stream's descriptor module, imported by the descriptors, is caught.

    Red with the chain that carries the simulator in, not merely the import.
    """
    src_root = _seeded_tree(
        tmp_path,
        "descriptors.py",
        f"from {GP_STREAM_DESCRIPTORS} import extract_behavior_descriptors\n",
    )
    violations = boundary_scan.decision_making_transitive_reach(
        ecology_root=_ecology_root(src_root), src_root=src_root
    )
    descriptors = f"{boundary_scan.ECOLOGY_PACKAGE}.descriptors"
    assert f"{descriptors} imports {GP_STREAM_DESCRIPTORS}" in violations, violations
    through = [
        violation
        for violation in violations
        if violation.startswith(f"{descriptors} reaches liq.sim")
    ]
    assert through, violations
    assert f"{descriptors} -> {GP_STREAM_DESCRIPTORS} -> liq.sim" in through[0]


@pytest.mark.parametrize(
    ("relative", "source", "reached"),
    [
        (
            "steady_state.py",
            "from .adapters.liq_sim import LiqSimExecutionAdapter\n",
            "liq.sim",
        ),
        (
            "steady_state.py",
            "from liq.evolution.ecology.adapters import ArrayGenomePopulation\n",
            "liq.gp.evolution",
        ),
        (
            "agent.py",
            "import liq.evolution.qd.orchestrator\n",
            "liq.evolution.qd.orchestrator",
        ),
        (
            "agent.py",
            "from liq.evolution import fitness\n",
            "liq.sim",
        ),
    ],
)
def test_the_transitive_reach_follows_every_import_form(
    tmp_path: Path, relative: str, source: str, reached: str
) -> None:
    """A relative import, a package re-export, a plain import and a package named
    as the dependency are each followed."""
    src_root = _seeded_tree(tmp_path, relative, source)
    violations = boundary_scan.decision_making_transitive_reach(
        ecology_root=_ecology_root(src_root), src_root=src_root
    )
    module = f"{boundary_scan.ECOLOGY_PACKAGE}.{relative.removesuffix('.py')}"
    assert any(
        violation.startswith(f"{module} ") and reached in violation
        for violation in violations
    ), violations


def test_no_ecology_module_reaches_a_provider_transitively() -> None:
    """Outside the two modules that hold the provider, no chain carries one in.

    Sources are every ecology module, adapters included, except
    ``adapters/liq_sim.py`` and ``adapters/__init__.py``; those two are still
    followed when a source names them as a dependency. No source reaches the
    simulator, the GP stream's descriptor module or its QD orchestrator, and
    none but ``adapters/array_genome.py`` reaches the evolution engine.

    Static, over module files, and a green scan does not say a module never
    loads the simulator at run time. Three run-time reaches are outside it,
    each pre-existing and not repaired here:

    1. ``liq/evolution/__init__.py`` loads ``liq.sim`` for any
       ``liq.evolution.*`` import — the GP stream's package initialiser.
    2. ``ecology/adapters/__init__.py`` re-exports ``LiqSimExecutionAdapter``,
       so importing any ``adapters`` submodule executes ``adapters.liq_sim``;
       consumers rely on that re-export.
    3. ``adapters/array_genome.py``'s import of
       ``liq.evolution.fitness.evaluation_schema`` executes
       ``liq/evolution/fitness/__init__.py``, which loads ``liq.sim`` and the
       GP stream's descriptor module.

    Each is an ancestor initialiser Python executes, not the importing
    module's own dependency.
    """
    violations = boundary_scan.ecology_transitive_reach()
    assert violations == [], (
        f"a provider is reached through a chain of first-party imports: "
        f"{violations}. Only the execution adapter holds the simulator and only "
        "the population adapter holds the evolution engine."
    )
    sources = boundary_scan.ecology_reach_sources()
    codec = boundary_scan.ECOLOGY_ROOT / "adapters" / "archive_codec.py"
    assert codec in sources
    assert boundary_scan.ECOLOGY_ROOT / "adapters" / "null.py" in sources
    for exempt in boundary_scan.REACH_EXEMPT_SOURCES:
        assert boundary_scan.ECOLOGY_ROOT / exempt not in sources


@pytest.mark.parametrize(
    ("source", "reached", "through"),
    [
        (
            "from liq.evolution.ecology.adapters import LiqSimExecutionAdapter\n",
            "liq.sim",
            "adapters.liq_sim",
        ),
        (
            "from liq.evolution.ecology.adapters import ArrayGenomePopulation\n",
            "liq.gp.evolution",
            "adapters.array_genome",
        ),
    ],
)
def test_the_full_scope_reach_catches_a_seeded_re_export(
    tmp_path: Path, source: str, reached: str, through: str
) -> None:
    """A name taken from the adapters' re-export makes it, and what it holds, a dependency."""
    src_root = _seeded_tree(tmp_path, "adapters/null.py", source)
    violations = boundary_scan.ecology_transitive_reach(
        ecology_root=_ecology_root(src_root), src_root=src_root
    )
    adapters = f"{boundary_scan.ECOLOGY_PACKAGE}.adapters"
    chain = (
        f"{adapters}.null -> {adapters} -> {boundary_scan.ECOLOGY_PACKAGE}.{through}"
    )
    assert any(
        violation.startswith(f"{adapters}.null reaches {reached}")
        and f"{chain} -> {reached}" in violation
        for violation in violations
    ), violations


def test_a_submodule_taken_from_its_package_follows_only_the_submodule() -> None:
    """``from …adapters import archive_codec`` names the codec, not the re-export."""
    named = boundary_scan.named_modules(
        boundary_scan.ECOLOGY_ROOT / "adapters" / "array_genome.py"
    )
    adapters = f"{boundary_scan.ECOLOGY_PACKAGE}.adapters"
    assert f"{adapters}.archive_codec" in named
    assert adapters not in named


def test_the_ecology_holds_one_clip_and_reaches_no_other_clamp() -> None:
    """Beta's clip is the ecology's only one, and no GP-stream clamp is in reach."""
    assert boundary_scan.clips_outside_the_descriptor_module() == []
    reached = boundary_scan.clamp_helpers_reached()
    assert reached == [], (
        f"the GP stream's clamp helpers are reached from the ecology: {reached}. "
        "A value outside its bounds is an exclusion here, never pulled inside."
    )


@pytest.mark.parametrize(
    ("relative", "source", "caught"),
    [
        ("agent.py", "x = np.clip(x, 0.0, 1.0)\n", "clips"),
        ("adapters/archive_codec.py", "x = min(max(x, 0.0), 1.0)\n", "clips"),
        ("adapters/array_genome.py", "x = arr.clip(0.0, 1.0)\n", "clips"),
        (
            "learning_update.py",
            "x = np.minimum(np.maximum(x, 0.0), 1.0)\n",
            "clips",
        ),
        ("features.py", "def _clamp(value):\n    return value\n", "_clamp"),
        ("descriptors.py", "y = np.clip(y, 0.0, 1.0)\n", "clips"),
        (
            "adapters/archive_codec.py",
            "from liq.evolution.qd.orchestrator import _extract_descriptors\n",
            "_extract_descriptors",
        ),
        (
            "steady_state.py",
            f"from {GP_STREAM_DESCRIPTORS} import _clamp\n",
            "_clamp",
        ),
    ],
)
def test_the_clip_detector_catches_a_seeded_clamp(
    tmp_path: Path, relative: str, source: str, caught: str
) -> None:
    """A second clip, anywhere in the ecology or in the descriptor module, is caught."""
    src_root = _seeded_tree(tmp_path, relative, source)
    roots = {"ecology_root": _ecology_root(src_root), "src_root": src_root}
    found = boundary_scan.clips_outside_the_descriptor_module(
        ecology_root=roots["ecology_root"]
    ) + boundary_scan.clamp_helpers_reached(**roots)
    assert any(caught in violation for violation in found), found


def test_a_one_sided_floor_is_not_a_clip(tmp_path: Path) -> None:
    """A floor guarding a division bounds one side only and is not counted."""
    src_root = _seeded_tree(
        tmp_path, "agent.py", "x = np.maximum(x, 1.0e-12)\ny = max(y, 0.0)\n"
    )
    assert (
        boundary_scan.clips_outside_the_descriptor_module(
            ecology_root=_ecology_root(src_root)
        )
        == []
    )


def test_only_the_loop_calls_the_population_record() -> None:
    """No module but the loop offers to the diversity record."""
    violations = boundary_scan.record_calls_outside_the_loop()
    assert violations == [], (
        f"the population port's record is called outside the loop: {violations}. "
        "Only the loop offers; never the composition root, never an adapter on "
        "its own initiative."
    )


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("adapters/archive_codec.py", "population.record(entry)\n"),
        ("adapters/array_genome.py", "self.record(entry)\n"),
        ("adapters/null.py", 'getattr(population, "record")(entry)\n'),
        ("driver.py", "record = population.record\nrecord(entry)\n"),
        ("account_state.py", "self.log.record(entry)\n"),
    ],
)
def test_the_record_call_detector_catches_a_seeded_offer(
    tmp_path: Path, relative: str, source: str
) -> None:
    """An offer from an adapter or a boundary module is caught, however spelled."""
    src_root = _seeded_tree(tmp_path, relative, source)
    violations = boundary_scan.record_calls_outside_the_loop(
        ecology_root=_ecology_root(src_root)
    )
    assert any(violation.startswith(f"{relative}:") for violation in violations), (
        violations
    )


def test_the_record_call_detector_passes_the_loop_and_the_run_log(
    tmp_path: Path,
) -> None:
    """The loop's own offer and a run-log line are not offers outside the loop."""
    src_root = _seeded_tree(tmp_path, "steady_state.py", "population.record(entry)\n")
    log_line = 'log.record("bar_excluded", bar_ts=as_of, stage="feature")\n'
    with (_ecology_root(src_root) / "driver.py").open("a", encoding="utf-8") as h:
        h.write("\n" + log_line)
    assert (
        boundary_scan.record_calls_outside_the_loop(
            ecology_root=_ecology_root(src_root)
        )
        == []
    )


def test_the_descriptor_version_is_read_never_restated() -> None:
    """The vocabulary is spelled once, in the descriptor module, and read elsewhere."""
    violations = boundary_scan.descriptor_version_restated(declared_axes.VOCABULARY)
    assert violations == [], (
        f"the descriptor vocabulary is restated: {violations}. Read "
        "``descriptors.VOCABULARY``; a second spelling is a second version that "
        "can drift from the computation it names."
    )


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("adapters/array_genome.py", f'SCHEMA = "{declared_axes.VOCABULARY}"\n'),
        (
            "adapters/archive_codec.py",
            f'def check(v):\n    return v == "{declared_axes.VOCABULARY}"\n',
        ),
        ("steady_state.py", f'm = f"under {declared_axes.VOCABULARY} only"\n'),
        ("descriptors.py", f'AGAIN = "{declared_axes.VOCABULARY}"\n'),
    ],
)
def test_the_version_detector_catches_a_restated_version(
    tmp_path: Path, relative: str, source: str
) -> None:
    """A second spelling, in another module or the descriptor module, is caught."""
    src_root = _seeded_tree(tmp_path, relative, source)
    violations = boundary_scan.descriptor_version_restated(
        declared_axes.VOCABULARY, ecology_root=_ecology_root(src_root)
    )
    assert any(violation.startswith(f"{relative}:") for violation in violations), (
        violations
    )


def test_a_docstring_naming_the_version_is_not_a_restatement(tmp_path: Path) -> None:
    """Prose that names the vocabulary spells nothing a computation reads."""
    src_root = _seeded_tree(
        tmp_path,
        "agent.py",
        f'def f():\n    """Described under {declared_axes.VOCABULARY}."""\n',
    )
    assert (
        boundary_scan.descriptor_version_restated(
            declared_axes.VOCABULARY, ecology_root=_ecology_root(src_root)
        )
        == []
    )


def test_the_descriptor_scans_cover_the_archive_codec() -> None:
    """The codec is read by every descriptor-era scan, not skipped as an adapter."""
    codec = boundary_scan.ECOLOGY_ROOT / "adapters" / "archive_codec.py"
    assert codec.is_file()
    assert codec in boundary_scan.ecology_files()
    assert codec in boundary_scan.transitive_closure(boundary_scan.ecology_files())
    assert codec not in boundary_scan.decision_making_files()
    assert codec in boundary_scan.descriptor_scan_files()
