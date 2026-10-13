"""What each birth handed the platform's variation primitives, and what they handed back.

A birth's genome is claimed to be the platform's variation output: a scale step,
then recombination, then mutation, each applied by a primitive of
``liq.gp.evolution.array_operators``. Counting calls cannot tell that from a
birth that calls a primitive and then overwrites, clamps or redraws what it
returned. So this module records the data: every call each birth makes into the
three primitives, intercepted where the population adapter looks them up — a
name it holds directly, or an attribute of the platform module it holds — with
the arguments each call was handed and the value it returned, copied at the
call so a later write to the same array cannot rewrite the record.

Two further things are recorded because a check of the rule at the bounds needs
them, and neither is computed here:

* **The standard-normal draws.** The generator each primitive is handed is
  swapped, for that call only, for one over the *same* bit generator that
  records what its ``standard_normal`` returned. The draws are therefore the
  primitive's own, in the order it made them, and the generator's state after
  the call is what it would have been without the record.
* **The gene names, in the positions the arrays hold them.** The genes a birth
  varies are the first parent's genes in the store's column order
  (:func:`~liq.evolution.ecology.agent_genes._gene_layout`), less the strategy
  genes :mod:`~liq.evolution.ecology.genome_schema` declares. The *order* is
  read from the store, not restated here. The *set* is checked against
  :func:`~liq.evolution.ecology.genome_schema.declared` over the store's
  feature vocabulary, less the strategy genes and, for a population that does
  not learn, less :data:`UNLEARNED_GENES`; so a read of the parent's genes that
  dropped or added a gene does not agree with this record by construction.

Nothing here restates a primitive's arithmetic, its draw order or the order in
which a birth calls the primitives. :func:`flow_problems` compares only
identities between recorded values.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import numpy as np

from liq.evolution.ecology import genome_schema
from liq.evolution.ecology.adapters import ArrayGenomePopulation, array_genome
from liq.evolution.ecology.agent import MASK_PREFIX, PopulationState
from liq.evolution.ecology.agent_genes import (
    FORGETTING_FACTOR,
    MUTATION_SCALE,
    _gene_layout,
)
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.types import AgentId, Offspring

#: The module the platform's array-genome variation primitives are defined in.
PLATFORM_VARIATION = "liq.gp.evolution.array_operators"

#: The three primitives the decided operator composes a birth from.
SCALE_STEP = "lognormal_scale_step"
CROSSOVER = "uniform_crossover"
MUTATION = "gaussian_mutation"
PRIMITIVES = (SCALE_STEP, CROSSOVER, MUTATION)

#: How many times each primitive is called per ``spawn`` under the decided
#: operator: the scale step, recombination and mutation, once each, for every
#: parent shape. The adapter's ``spawn`` calls each unconditionally — one
#: parent, or the same agent twice, recombines with itself rather than skipping
#: recombination — so no shape skips one.
CALLS_PER_SPAWN: Mapping[str, int] = types.MappingProxyType(
    dict.fromkeys(PRIMITIVES, 1)
)


class RecordingGenerator(np.random.Generator):
    """A generator over another's bit generator that records its normal draws."""

    def __init__(self, bit_generator: np.random.BitGenerator) -> None:
        super().__init__(bit_generator)
        self.normals: list[float] = []

    def standard_normal(self, *args: Any, **kwargs: Any) -> Any:
        drawn = super().standard_normal(*args, **kwargs)
        self.normals.extend(np.atleast_1d(np.asarray(drawn, dtype=np.float64)))
        return drawn


def _copied(value: object) -> object:
    """``value``, with any array in it copied so later writes cannot reach it."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, array_genome.array_operators.Mutation):
        return type(value)(
            values=value.values.copy(),
            selected=value.selected.copy(),
            fired=value.fired.copy(),
        )
    return value


@dataclass(frozen=True)
class Call:
    """One intercepted call into a platform primitive.

    Attributes:
        primitive: The primitive's name.
        arguments: Every argument, by its parameter name, copied at the call.
        returned: What it returned, copied at the call.
        normals: The standard-normal draws it made, in order.
    """

    primitive: str
    arguments: Mapping[str, object]
    returned: object
    normals: tuple[float, ...]


@dataclass
class Spawned:
    """One ``spawn``: whom it was asked of, every primitive call, what it returned.

    Attributes:
        parents: The parents it was asked for.
        seed: The seed it was drawn under.
        parent_genes: Each parent's genes as the store held them at the call.
        varied: The genes it varies, in the positions its arrays hold them.
        learning: The learning configuration its ranges are declared under:
            the population's own, or the configuration's defaults where it
            declares none.
        learns: Whether the population learns (it declares a
            ``LearningConfig``).
        features: The store's feature vocabulary, fixed at construction.
        calls: Every intercepted primitive call it made, in order.
        offspring: What it returned, once it has returned.
    """

    parents: tuple[AgentId, ...]
    seed: int
    parent_genes: tuple[Mapping[str, float], ...]
    varied: tuple[str, ...]
    learning: LearningConfig
    learns: bool
    features: tuple[str, ...]
    calls: list[Call] = field(default_factory=list)
    offspring: Offspring | None = None

    def only(self, primitive: str) -> Call:
        """The one call this birth made to ``primitive``.

        Raises:
            AssertionError: If it made none, or more than one.
        """
        made = [call for call in self.calls if call.primitive == primitive]
        assert len(made) == 1, f"{len(made)} calls to {primitive}"
        return made[0]


#: The genes only a learning population reads: a population that does not
#: learn neither needs nor varies them.
UNLEARNED_GENES: frozenset[str] = frozenset({FORGETTING_FACTOR})


def varied_genes(genes: Mapping[str, float]) -> tuple[str, ...]:
    """The genes a birth from a parent carrying ``genes`` varies, in column order."""
    names, _ = _gene_layout(_as_genome(genes))
    return tuple(name for name in names if name not in genome_schema.STRATEGY_GENES)


def _as_genome(genes: Mapping[str, float]) -> Any:
    """A stand-in carrying ``genes`` for a layout read that reads only them."""
    return types.SimpleNamespace(genes=genes)


def _signature_names(primitive: Callable[..., object]) -> tuple[str, ...]:
    code = primitive.__code__
    return code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]


@contextlib.contextmanager
def traced() -> Iterator[list[Spawned]]:
    """Record every ``ArrayGenomePopulation.spawn`` and the primitive calls in it.

    Yields the list the births are appended to, in the order ``spawn`` was
    called. A primitive call made outside any ``spawn`` is recorded nowhere.
    """
    births: list[Spawned] = []
    current: list[Spawned] = []

    def intercepting(name: str, primitive: Callable[..., object]) -> Callable:
        parameters = _signature_names(primitive)

        def intercepted(*args: object, **kwargs: object) -> object:
            bound = dict(zip(parameters, args, strict=False)) | kwargs
            generator = bound.get("rng")
            recording = None
            if isinstance(generator, np.random.Generator):
                recording = RecordingGenerator(generator.bit_generator)
                bound["rng"] = recording
            returned = primitive(**bound)
            if current:
                current[-1].calls.append(
                    Call(
                        primitive=name,
                        arguments=types.MappingProxyType(
                            {
                                key: _copied(value)
                                for key, value in bound.items()
                                if key != "rng"
                            }
                        ),
                        returned=_copied(returned),
                        normals=tuple(recording.normals) if recording else (),
                    )
                )
            return returned

        return intercepted

    real_spawn = ArrayGenomePopulation.spawn

    def spawn(
        self: ArrayGenomePopulation, *, parents: Sequence[AgentId], seed: int
    ) -> Offspring:
        genes = tuple(dict(self.state.genome(agent_id).genes) for agent_id in parents)
        record = Spawned(
            parents=tuple(parents),
            seed=seed,
            parent_genes=genes,
            varied=varied_genes(genes[0]) if genes else (),
            learning=self.state.learning or LearningConfig(),
            learns=self.state.learning is not None,
            features=tuple(self.state.feature_names),
        )
        current.append(record)
        try:
            record.offspring = real_spawn(self, parents=parents, seed=seed)
        finally:
            current.pop()
        births.append(record)
        return record.offspring

    with contextlib.ExitStack() as stack:
        for holder_name, value in list(vars(array_genome).items()):
            if (
                isinstance(value, types.ModuleType)
                and value.__name__ == PLATFORM_VARIATION
            ):
                for name in PRIMITIVES:
                    stack.enter_context(
                        mock.patch.object(
                            value, name, intercepting(name, getattr(value, name))
                        )
                    )
            elif (
                callable(value)
                and getattr(value, "__module__", None) == PLATFORM_VARIATION
                and getattr(value, "__name__", None) in PRIMITIVES
            ):
                stack.enter_context(
                    mock.patch.object(
                        array_genome, holder_name, intercepting(value.__name__, value)
                    )
                )
        stack.enter_context(mock.patch.object(ArrayGenomePopulation, "spawn", spawn))
        yield births


#: The features a population carrying every gene class is founded over, and how
#: many founders it holds: declared placeholders, not anybody's population.
EVERY_CLASS_FEATURES = ("first_feature", "second_feature")
EVERY_CLASS_FOUNDERS = 3
EVERY_CLASS_SEED = 0


def every_gene_class_population() -> ArrayGenomePopulation:
    """Founders designed across the declared ranges, carrying every gene class.

    Masks, weights, the entry threshold, the forgetting factor and the scale
    gene, so a birth from them varies every class the schema declares and
    carries a scale gene the step must have set.
    """
    learning = LearningConfig()
    births = array_genome.design_founders(
        features=EVERY_CLASS_FEATURES,
        learning=learning,
        count=EVERY_CLASS_FOUNDERS,
        seed=EVERY_CLASS_SEED,
    )
    return ArrayGenomePopulation(
        state=PopulationState.founded(births, learning=learning),
        descriptor_schema_version="variation-flow-descriptors-1",
    )


def _same(left: object, right: object) -> bool:
    """Whether two recorded values are the same data: equal shape and values."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def flow_problems(birth: Spawned) -> list[str]:
    """What stops one birth's genome being its primitives' output.

    Data-flow identities only, each between two recorded values:

    * each primitive is called :data:`CALLS_PER_SPAWN` times;
    * the varied genes are, as a set, the genes
      :func:`~liq.evolution.ecology.genome_schema.declared` declares over the
      store's feature vocabulary, less the strategy genes and, for a population
      that does not learn, less :data:`UNLEARNED_GENES`;
    * the genes handed over are the genes the schema declares varied: the scale
      step serves that many, recombination receives each parent's non-mask
      varied genes, mutation receives every varied gene, with the schema's
      declared range at each position and the mask genes as its bits;
    * on non-mask positions, the mutation's input is the recombination's output;
    * the mutation's ``scale`` is the scale the step returned;
    * every varied gene of the returned genome is the mutation's output at its
      position, the returned genome carries no other gene but the scale gene,
      and the scale gene, where the first parent carries one, is the step's
      scale.
    """
    label = f"spawn of {birth.parents} under seed {birth.seed}"
    counted = {
        name: sum(call.primitive == name for call in birth.calls) for name in PRIMITIVES
    }
    if counted != dict(CALLS_PER_SPAWN):
        return [
            f"{label}: called the primitives {counted}, the decided operator "
            f"calls them {dict(CALLS_PER_SPAWN)}"
        ]
    problems: list[str] = []
    varied = birth.varied
    is_mask = np.array([name.startswith(MASK_PREFIX) for name in varied])
    first, last = birth.parent_genes[0], birth.parent_genes[-1]
    features = tuple(
        name.removeprefix(MASK_PREFIX)
        for name in varied
        if name.startswith(MASK_PREFIX)
    )
    ranges = genome_schema.declared(features=features, learning=birth.learning)
    unread = frozenset() if birth.learns else UNLEARNED_GENES
    expected = (
        set(genome_schema.declared(features=birth.features, learning=birth.learning))
        - genome_schema.STRATEGY_GENES
        - unread
    )
    if set(varied) != expected:
        return [
            f"{label}: varies {sorted(varied)}; the schema declares "
            f"{sorted(expected)} varied for a population that "
            f"{'learns' if birth.learns else 'does not learn'} over "
            f"{list(birth.features)} (missing {sorted(expected - set(varied))}, "
            f"undeclared {sorted(set(varied) - expected)})"
        ]
    step = birth.only(SCALE_STEP)
    crossover = birth.only(CROSSOVER)
    mutation = birth.only(MUTATION)
    stepped = step.returned
    mutated = mutation.returned

    if step.arguments.get("n") != len(varied):
        problems.append(
            f"{label}: the scale step serves {step.arguments.get('n')!r} genes, "
            f"the schema declares {len(varied)} varied"
        )
    objects = [name for name in varied if not name.startswith(MASK_PREFIX)]
    for argument, parent in (("parent_1", first), ("parent_2", last)):
        expected = [parent[name] for name in objects]
        if not _same(crossover.arguments.get(argument), expected):
            problems.append(
                f"{label}: recombination's {argument} is not the parent's "
                f"non-mask varied genes {objects}"
            )
    handed = mutation.arguments.get("values")
    if np.asarray(handed).shape != (len(varied),):
        problems.append(
            f"{label}: mutation was handed {np.asarray(handed).shape} genes, the "
            f"schema declares {len(varied)} varied: {varied}"
        )
        return problems
    for bound in ("low", "high"):
        declared = [getattr(ranges[name], bound) for name in varied]
        if not _same(mutation.arguments.get(bound), declared):
            problems.append(f"{label}: mutation's {bound} is not the declared ranges")
    bits = mutation.arguments.get("bits")
    if bits is None or not np.array_equal(np.asarray(bits), is_mask):
        problems.append(f"{label}: mutation's bits are not the mask genes")
    if not _same(np.asarray(handed)[~is_mask], crossover.returned):
        problems.append(
            f"{label}: mutation's input on non-mask genes is not recombination's output"
        )
    if mutation.arguments.get("scale") != stepped.scale:  # type: ignore[attr-defined]
        problems.append(
            f"{label}: mutation applied scale {mutation.arguments.get('scale')!r}, "
            f"the step returned {stepped.scale!r}"  # type: ignore[attr-defined]
        )
    offspring = birth.offspring
    genes = {} if offspring is None else dict(offspring.genome.genes)
    for position, name in enumerate(varied):
        produced = float(mutated.values[position])  # type: ignore[attr-defined]
        if genes.get(name) != produced:
            problems.append(
                f"{label}: {name} is {genes.get(name)!r}, the mutation's output "
                f"there is {produced!r}"
            )
    extra = set(genes) - set(varied) - {MUTATION_SCALE}
    if extra:
        problems.append(
            f"{label}: the genome carries {sorted(extra)} beyond its variation"
        )
    if MUTATION_SCALE in first and genes.get(MUTATION_SCALE) != stepped.scale:  # type: ignore[attr-defined]
        problems.append(
            f"{label}: {MUTATION_SCALE} is {genes.get(MUTATION_SCALE)!r}, the step's "
            f"scale is {stepped.scale!r}"  # type: ignore[attr-defined]
        )
    return problems
