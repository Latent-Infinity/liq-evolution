"""Evidence that genes stay inside their declared ranges, and that a crossing is ruled.

Two claims. Every gene a genome carries lies inside the range declared for it —
when founders are designed, when one parent is mutated and when two are
crossed — and the forgetting factor inside the range the learning
configuration declares, read from it and never restated. And where an operator
would carry a gene past a bound, the value produced is the one the declared
boundary rule predicts, and the firing is counted.

The weight genes' range is the part that can be checked against shipped code
today. A ``weight.`` gene is the prior a learned weight starts from, so its
range is whatever keeps a newborn's starting weights inside the norm bound the
admission path enforces: [−c_F, c_F], derived by the oracle from the
configuration and the shipped norm. The cases below hold that derivation to the
admission path itself — its corner admitted, one step past it refused, points
inside it admitted — at every F from 1 to 256, and pin the floating-point
observation the derivation exists for.

Nothing here is market data. Every number is a declared configuration value, a
gene bound or a seeded draw over a declared box.
"""

from __future__ import annotations

import math
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

from liq.evolution import ecology
from liq.evolution.ecology import genome_schema
from liq.evolution.ecology.adapters import ArrayGenomePopulation, array_genome
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    MASK_PREFIX,
    WEIGHT_PREFIX,
    AgentBirth,
    PopulationState,
)
from liq.evolution.ecology.agent_contracts import StartingStateOutsideItsBound
from liq.evolution.ecology.agent_genes import MUTATION_SCALE
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.types import Genome
from tests.support import gene_range_probe as probe
from tests.support import mutation_scale_oracle as scale_oracle
from tests.support import variation_flow

GENE_SCHEMA_VERSION = genome_schema.SCHEMA_VERSION
FEATURE_SCHEMA_VERSION = "gene-ranges-features-1"
MODEL_VERSION = "gene-ranges-model-1"
DESCRIPTOR_SCHEMA_VERSION = "gene-ranges-descriptors-1"

#: The features the range cases are asked over. Five, because five weight genes
#: is the fewest whose correctly rounded quotient the shipped norm refuses, so
#: the weight range under test is one the derivation had to step.
FEATURES = tuple(f"feature_{index}" for index in range(5))

#: Births per operator and parent shape: the variation probe's own N.
SEEDS = range(200)

#: How many founders a designed initial population is asked for.
FOUNDERS = 50

#: How far past a bound a crossing is provoked, as a fraction of the range's
#: width: far enough that reflection, resampling and clipping each land
#: somewhere different, near enough that one fold brings it back.
OVERSHOOT = 0.25

#: Every F the weight bound is held to the admission path at.
WEIGHT_GENE_COUNTS = range(1, 257)

#: The F seeded points inside the weight box are drawn at, and how many.
BOX_GENE_COUNTS = (3, 13, 64)
BOX_POINTS = 200
BOX_SEED = 20261001

#: What the double-rounded start ``b / math.sqrt(31)`` steps to: above the
#: correctly rounded quotient, so outside the bound's definition.
MATH_START_AT_31 = 17.960530202677493


def _birth(agent_id: str, genes: dict[str, float]) -> AgentBirth:
    """One founder carrying exactly ``genes``."""
    return AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=MappingProxyType(genes), schema_version=GENE_SCHEMA_VERSION
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def _weights_at(values: np.ndarray) -> dict[str, float]:
    """A genome reading every one of ``len(values)`` features, priors ``values``."""
    names = tuple(f"feature_{index:03d}" for index in range(len(values)))
    return {
        **{f"{MASK_PREFIX}{name}": 1.0 for name in names},
        **{
            f"{WEIGHT_PREFIX}{name}": float(value)
            for name, value in zip(names, values, strict=True)
        },
        ENTRY_THRESHOLD: 0.0,
        FORGETTING_FACTOR: LearningConfig().forgetting_maximum,
    }


def _admitted(*rows: np.ndarray) -> bool:
    """Whether the shipped admission path founds agents starting from ``rows``."""
    try:
        PopulationState.founded(
            tuple(
                _birth(f"agent-{index}", _weights_at(row))
                for index, row in enumerate(rows)
            ),
            learning=LearningConfig(),
        )
    except StartingStateOutsideItsBound:
        return False
    return True


def _corner(value: float, genes: int) -> np.ndarray:
    """The 1 × ``genes`` all-``value`` row."""
    return np.full(genes, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# The weight genes' bound, held to the shipped admission path.
# ---------------------------------------------------------------------------


def test_the_weight_bound_starts_from_the_correctly_rounded_quotient() -> None:
    """The values the operator ablation ran under, and never above fl(b/√F)."""
    bound = LearningConfig().weight_norm_bound

    assert probe.per_gene_weight_bound(bound, 1) == 100.0
    assert probe.per_gene_weight_bound(bound, 2) == 70.71067811865476
    assert probe.per_gene_weight_bound(bound, 64) == 12.5
    for genes in WEIGHT_GENE_COUNTS:
        assert probe.per_gene_weight_bound(bound, genes) <= probe.quotient_by_root(
            bound, genes
        )


#: Every F the schema's own weight bound is held to the oracle's c_F at.
SCHEMA_WEIGHT_GENE_COUNTS = range(1, 200)


def test_the_declared_weight_range_is_the_decided_c_F() -> None:
    """The bound the genome schema declares is the oracle's c_F at every F, 1 to 199.

    The cases above hold the oracle's derivation to the admission path; this
    holds the schema, which is what a birth's ranges are read from, to that
    derivation. A schema that started its step from ``b / math.sqrt(F)`` admits
    the same corners at most F and differs from c_F at others.
    """
    learning = LearningConfig()
    differing = [
        (genes, declared, decided)
        for genes in SCHEMA_WEIGHT_GENE_COUNTS
        if (declared := ecology.genome_schema.weight_bound(learning, genes))  # type: ignore[attr-defined]
        != (decided := probe.per_gene_weight_bound(learning.weight_norm_bound, genes))
    ]
    assert differing == [], (
        f"the schema's weight bound differs from c_F at {len(differing)} of "
        f"{len(SCHEMA_WEIGHT_GENE_COUNTS)} F, first (F, declared, c_F) = "
        f"{differing[0]!r}"
    )


def test_the_double_rounded_start_is_told_apart() -> None:
    """``b / math.sqrt(F)`` steps to a different bound at 2 and at 31."""
    bound = LearningConfig().weight_norm_bound

    def stepped_from(start: float, genes: int) -> float:
        while probe.corner_norm(start, genes) > bound:
            start = math.nextafter(start, 0.0)
        return start

    assert stepped_from(bound / math.sqrt(31), 31) == MATH_START_AT_31
    assert probe.per_gene_weight_bound(bound, 31) != MATH_START_AT_31
    assert probe.quotient_by_root(bound, 31) < MATH_START_AT_31
    assert stepped_from(bound / math.sqrt(2), 2) < probe.per_gene_weight_bound(bound, 2)


@pytest.mark.parametrize("genes", WEIGHT_GENE_COUNTS)
def test_the_weight_bound_corner_is_admitted_and_one_step_past_it_is_not(
    genes: int,
) -> None:
    """Admitting the corner admits the box; the step past it is refused if stepped."""
    learning = LearningConfig()
    bound = probe.weight_bound_for(learning, genes)

    assert _admitted(_corner(bound, genes))
    assert _admitted(_corner(-bound, genes))
    if bound < probe.quotient_by_root(learning.weight_norm_bound, genes):
        assert not _admitted(_corner(math.nextafter(bound, math.inf), genes))


@pytest.mark.parametrize("genes", BOX_GENE_COUNTS)
def test_seeded_points_inside_the_weight_box_are_admitted(genes: int) -> None:
    """The corner argument rests on the norm's monotonicity; sample the box too."""
    bound = probe.weight_bound_for(LearningConfig(), genes)
    points = np.random.default_rng(BOX_SEED).uniform(
        -bound, bound, size=(BOX_POINTS, genes)
    )

    assert _admitted(*points)


def test_the_unstepped_quotient_is_first_refused_at_five_genes() -> None:
    """The floating-point observation the stepping exists for.

    At three genes the correctly rounded quotient's corner has a norm of
    exactly the bound and is admitted; at five the shipped norm of the
    unstepped corner is one unit in the last place over, and it is refused —
    the first F at which that happens. Eighty-nine F in 1–256 need a step.
    """
    bound = LearningConfig().weight_norm_bound

    def unstepped(genes: int) -> float:
        return probe.quotient_by_root(bound, genes)

    assert probe.corner_norm(unstepped(3), 3) == bound
    assert _admitted(_corner(unstepped(3), 3))
    assert probe.corner_norm(unstepped(5), 5) == 100.00000000000001
    assert not _admitted(_corner(unstepped(5), 5))
    stepped = [
        genes
        for genes in WEIGHT_GENE_COUNTS
        if probe.per_gene_weight_bound(bound, genes) != unstepped(genes)
    ]
    assert stepped[0] == 5
    assert len(stepped) == 89


# ---------------------------------------------------------------------------
# The oracle's own arithmetic, exercised rather than trusted.
# ---------------------------------------------------------------------------


def test_the_correctly_rounded_root_is_exact_and_ties_to_even() -> None:
    """An exact square, both rounding directions, and a tie each way."""
    one_ulp = math.ulp(1.0)
    assert probe.correctly_rounded_root(Fraction(4)) == 2.0
    assert probe.correctly_rounded_root(Fraction(2)) == math.sqrt(2.0)
    halfway_up = Fraction(1) + Fraction(one_ulp) / 2
    assert probe.correctly_rounded_root(halfway_up**2) == 1.0
    halfway_odd = Fraction(1) + 3 * Fraction(one_ulp) / 2
    assert probe.correctly_rounded_root(halfway_odd**2) == 1.0 + 2 * one_ulp
    above_midpoint = Fraction(1) + Fraction(3 * one_ulp) / 4
    assert probe.correctly_rounded_root(above_midpoint**2) == 1.0 + one_ulp


def test_the_reflection_oracle_folds_until_inside() -> None:
    """One fold, several folds, either side, and a value no fold brings in."""
    assert probe.reflected(1.25, 0.0, 1.0) == 0.75
    assert probe.reflected(-0.25, 0.0, 1.0) == 0.25
    assert probe.reflected(3.5, 0.0, 1.0) == 0.5
    assert probe.reflected(0.5, 0.0, 1.0) == 0.5
    assert probe.reflection_agrees(0.75, 1.25, 0.0, 1.0)
    assert not probe.reflection_agrees(1.0, 1.25, 0.0, 1.0)
    with pytest.raises(ValueError, match="cannot be folded"):
        probe.reflected(math.inf, 0.0, 1.0)


def test_the_resampling_oracle_tells_a_redraw_from_anything_else() -> None:
    """A uniform grid passes; a clip, a value outside and a lopsided draw do not."""
    grid = [(index + 0.5) / 200 for index in range(200)]
    assert probe.resampling_problems(grid, 0.0, 1.0) == []
    assert probe.resampling_problems([1.0] * 200, 0.0, 1.0) != []
    assert probe.resampling_problems([*grid[:-1], 1.5], 0.0, 1.0) != []
    assert probe.resampling_problems([value / 2 for value in grid], 0.0, 1.0) != []


def test_the_range_oracle_names_a_gene_outside_or_undeclared() -> None:
    """Inside passes; outside and undeclared are each named."""
    ranges = {"a": probe.DeclaredRange(low=0.0, high=1.0, rule=probe.REFLECTION)}
    assert probe.out_of_range({"a": 1.0}, ranges) == []
    assert probe.out_of_range({"a": 1.5, "b": 0.0}, ranges) == [
        "a = 1.5 outside [0.0, 1.0]",
        "b has no declared range",
    ]


# ---------------------------------------------------------------------------
# The declared ranges and rules, through the genome schema.
# ---------------------------------------------------------------------------


def _declared(schema: object) -> dict[str, probe.DeclaredRange]:
    """The schema's ranges over :data:`FEATURES`, checked against what is not restated."""
    learning = LearningConfig()
    ranges = dict(probe.declared_ranges(schema, features=FEATURES, learning=learning))
    bound = probe.weight_bound_for(learning, len(FEATURES))
    forgetting = ranges[FORGETTING_FACTOR]
    assert (forgetting.low, forgetting.high) == (
        learning.forgetting_minimum,
        learning.forgetting_maximum,
    )
    for name in FEATURES:
        weight = ranges[f"{WEIGHT_PREFIX}{name}"]
        assert (weight.low, weight.high) == (-bound, bound)
        assert f"{MASK_PREFIX}{name}" in ranges
    assert ENTRY_THRESHOLD in ranges
    return ranges


def _on_their_bounds(ranges: dict[str, probe.DeclaredRange]) -> ArrayGenomePopulation:
    """Two founders, one on every lower bound and one on every upper bound."""
    return ArrayGenomePopulation(
        state=PopulationState.founded(
            (
                _birth("low", {n: r.low for n, r in ranges.items()}),
                _birth("high", {n: r.high for n, r in ranges.items()}),
            ),
            learning=LearningConfig(),
        ),
        descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION,
    )


@pytest.mark.parametrize("operation", ["initialised", "mutated", "crossed"])
def test_genes_stay_within_declared_ranges(operation: str) -> None:
    """Designed, mutated from one parent, or crossed from two: always in range."""
    ranges = _declared(ecology.genome_schema)  # type: ignore[attr-defined]
    if operation == "initialised":
        births = array_genome.design_founders(  # type: ignore[attr-defined]
            features=FEATURES, learning=LearningConfig(), count=FOUNDERS, seed=0
        )
        genomes = [birth.genome for birth in births]
    else:
        population = _on_their_bounds(ranges)
        parents = ("low",) if operation == "mutated" else ("low", "high")
        genomes = [
            population.spawn(parents=parents, seed=seed).genome for seed in SEEDS
        ]
    problems = [
        problem
        for genome in genomes
        for problem in probe.out_of_range(genome.genes, ranges)
    ]
    assert problems == [], problems[:10]


def _unchanged(value: float) -> float:
    """A gene held in its own units is provoked in them."""
    return value


def test_boundary_rule_produces_the_value_it_declares() -> None:
    """A crossing provoked at each bound of each ruled gene meets the rule's prediction.

    A crossing is provoked in the coordinates the gene is held in: its own
    units, or, for a gene whose rule is reflection in log space, logarithms —
    a quarter of the width past either end, where one fold brings it back.
    """
    schema = ecology.genome_schema  # type: ignore[attr-defined]
    learning = LearningConfig()
    ruled = {n: r for n, r in _declared(schema).items() if r.rule is not None}
    assert ruled, "no gene declares a boundary rule, so no crossing can be ruled"
    problems: list[str] = []
    for gene, declared in sorted(ruled.items()):
        if declared.rule not in probe.PREDICTED_RULES:
            problems.append(f"{gene} declares {declared.rule!r}, which has no oracle")
            continue
        in_logs = declared.rule == probe.LOG_REFLECTION
        held = (math.log, math.exp) if in_logs else (_unchanged, _unchanged)
        floor, ceiling = held[0](declared.low), held[0](declared.high)
        width = ceiling - floor
        inside = held[1](floor + width / 2)

        def bounded(value: float, seed: int, gene: str = gene) -> tuple[float, int]:
            return probe.apply_declared_rule(
                schema,
                features=FEATURES,
                learning=learning,
                gene=gene,
                value=value,
                seed=seed,
            )

        if bounded(inside, 0) != (inside, 0):
            problems.append(f"{gene}: a value inside its range was moved or counted")
        for crossing in (
            held[1](ceiling + OVERSHOOT * width),
            held[1](floor - OVERSHOOT * width),
        ):
            produced = [bounded(crossing, seed) for seed in SEEDS]
            if any(fired != 1 for _, fired in produced):
                problems.append(
                    f"{gene}: a crossing at {crossing!r} was not counted once"
                )
            values = [value for value, _ in produced]
            if in_logs:
                if not all(
                    probe.log_reflection_agrees(
                        v, crossing, declared.low, declared.high
                    )
                    for v in values
                ):
                    problems.append(f"{gene}: {crossing!r} was not reflected in logs")
            elif declared.rule == probe.REFLECTION:
                if not all(
                    probe.reflection_agrees(v, crossing, declared.low, declared.high)
                    for v in values
                ):
                    problems.append(f"{gene}: {crossing!r} was not reflected")
            else:
                problems.extend(
                    f"{gene}: {problem}"
                    for problem in probe.resampling_problems(
                        values, declared.low, declared.high
                    )
                )
    population = _on_their_bounds(_declared(schema))
    reported = sum(
        population.spawn(parents=("high",), seed=seed).boundary_firings[gene]
        for seed in SEEDS
        for gene in ruled
    )
    if reported == 0:
        problems.append("births from parents on their bounds reported no firing")
    assert problems == []


# ---------------------------------------------------------------------------
# A crossing an operator makes, through ``spawn``, gene by gene.
#
# The case above holds the schema's declared rule to its prediction, but no
# production birth calls that rule: births reach the bounds through the
# platform's primitives, inside ``spawn``. So births are drawn through ``spawn``
# from parents on their bounds and half a founding step inside them, every
# primitive call is recorded (``tests/support/variation_flow.py``), and each
# gene's value is held to the declared rule applied to its pre-image.
# ---------------------------------------------------------------------------

#: How far inside each bound the second population's founders sit, in unit
#: coordinates of each gene's range (log units for the scale gene): half a
#: founding step, so most steps from them stay inside and some cross.
INSIDE_BY = 0.5 * ecology.genome_schema.FOUNDING_SCALE  # type: ignore[attr-defined]


def _inside_their_bounds(
    ranges: dict[str, probe.DeclaredRange],
) -> ArrayGenomePopulation:
    """Two founders, :data:`INSIDE_BY` inside every lower and every upper bound.

    A mask is a bit and sits on its bound. The living range of these two is
    narrower than every declared range, so a rule applied against the living
    population's range rather than the declared one is told apart.
    """

    def placed(at: float) -> dict[str, float]:
        genes: dict[str, float] = {}
        for name, declared in ranges.items():
            if declared.rule is None:
                genes[name] = declared.low if at < 0.5 else declared.high
            elif declared.rule == probe.LOG_REFLECTION:
                floor, ceiling = math.log(declared.low), math.log(declared.high)
                genes[name] = math.exp(floor + at * (ceiling - floor))
            else:
                genes[name] = declared.low + at * (declared.high - declared.low)
        return genes

    return ArrayGenomePopulation(
        state=PopulationState.founded(
            (
                _birth("near-low", placed(INSIDE_BY)),
                _birth("near-high", placed(1.0 - INSIDE_BY)),
            ),
            learning=LearningConfig(),
        ),
        descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION,
    )


def _ruled(unit: float, declared: probe.DeclaredRange) -> float:
    """The declared reflection applied to ``unit``, as the schema words it.

    The schema declares the rule in unit coordinates of the gene's range: a
    value past an end is "folded back, modulo two and mirrored, until inside,
    and mapped back; a fold landing exactly on an end returns the end itself".
    So the pre-image is taken modulo two first, in plain ``math``, and mirrored
    by the oracle's ``gene_range_probe.reflected``. Mirroring a value below zero
    without first taking it modulo two is the same in exact arithmetic but
    rounds differently, and once mapped back onto a narrow range far from zero
    the two land a unit in the last place of the gene's value apart — more than
    the oracle's tolerance, which is set in units of the range's width.
    """
    reduced = math.fmod(unit, 2.0)
    if reduced < 0.0:
        reduced += 2.0
    folded = probe.reflected(reduced, 0.0, 1.0)
    if folded == 0.0:
        return declared.low
    if folded == 1.0:
        return declared.high
    return declared.low + folded * (declared.high - declared.low)


def _crossing_problems(
    birth: variation_flow.Spawned,
    ranges: dict[str, probe.DeclaredRange],
    crossings: dict[str, int],
) -> list[str]:
    """What stops one birth's genes being the declared rule applied to their pre-images.

    The pre-image of a moved gene is its input to the mutation plus the
    mutation's scale times the draw it moved by, in unit coordinates of the
    declared range, where the schema declares the rule (an unmoved gene's is
    its input); of the scale gene, the parents' geometric mean times
    ``exp(z / sqrt(n))``. The draws are the primitives' own, recorded from the
    generator they were handed; nothing is re-drawn from the seed.
    """
    label = f"spawn of {birth.parents} under seed {birth.seed}"
    problems: list[str] = []
    step = birth.only(variation_flow.SCALE_STEP)
    mutation = birth.only(variation_flow.MUTATION)
    varied = birth.varied
    for bound in ("low", "high"):
        handed = [float(value) for value in mutation.arguments[bound]]  # type: ignore[union-attr]
        declared = [getattr(ranges[name], bound) for name in varied]
        if handed != declared:
            problems.append(
                f"{label}: the {bound} bounds handed to the mutation are {handed}, "
                f"the declared ranges' are {declared}"
            )
    scale_range = ranges[MUTATION_SCALE]
    if (step.arguments["low"], step.arguments["high"]) != (
        scale_range.low,
        scale_range.high,
    ):
        problems.append(
            f"{label}: the scale step was bounded by "
            f"[{step.arguments['low']!r}, {step.arguments['high']!r}], the "
            f"declared range is [{scale_range.low!r}, {scale_range.high!r}]"
        )
    offspring = birth.offspring
    assert offspring is not None
    genes = offspring.genome.genes
    firings = offspring.boundary_firings
    returned = mutation.returned
    selected = returned.selected  # type: ignore[attr-defined]
    values = mutation.arguments["values"]
    scale = float(mutation.arguments["scale"])  # type: ignore[arg-type]
    draws = iter(mutation.normals)
    for position, name in enumerate(varied):
        declared = ranges[name]
        if declared.rule is None:
            if firings.get(name) != 0:
                problems.append(
                    f"{label}: {name} has no rule, but fired {firings.get(name)!r}"
                )
            continue
        start = float(values[position])  # type: ignore[index]
        width = declared.high - declared.low
        unit = (start - declared.low) / width
        if selected[position]:
            unit += scale * next(draws)
        crossed = int(not 0.0 <= unit <= 1.0)
        crossings[name] = crossings.get(name, 0) + crossed
        predicted = _ruled(unit, declared)
        if not math.isclose(
            genes[name],
            predicted,
            rel_tol=0.0,
            abs_tol=probe.REFLECTION_TOLERANCE_ULPS * math.ulp(width),
        ):
            problems.append(
                f"{label}: {name} is {genes[name]!r}, the declared rule makes "
                f"{predicted!r} of its pre-image {unit!r} in unit coordinates"
            )
        if firings.get(name) != crossed:
            problems.append(
                f"{label}: {name} reported {firings.get(name)!r} firings, its "
                f"pre-image crossed {crossed} times"
            )
    if MUTATION_SCALE in genes:
        mean = scale_oracle.parents_mean(
            [float(step.arguments["scale_1"]), float(step.arguments["scale_2"])]  # type: ignore[arg-type]
        )
        pre_image = mean * math.exp(
            step.normals[0] / math.sqrt(int(step.arguments["n"]))  # type: ignore[arg-type]
        )
        crossed = int(not scale_range.low <= pre_image <= scale_range.high)
        crossings[MUTATION_SCALE] = crossings.get(MUTATION_SCALE, 0) + crossed
        if not probe.log_reflection_agrees(
            genes[MUTATION_SCALE], pre_image, scale_range.low, scale_range.high
        ):
            problems.append(
                f"{label}: {MUTATION_SCALE} is {genes[MUTATION_SCALE]!r}, not "
                f"{pre_image!r} reflected in logs"
            )
        if firings.get(MUTATION_SCALE) != crossed:
            problems.append(
                f"{label}: {MUTATION_SCALE} reported "
                f"{firings.get(MUTATION_SCALE)!r} firings, its pre-image crossed "
                f"{crossed} times"
            )
    return problems


def test_an_operator_crossing_meets_the_declared_rule_gene_by_gene() -> None:
    """Every gene of every birth through ``spawn`` is the declared rule at its pre-image.

    Births from parents on their bounds and half a founding step inside them,
    alone and crossed, at each of :data:`SEEDS`. The bounds handed to the
    primitive that applies the rule are the declared ranges; every produced
    value is the declared rule applied in plain ``math`` to its pre-image — for a
    linear gene, the pre-image in unit coordinates taken modulo two, mirrored by
    ``gene_range_probe.reflected`` on [0, 1] and mapped back (:func:`_ruled`),
    compared by ``math.isclose`` within ``REFLECTION_TOLERANCE_ULPS`` units in the
    last place of the range's width; for the scale gene,
    ``gene_range_probe.log_reflection_agrees`` — and the firings reported for each
    gene are the crossings its
    pre-images make, birth by birth. The pre-image is read from the primitive's
    input and the standard-normal draws recorded by intercepting the generator
    it was handed (``tests/support/variation_flow.py``), not re-derived from the
    seed. Every ruled gene must cross at least once, so no gene's rule goes
    unasked.
    """
    ranges = _declared(ecology.genome_schema)  # type: ignore[attr-defined]
    births = 0
    problems: list[str] = []
    crossings: dict[str, int] = {}
    for population, (first, second) in (
        (_on_their_bounds(ranges), ("low", "high")),
        (_inside_their_bounds(ranges), ("near-low", "near-high")),
    ):
        with variation_flow.traced() as spawned:
            for parents in ((first,), (second,), (first, second), (second, first)):
                for seed in SEEDS:
                    population.spawn(parents=parents, seed=seed)
        births += len(spawned)
        for birth in spawned:
            problems.extend(_crossing_problems(birth, ranges, crossings))
    ruled = sorted(name for name, declared in ranges.items() if declared.rule)
    assert births == 2 * 4 * len(SEEDS)
    assert not problems, (
        f"{problems[0]}; {len(problems)} departures from the declared rule over "
        f"{births} births"
    )
    assert [name for name in ruled if not crossings.get(name)] == [], (
        f"crossings per gene {crossings}: a ruled gene never crossed"
    )
