"""Evidence that the mutation scale recorded with each birth is the declared rule's.

One claim. Every birth records the mutation scale in force when its genome was
drawn, and that scale is the one the declared rule predicts from the state the
birth was drawn from: its parents' scale genes and the seed it was drawn under.
A scale held constant, with or without noise added, is not adaptation and does
not meet the prediction.

The births observed are those of a completed steady-state run, and everything
a prediction needs is read from what outlives the agents: who each birth's
parents were and the seed it was drawn under, from the population's lineage
record; and each parent's scale gene, from the founders' declared genomes or
from the genome recorded with the parent's own birth. A prediction is never
read off the living population, so a birth whose parents have died is predicted
exactly as one whose parents live — and the run is required to hold such
births, and births whose parents were themselves born, so that neither half of
that is vacuous.

The rule and its arithmetic are the oracle's (``tests/support/mutation_scale_oracle.py``);
it is the operator decision recorded in
``liq-docs/plans/agent-ecology-adr-evolution-operators.md``.

A run's parents rarely sit at a bound of the scale's range, so the rule's
reflection is put to the test apart from the run: births spawned through the
population port from a parent on each bound, where about half of all draws
carry the scale past it, are held to the same prediction.

Nothing here is market data: the stream is declared arithmetic
(``tests/support/turnover_stream.py``), and the founders' scales are a declared
grid inside the rule's range.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology.adapters import ArrayGenomePopulation
from liq.evolution.ecology.agent import AgentBirth, PopulationState
from liq.evolution.ecology.config import Inheritance, LearningConfig
from liq.evolution.ecology.types import Genome
from tests.support import mutation_scale_oracle as oracle
from tests.support import turnover_stream as streams
from tests.support import variation_flow

#: How many decision points the run walks.
DECISION_POINTS = 120

#: The founders' scale genes, cycled over the founders by index: a declared grid
#: inside the rule's range, spanning most of it.
FOUNDER_SCALES = (0.0012, 0.01, 0.1, 0.3, 0.45)

#: Births spawned from each bound's parent: the variation probe's own N.
SEEDS = range(200)

DESCRIPTOR_SCHEMA_VERSION = "mutation-scale-descriptors-1"


def _turnover():
    """The turnover the run walks under; every key declared here."""
    return ecology.steady_state.TurnoverConfig(
        band_lower=14,
        band_upper=20,
        replacement_count=2,
        trigger_period=3,
        minimum_age=6,
        replacement_window=6,
        maximum_replaced_fraction=0.3,
        fitness_window=8,
        parents_per_birth=2,
        inheritance=Inheritance.BALDWINIAN,
        admission_redraws=8,
        seed=11,
    )


def _founder(index: int, scale: float | None = None) -> AgentBirth:
    """Founder ``index`` of the declared stream, carrying a scale gene.

    The gene is ``scale`` where one is given, and otherwise the grid's value
    for the founder's index.
    """
    born = streams.founder(index)
    genes = {
        **born.genome.genes,
        oracle.SCALE_GENE: FOUNDER_SCALES[index % len(FOUNDER_SCALES)]
        if scale is None
        else scale,
    }
    return replace(
        born,
        genome=Genome(
            genes=MappingProxyType(genes), schema_version=born.genome.schema_version
        ),
    )


def _run() -> tuple[tuple[AgentBirth, ...], streams.Steady]:
    """A completed run over the declared stream, founded with scale genes."""
    config = _turnover()
    founding = tuple(_founder(index) for index in range(config.band_lower))
    run = streams.steady(
        windows=streams.priced(count=DECISION_POINTS),
        founding=founding,
        turnover=config,
        deflator=streams.ScriptedDeflator(
            streams.per_sample(streams.negative_is_worse)
        ),
    )
    return founding, run


def _recorded_births(
    founding: tuple[AgentBirth, ...], run: streams.Steady
) -> list[oracle.RecordedBirth]:
    """Every birth of ``run``, as the records that outlive its agents state it."""
    genomes = {birth.agent_id: birth.genome for birth in founding}
    births: list[oracle.RecordedBirth] = []
    for birth in run.run.report.births:  # ty: ignore[unresolved-attribute]
        lineage = run.state.lineage(birth.agent_id)
        assert lineage.spawn_seed is not None, (
            f"{birth.agent_id}'s lineage records no seed, so its scale cannot be "
            "predicted"
        )
        births.append(
            oracle.RecordedBirth(
                agent_id=birth.agent_id,
                parent_scales=tuple(
                    genomes[parent].genes[oracle.SCALE_GENE]
                    for parent in lineage.parents
                ),
                genes=birth.genome.genes,
                seed=lineage.spawn_seed,
                recorded_scale=lineage.mutation_scale,
            )
        )
        genomes[birth.agent_id] = birth.genome
    return births


def test_recorded_mutation_scale_is_the_declared_rules_prediction() -> None:
    """At every birth of a completed run the recorded scale is the rule's prediction."""
    founding, run = _run()
    founders = {birth.agent_id for birth in founding}
    births = _recorded_births(founding, run)

    assert len(births) >= 20, f"only {len(births)} births, too few to say anything"
    lineages = [run.state.lineage(birth.agent_id) for birth in births]
    assert any(set(lineage.parents) - founders for lineage in lineages), (
        "no birth had a parent that was itself born, so no scale was predicted "
        "from a scale a birth had produced"
    )
    assert any(
        run.state.lineage(parent).died_at is not None
        for lineage in lineages
        for parent in lineage.parents
    ), (
        "no birth's parent has died, so no prediction was made from a parent "
        "that only the records still hold"
    )

    problems = oracle.disagreements(births)
    assert not problems, (
        f"{problems[0]}; {len(problems)} disagreements over {len(births)} births"
    )


def test_recorded_mutation_scale_is_the_prediction_at_each_bound_of_its_range() -> None:
    """From a parent on either bound, a scale carried past it is the reflected prediction."""
    low = _founder(0, oracle.SCALE_MINIMUM)
    high = _founder(1, oracle.SCALE_MAXIMUM)
    population = ArrayGenomePopulation(
        state=PopulationState.founded((low, high), learning=LearningConfig()),
        descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION,
    )
    births: list[oracle.RecordedBirth] = []
    reflected = {low.agent_id: 0, high.agent_id: 0}
    for parent in (low, high):
        scale = parent.genome.genes[oracle.SCALE_GENE]
        for seed in SEEDS:
            offspring = population.spawn(parents=(parent.agent_id,), seed=seed)
            births.append(
                oracle.RecordedBirth(
                    agent_id=f"offspring of {parent.agent_id} under seed {seed}",
                    parent_scales=(scale,),
                    genes=offspring.genome.genes,
                    seed=seed,
                    recorded_scale=offspring.mutation_scale,
                )
            )
            reflected[parent.agent_id] += oracle.predict(
                parent_scales=(scale,),
                genes=oracle.selectable_genes(offspring.genome.genes),
                seed=seed,
            ).reflected

    assert all(count > 0 for count in reflected.values()), (
        f"the reflection was predicted at {reflected} births per bound, so a bound "
        "went untested"
    )
    problems = oracle.disagreements(births)
    assert not problems, (
        f"{problems[0]}; {len(problems)} disagreements over {len(births)} births"
    )


def test_the_recorded_scale_is_the_scale_the_mutation_applied() -> None:
    """The scale each birth records is the one its mutation was called with, and used.

    Over every birth of the run above, with every platform primitive call
    recorded where the population adapter looks it up
    (``tests/support/variation_flow.py``): the ``scale`` the mutation primitive
    was called with is the ``Offspring.mutation_scale`` that ``spawn`` returned
    and the scale the birth's records hold, and every gene the mutation
    selected is, in the genome the birth carries, that call's output. A scale
    recorded but not in force fails the first; a gene moved again after the
    mutation, at any scale, fails the second. That the displacement is the
    scale times a draw is the platform's own claim, held by its goldens.
    """
    with variation_flow.traced() as spawned:
        _, run = _run()
    by_draw = {(birth.parents, birth.seed): birth for birth in spawned}
    births = run.run.report.births  # ty: ignore[unresolved-attribute]
    assert len(births) >= 20, f"only {len(births)} births, too few to say anything"
    problems: list[str] = []
    for birth in births:
        lineage = run.state.lineage(birth.agent_id)
        drawn = by_draw[(tuple(lineage.parents), lineage.spawn_seed)]
        mutation = drawn.only(variation_flow.MUTATION)
        applied = mutation.arguments["scale"]
        offspring = drawn.offspring
        assert offspring is not None
        for what, recorded in (
            ("spawn returned", offspring.mutation_scale),
            ("the birth recorded", birth.mutation_scale),
            ("the lineage recorded", lineage.mutation_scale),
        ):
            if recorded != applied:
                problems.append(
                    f"{birth.agent_id}: {what} scale {recorded!r}, the mutation "
                    f"was called with {applied!r}"
                )
        output = mutation.returned
        for position, name in enumerate(drawn.varied):
            if not output.selected[position]:  # type: ignore[attr-defined]
                continue
            produced = float(output.values[position])  # type: ignore[attr-defined]
            if birth.genome.genes[name] != produced:
                problems.append(
                    f"{birth.agent_id}: mutated {name} is "
                    f"{birth.genome.genes[name]!r}, the mutation's output there "
                    f"is {produced!r}"
                )
    assert not problems, (
        f"{problems[0]}; {len(problems)} departures over {len(births)} births"
    )


# ---------------------------------------------------------------------------
# The oracle's own arithmetic, exercised rather than trusted.
# ---------------------------------------------------------------------------


def test_the_parents_mean_is_geometric_and_exact_for_equal_scales() -> None:
    """σ₁ itself for one parent or equal scales; √(σ₁σ₂) otherwise."""
    assert oracle.parents_mean([0.1]) == 0.1
    assert oracle.parents_mean([0.1, 0.1]) == 0.1
    assert oracle.parents_mean([0.01, 0.25]) == math.sqrt(0.01 * 0.25)
    for refused in ([], [0.1, 0.1, 0.1], [0.0, 0.1], [math.nan], [-0.1]):
        with pytest.raises(ValueError):
            oracle.parents_mean(refused)


def test_the_reflection_folds_in_log_space_until_inside() -> None:
    """Inside is untouched; past either bound it is mirrored about that bound's log."""
    low, high = oracle.SCALE_MINIMUM, oracle.SCALE_MAXIMUM
    assert oracle.reflected_in_log_space(0.1) == 0.1
    assert oracle.reflected_in_log_space(high) == high
    assert math.isclose(oracle.reflected_in_log_space(high * 1.5), high / 1.5)
    assert math.isclose(oracle.reflected_in_log_space(low / 2.0), low * 2.0)
    width = math.log(high) - math.log(low)
    twice_past = math.exp(math.log(high) + width + 0.25)
    assert math.isclose(oracle.reflected_in_log_space(twice_past), low * math.exp(0.25))
    assert oracle.reflected_in_log_space(high * high / low) == low
    for refused in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="cannot be folded"):
            oracle.reflected_in_log_space(refused)


def test_the_prediction_is_the_declared_step_from_the_first_draw() -> None:
    """σ_x · exp(z₀/√n), with z₀ the first draw of the seed's generator."""
    seed, genes = 7, 4
    draw = oracle.first_standard_normal(seed)
    prediction = oracle.predict(parent_scales=[0.04, 0.09], genes=genes, seed=seed)

    assert prediction.parents_mean == math.sqrt(0.04 * 0.09)
    assert prediction.first_draw == draw
    assert prediction.unbounded == prediction.parents_mean * math.exp(draw / 2.0)
    assert prediction.scale == prediction.unbounded
    assert not prediction.reflected
    with pytest.raises(ValueError, match="at least one gene"):
        oracle.predict(parent_scales=[0.1], genes=0, seed=seed)


def test_a_step_past_a_bound_is_predicted_reflected() -> None:
    """From parents at the upper bound, an upward draw is folded back inside."""
    seed = next(s for s in range(100) if oracle.first_standard_normal(s) > 0.0)
    prediction = oracle.predict(
        parent_scales=[oracle.SCALE_MAXIMUM], genes=4, seed=seed
    )

    assert prediction.reflected
    assert prediction.unbounded > oracle.SCALE_MAXIMUM
    assert oracle.SCALE_MINIMUM <= prediction.scale < oracle.SCALE_MAXIMUM


def test_n_counts_every_gene_but_the_scale_gene() -> None:
    """The mutation selects among the behaviour-bearing genes, not its own scale."""
    assert oracle.selectable_genes({"a": 0.0, "b": 1.0, oracle.SCALE_GENE: 0.1}) == 2
    assert oracle.selectable_genes({"a": 0.0}) == 1


def test_agreement_refuses_a_missing_or_a_moved_scale() -> None:
    """Only a recording within the tolerance of a non-zero prediction agrees."""
    assert oracle.agrees(0.1, 0.1)
    assert oracle.agrees(0.1 * (1.0 + 0.5e-12), 0.1)
    assert not oracle.agrees(0.1 * (1.0 + 1.0e-9), 0.1)
    assert not oracle.agrees(None, 0.1)
    assert not oracle.agrees(math.nan, 0.1)
    assert not oracle.agrees(0.0, 0.0)


def _birth_drawn_by_the_rule(seed: int, scales: tuple[float, ...], scale: float):
    genes = {"a": 0.0, "b": 1.0, oracle.SCALE_GENE: scale}
    return oracle.RecordedBirth(
        agent_id=f"child-{seed}",
        parent_scales=scales,
        genes=MappingProxyType(genes),
        seed=seed,
        recorded_scale=scale,
    )


def test_the_check_passes_the_rule_and_names_a_constant() -> None:
    """Births drawn by the rule agree; a constant scale disagrees, first birth first."""
    scales = (0.02, 0.2)
    drawn = [
        _birth_drawn_by_the_rule(
            seed, scales, oracle.predict(parent_scales=scales, genes=2, seed=seed).scale
        )
        for seed in range(50)
    ]
    assert oracle.disagreements(drawn) == []

    constant = [replace(birth, recorded_scale=0.1) for birth in drawn]
    problems = oracle.disagreements(constant)
    assert len(problems) == 50
    assert problems[0].startswith("recorded scale 0.1, predicted ")
    assert problems[0].endswith(" at child-0")

    unrecorded = replace(drawn[0], recorded_scale=None)
    assert oracle.disagreements([unrecorded])[0].startswith("recorded scale None, ")

    copied = replace(drawn[0], genes=MappingProxyType({"a": 0.0, "b": 1.0}))
    assert oracle.disagreements([copied]) == [
        f"child-0 carries {oracle.SCALE_GENE} None, predicted {drawn[0].recorded_scale!r}"
    ]
