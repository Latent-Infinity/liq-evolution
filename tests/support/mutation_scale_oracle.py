"""The oracle for one claim: the mutation scale recorded with a birth is the declared rule's.

This module decides what "the scale recorded with each birth is the one the
declared rule predicts" is *measured as*. It predicts each birth's scale from
the state the birth was drawn from — its parents' scale genes and the seed its
genome was drawn under — and nothing the birth itself reports. A recorded scale
that was a constant with noise added, or a scale read from anywhere but the
parents, disagrees with the prediction at almost every birth.

**It restates no platform code and calls no implementation.** Every prediction
below is the arithmetic of the declared rule as the operator decision words it
(``liq-docs/plans/agent-ecology-adr-evolution-operators.md``, "Successor-3
result", factor M = M1; the rule's own text is
``agent-ecology-adr-evolution-operators.decision-rule.md`` §2 and amendment
A1(5)), written here with scalar ``math``, so that a rule implemented wrongly
upstream disagrees with it instead of agreeing by construction.

**The declared rule (self-adaptive log-normal scale, one scale per agent).**

* Every agent carries one scale gene, :data:`SCALE_GENE`. One scale serves
  every gene the mutation acts on, the forgetting factor included: there is no
  second scale gene for any block.
* A child's scale is σ' = σ_x · exp(z₀ / √n), where σ_x is the geometric mean
  of its parents' scales — σ₁ itself when the parents' scales are equal, and
  √(σ₁σ₂) otherwise — n is the number of genes the mutation selects among
  (every gene the genome carries except the scale gene itself), and z₀ is the
  birth generator's first standard-normal draw.
* The scale lies in [:data:`SCALE_MINIMUM`, :data:`SCALE_MAXIMUM`] in log
  space. A σ' outside it is reflected back in, under the decided boundary
  rule, reflection: its logarithm is folded about the logarithm of the bound
  it crossed, again and again, until it is inside. Reflection draws nothing.
* The child carries σ' as its own scale gene, and the birth records σ' as the
  scale in force.

**The seam onto the birth's generator.** The rule names z₀ as the birth
generator's *first* standard-normal draw. A birth is drawn under one integer
seed, and the generator that seed names is ``numpy.random.default_rng(seed)``,
the seeded ``Generator`` the platform's variation primitives take.
:func:`first_standard_normal` is the one place this oracle states that, and it
draws exactly one value from a generator it builds itself; it does not reach
the generator the implementation used.

**Agreement.** The recorded scale agrees with the prediction when their
relative difference is at most :data:`RELATIVE_TOLERANCE`. That absorbs only
different but equivalent orders of evaluating the same arithmetic — a geometric
mean taken through logarithms rather than a square root, a fold taken in unit
coordinates rather than in log units — and nothing a different rule produces:
a different draw, a different mean or a constant moves the scale by the step
itself, a sizeable fraction of its value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: The scale gene's name, as the operator decision names it.
SCALE_GENE = "mutation_scale"

#: The scale gene's declared range, in its own units; held in log space.
SCALE_MINIMUM = 1.0e-3
SCALE_MAXIMUM = 0.5

#: The largest relative difference between a recorded and a predicted scale
#: that is still agreement: about 4,500 units in the last place of a double,
#: well above the rounding of any equivalent evaluation order, far below the
#: smallest step a different rule would make.
RELATIVE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class Prediction:
    """What the rule predicts for one birth, and how it got there.

    Attributes:
        parents_mean: σ_x, the geometric mean of the parents' scales.
        first_draw: z₀, the birth generator's first standard-normal draw.
        genes: n, the number of genes the mutation selects among.
        unbounded: σ_x · exp(z₀ / √n), before the range is applied.
        scale: The predicted scale: ``unbounded``, reflected into the range
            where it fell outside.
        reflected: Whether the reflection acted.
    """

    parents_mean: float
    first_draw: float
    genes: int
    unbounded: float
    scale: float
    reflected: bool


def first_standard_normal(seed: int) -> float:
    """z₀: the first standard-normal draw of the generator ``seed`` names."""
    return float(np.random.default_rng(seed).standard_normal())


def parents_mean(scales: Sequence[float]) -> float:
    """σ_x: σ₁ when the parents' scales are equal, √(σ₁σ₂) otherwise.

    Raises:
        ValueError: For no parent, more than two, or a scale that is not a
            positive finite number — none of which the rule declares a mean of.
    """
    if not 1 <= len(scales) <= 2:
        raise ValueError(
            f"the declared rule takes the mean of one or two parents' scales, "
            f"not {len(scales)}"
        )
    for scale in scales:
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"a parent's scale must be positive and finite: {scale!r}")
    first = scales[0]
    if all(scale == first for scale in scales):
        return first
    return math.sqrt(scales[0] * scales[1])


def reflected_in_log_space(value: float) -> float:
    """``value`` folded into the scale's range about the logarithms of its bounds.

    Raises:
        ValueError: For a value that is not a positive finite number, which no
            fold brings into the range.
    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{value!r} cannot be folded into the scale's range")
    if SCALE_MINIMUM <= value <= SCALE_MAXIMUM:
        return value
    low, high = math.log(SCALE_MINIMUM), math.log(SCALE_MAXIMUM)
    width = high - low
    offset = math.fmod(math.log(value) - low, 2.0 * width)
    if offset < 0.0:
        offset += 2.0 * width
    if offset > width:
        offset = 2.0 * width - offset
    if offset == 0.0:
        return SCALE_MINIMUM
    if offset == width:
        return SCALE_MAXIMUM
    return math.exp(low + offset)


def selectable_genes(genes: Mapping[str, float]) -> int:
    """n: every gene the genome carries except the scale gene itself."""
    return sum(name != SCALE_GENE for name in genes)


def predict(*, parent_scales: Sequence[float], genes: int, seed: int) -> Prediction:
    """The scale the declared rule gives a birth drawn from this state under ``seed``."""
    if genes < 1:
        raise ValueError(f"the mutation selects among at least one gene, not {genes}")
    mean = parents_mean(parent_scales)
    draw = first_standard_normal(seed)
    unbounded = mean * math.exp(draw / math.sqrt(genes))
    scale = reflected_in_log_space(unbounded)
    return Prediction(
        parents_mean=mean,
        first_draw=draw,
        genes=genes,
        unbounded=unbounded,
        scale=scale,
        reflected=not (SCALE_MINIMUM <= unbounded <= SCALE_MAXIMUM),
    )


def agrees(recorded: float | None, predicted: float) -> bool:
    """Whether ``recorded`` is ``predicted`` to within :data:`RELATIVE_TOLERANCE`.

    A missing or non-finite recording, or a prediction of zero, is never
    agreement.
    """
    if recorded is None or not math.isfinite(recorded) or predicted == 0.0:
        return False
    return abs(recorded - predicted) <= RELATIVE_TOLERANCE * abs(predicted)


@dataclass(frozen=True)
class RecordedBirth:
    """One birth as its records state it, read before any prediction is made.

    Attributes:
        agent_id: Who was born.
        parent_scales: The scale gene each parent carried, in parent order.
        genes: The child's genome, scale gene included.
        seed: The seed its genome was drawn under.
        recorded_scale: The scale its records say was in force.
    """

    agent_id: str
    parent_scales: tuple[float, ...]
    genes: Mapping[str, float]
    seed: int
    recorded_scale: float | None


def disagreements(births: Sequence[RecordedBirth]) -> list[str]:
    """Every birth whose recorded scale, or whose own scale gene, is not the rule's.

    Births are reported in the order given, so the first line names the first
    birth that disagrees.
    """
    problems: list[str] = []
    for birth in births:
        predicted = predict(
            parent_scales=birth.parent_scales,
            genes=selectable_genes(birth.genes),
            seed=birth.seed,
        ).scale
        if not agrees(birth.recorded_scale, predicted):
            problems.append(
                f"recorded scale {birth.recorded_scale!r}, predicted {predicted!r} "
                f"at {birth.agent_id}"
            )
        carried = birth.genes.get(SCALE_GENE)
        if not agrees(carried, predicted):
            problems.append(
                f"{birth.agent_id} carries {SCALE_GENE} {carried!r}, predicted "
                f"{predicted!r}"
            )
    return problems
