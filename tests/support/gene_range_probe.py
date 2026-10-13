"""The oracle for two claims about genes: they stay in range, and a crossing is ruled.

This module decides what "every gene lies within its declared range" and "the
value produced is the one the boundary rule predicts" are each *measured as*. It
is declared inside the surfaces of the bindings that lean on it, because an
oracle that predicted the wrong value would leave both claims reading green
while the rule they name had changed.

**It restates no platform code.** Every prediction below is the arithmetic of a
rule as the rule is worded, written here rather than imported, so that a rule
implemented wrongly upstream disagrees with it instead of agreeing by
construction.

**The weight genes' bound.** A ``weight.`` gene is the prior a learned weight
starts from, and a birth whose starting weights have a norm above the learning
configuration's ``weight_norm_bound`` is refused. So the per-gene range for F
weight genes is [−c_F, c_F], where c_F is the largest double c at or below the
correctly rounded b/√F whose 1 × F row of c has a norm, *as the shipped norm
computes it*, at or below b. The start is determined exactly, in rationals, and
never as ``b / math.sqrt(F)``, which rounds twice: that start is above the
correctly rounded quotient at some F (the first is 31) and one step short of
the tight value at others (2 among them). b is read from the configuration and
the norm is the shipped one; neither is restated, and c_F is derived wherever it
is asked for rather than tabulated.

**Reflection.** A value past a bound is folded back about the bound it crossed,
again and again, until it is inside — "fold back into the range until inside",
worded in the gene's own units. The rule may be applied in unit coordinates and
mapped back, which rounds. :data:`REFLECTION_TOLERANCE_ULPS` units in the last
place of the range's width absorb that rounding only when the prediction takes
the same arithmetic path as the rule it checks (into unit coordinates, modulo
two, mirrored, mapped back). Two paths equal in exact arithmetic — folding in
the gene's own units against folding in unit coordinates, or mirroring without
first taking the value modulo two — can land one unit in the last place of the
*value* apart, and on a narrow range far from zero (``forgetting_factor``,
width about 0.1 near 0.99) that exceeds the tolerance. A rule that did anything
else to a crossing — clip it, redraw it, leave it — misses by a large fraction
of the width.

**Reflection in log space.** The scale gene is held in log space, and its rule
is reflection there: the logarithm of a value past a bound is folded back about
the logarithm of the bound it crossed, until inside, and the value is the
exponential of the result. The prediction is the reflection above applied to
logarithms; agreement is judged in logarithms, at the same tolerance in units in
the last place of the logarithms' width.

**Resampling.** A value past a bound is replaced by a uniform draw over the
gene's whole range. Its prediction is a distribution, not a number: every value
is inside the range, and the values drawn for one provoked crossing over many
seeds are not told apart from uniform by a one-sample Kolmogorov–Smirnov
distance at the declared critical value.

**Clipping has no oracle here.** It was struck by derivation before any rule
was declared, so a schema that declared it would meet no prediction and be
reported as such, rather than checked against one written in case.

**The seam onto the schema.** :func:`declared_ranges` and
:func:`apply_declared_rule` are the only two places this oracle reads the
genome schema. They state what they expect of it; nothing else here does.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.learning_update import _norms

#: The rule names the schema declares, and the three this oracle predicts.
REFLECTION = "reflection"
LOG_REFLECTION = "reflection in log space"
RESAMPLING = "resampling"
PREDICTED_RULES = frozenset({REFLECTION, LOG_REFLECTION, RESAMPLING})

#: Units in the last place of a range's width a reflected value may differ from
#: a prediction that takes the same arithmetic path as the rule (the module
#: docstring says why an exactly equivalent but different path is not covered on
#: a range far from zero). A rule that did anything else to a provoked crossing —
#: clip it, redraw it, leave it — misses by a large fraction of the width.
REFLECTION_TOLERANCE_ULPS = 4

#: The one-sample Kolmogorov–Smirnov critical value at a 0.001 significance
#: level is this coefficient over the square root of the sample size (the
#: asymptotic Smirnov value), declared rather than tuned.
UNIFORM_KS_COEFFICIENT = 1.949


@dataclass(frozen=True)
class DeclaredRange:
    """One gene's declared range and the rule that acts at its bounds.

    Attributes:
        low: The smallest value the gene may hold.
        high: The largest.
        rule: The boundary rule's name, or ``None`` for a gene no operator can
            carry past a bound (a mask bit is flipped, never pushed).
    """

    low: float
    high: float
    rule: str | None


def correctly_rounded_root(square: Fraction) -> float:
    """The double nearest the real square root of ``square``, ties to even.

    Decided exactly. The floating square root of the rounded square is only a
    starting point: it is walked down while its square is above ``square``,
    and up while the next double's square is not, so the walk ends on the
    largest double at or below the real root. The real root is then compared
    with the midpoint to the next double in rationals.
    """
    below = math.sqrt(float(square))
    while Fraction(below) ** 2 > square:
        below = math.nextafter(below, 0.0)
    while Fraction(math.nextafter(below, math.inf)) ** 2 <= square:
        below = math.nextafter(below, math.inf)
    above = math.nextafter(below, math.inf)
    midpoint = (Fraction(below) + Fraction(above)) / 2
    if Fraction(below) ** 2 == square or square < midpoint**2:
        return below
    if square > midpoint**2:
        return above
    return below if _significand_is_even(below) else above


def _significand_is_even(value: float) -> bool:
    """Whether the last bit of a positive double's significand is zero."""
    return struct.unpack("<q", struct.pack("<d", value))[0] % 2 == 0


def quotient_by_root(bound: float, genes: int) -> float:
    """fl(b/√F): the double nearest the real ``bound / sqrt(genes)``."""
    return correctly_rounded_root(Fraction(bound) ** 2 / genes)


def corner_norm(value: float, genes: int) -> float:
    """The shipped norm of a 1 × ``genes`` row holding ``value`` in every column."""
    return float(_norms(np.full((1, genes), value, dtype=np.float64))[0])


def per_gene_weight_bound(bound: float, genes: int) -> float:
    """c_F: stepped down from fl(b/√F) until the shipped norm admits its corner."""
    value = quotient_by_root(bound, genes)
    while corner_norm(value, genes) > bound:
        value = math.nextafter(value, 0.0)
    return value


def weight_bound_for(learning: LearningConfig, genes: int) -> float:
    """c_F for ``genes`` weight genes under ``learning``'s declared norm bound."""
    return per_gene_weight_bound(learning.weight_norm_bound, genes)


def reflected(value: float, low: float, high: float) -> float:
    """``value`` folded back about whichever bound it crossed, until inside.

    Raises:
        ValueError: If ``value`` is not finite: no number of folds brings it in.
    """
    if not math.isfinite(value):
        raise ValueError(f"{value} cannot be folded into [{low}, {high}]")
    while not low <= value <= high:
        value = 2.0 * high - value if value > high else 2.0 * low - value
    return value


def reflection_agrees(produced: float, value: float, low: float, high: float) -> bool:
    """Whether ``produced`` is ``value`` reflected into [low, high]."""
    return math.isclose(
        produced,
        reflected(value, low, high),
        rel_tol=0.0,
        abs_tol=REFLECTION_TOLERANCE_ULPS * math.ulp(high - low),
    )


def log_reflection_agrees(
    produced: float, value: float, low: float, high: float
) -> bool:
    """Whether ``produced`` is ``value`` reflected into [low, high] in logarithms."""
    if not (math.isfinite(produced) and produced > 0.0):
        return False
    return reflection_agrees(
        math.log(produced),
        math.log(value),
        math.log(low),
        math.log(high),
    )


def uniform_distance(values: Sequence[float], low: float, high: float) -> float:
    """The one-sample Kolmogorov–Smirnov distance of ``values`` from uniform."""
    ordered = sorted((value - low) / (high - low) for value in values)
    count = len(ordered)
    return max(
        max((rank + 1) / count - unit, unit - rank / count)
        for rank, unit in enumerate(ordered)
    )


def resampling_problems(values: Sequence[float], low: float, high: float) -> list[str]:
    """What stops ``values`` reading as uniform redraws over [low, high]."""
    outside = [value for value in values if not low <= value <= high]
    if outside:
        return [f"{len(outside)} resampled values fell outside [{low}, {high}]"]
    distance = uniform_distance(values, low, high)
    critical = UNIFORM_KS_COEFFICIENT / math.sqrt(len(values))
    if distance > critical:
        return [
            f"resampled values sit {distance:.3f} from uniform, above the "
            f"critical {critical:.3f}"
        ]
    return []


def out_of_range(
    genes: Mapping[str, float], ranges: Mapping[str, DeclaredRange]
) -> list[str]:
    """Every gene in ``genes`` outside its declared range, or with none declared."""
    problems: list[str] = []
    for name, value in sorted(genes.items()):
        declared = ranges.get(name)
        if declared is None:
            problems.append(f"{name} has no declared range")
        elif not declared.low <= value <= declared.high:
            problems.append(
                f"{name} = {value!r} outside [{declared.low!r}, {declared.high!r}]"
            )
    return problems


def declared_ranges(
    schema: object, *, features: Sequence[str], learning: LearningConfig
) -> Mapping[str, DeclaredRange]:
    """The ranges and rules ``schema`` declares for a genome over ``features``.

    Expects ``schema.declared(features=..., learning=...)`` to return a mapping
    from gene name to an object carrying ``low``, ``high`` and ``rule``.
    """
    declared = schema.declared(features=features, learning=learning)  # type: ignore[attr-defined]
    return MappingProxyType(
        {
            name: DeclaredRange(low=item.low, high=item.high, rule=item.rule)
            for name, item in declared.items()
        }
    )


def apply_declared_rule(
    schema: object,
    *,
    features: Sequence[str],
    learning: LearningConfig,
    gene: str,
    value: float,
    seed: int,
) -> tuple[float, int]:
    """The value ``schema``'s declared rule makes of ``value``, and its firings.

    Expects ``schema.bounded(gene, value, features=..., learning=..., seed=...)``
    to return the bounded value and how many times the rule fired on it.
    """
    produced, fired = schema.bounded(  # type: ignore[attr-defined]
        gene, value, features=features, learning=learning, seed=seed
    )
    return float(produced), int(fired)
