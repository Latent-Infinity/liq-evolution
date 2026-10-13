"""The declared range of every gene an array genome carries, and the rule at its bounds.

A gene is a number, and a number a variation operator moves has to be kept
somewhere. This module says where, gene class by gene class, and what happens
when an operator would carry a gene past an end. It is the one place those
numbers are declared; the population adapter reads them here and applies them
through the platform's variation primitives, and nothing else restates them.

**The classes.** A genome over F features carries, per feature, a ``mask.`` gene
and a ``weight.`` gene, then ``entry_threshold``, ``forgetting_factor`` and,
where the variation adapts its own scale, ``mutation_scale``.

* ``mask.<f>`` — a bit, held at {0, 1}. It is flipped when selected for
  mutation, never pushed, so no rule acts at its ends.
* ``weight.<f>`` — the prior the learned weight for the feature starts from.
  Its range is [−c_F, c_F], where c_F is the largest double at or below the
  correctly rounded b/√F whose 1 × F row of c_F has a norm, as the shipped norm
  computes it, at or below b = ``LearningConfig.weight_norm_bound``. A newborn's
  starting weights are refused above b, and the shipped norm is monotone in
  each coordinate, so a genome inside the box is never refused: no birth from
  parents inside their ranges can be. The start is decided exactly, in
  rationals, because ``b / math.sqrt(F)`` rounds twice and lands above the
  correctly rounded quotient at some F. c_F is derived wherever it is asked
  for, never tabulated. :data:`WEIGHT_POLICY` names this policy.
* ``entry_threshold`` — compared against the agent's reading, which is the sum
  over its consulted features of learned weight times standardised feature.
  Its range is [−b, b], **in reading units**, by this rule: a standardised
  feature has unit variance, so for features uncorrelated with each other a
  reading's standard deviation is the norm of the learned weights it is formed
  from, and that norm is at most b. The range is therefore ± the largest
  standard deviation a reading can have when its consulted features are
  uncorrelated; correlated features can spread a reading further, and the range
  does not claim to cover that. It moves with b and is never a literal.
* ``forgetting_factor`` — the range ``LearningConfig`` declares, read from it.
* ``mutation_scale`` — [:data:`SCALE_MINIMUM`, :data:`SCALE_MAXIMUM`], held in
  log space. A strategy gene (:data:`STRATEGY_GENES`): it parameterises the
  variation, not the agent. Each birth redraws it by the self-adaptive step,
  and founders carry :data:`FOUNDING_SCALE`. Its rule is its own,
  :data:`SCALE_RULE`.

**The rule at a bound** is reflection (:data:`REFLECTION`) for every gene an
operator moves: in unit coordinates of the gene's range a value past an end is
folded back, modulo two and mirrored, until inside, and mapped back; a fold
landing exactly on an end returns the end itself. The scale gene's rule,
reflection in log space (:data:`SCALE_RULE`), is the same fold taken between
the logarithms of its ends, so a step past either end is folded back about that
end's logarithm. Each crossing is counted once. Neither rule draws anything.

**The version.** :data:`SCHEMA_VERSION` names this declaration. Any change to a
range, a rule or a class changes it, so a run's record says which ranges its
genes were held to. A birth varies only a genome under this version, and its
child carries it, so the ranges a genome is held to are the ranges its own
version declares.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from types import MappingProxyType

import numpy as np

from .agent_genes import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    MASK_PREFIX,
    MUTATION_SCALE,
    WEIGHT_PREFIX,
)
from .config import LearningConfig
from .learning_update import _norms

__all__ = [
    "FOUNDING_SCALE",
    "REFLECTION",
    "SCALE_MAXIMUM",
    "SCALE_MINIMUM",
    "SCALE_RULE",
    "SCHEMA_VERSION",
    "STRATEGY_GENES",
    "WEIGHT_POLICY",
    "GeneRange",
    "bounded",
    "declared",
    "entry_bound",
    "recorded",
    "weight_bound",
]

#: The version of this declaration; changed whenever a range, rule or class is.
SCHEMA_VERSION = "array-genome-ranges-1"

#: The boundary rule of every gene an operator moves: fold back until inside.
REFLECTION = "reflection"

#: The rule the scale gene's own update applies past either end of its range.
SCALE_RULE = "reflection in log space"

#: How the ``weight.`` genes' range is set, as a run records it.
WEIGHT_POLICY = (
    "per-gene [-c_F, c_F]: c_F is the largest double at or below the correctly "
    "rounded weight_norm_bound / sqrt(F) whose all-c_F row the shipped norm "
    "admits"
)

#: The scale gene's range, in its own units, held in log space.
SCALE_MINIMUM = 1.0e-3
SCALE_MAXIMUM = 0.5

#: The scale every founder carries, and the scale a parent carrying no scale
#: gene is read at.
FOUNDING_SCALE = 0.1

#: The genes that parameterise the variation rather than the agent. A
#: difference confined to them is not a variation of the agent.
STRATEGY_GENES: frozenset[str] = frozenset({MUTATION_SCALE})


@dataclass(frozen=True)
class GeneRange:
    """One gene's declared range and the rule that acts at its ends.

    Attributes:
        low: The smallest value the gene may hold.
        high: The largest; above ``low``.
        rule: The rule an operator's crossing of an end is resolved by, in the
            coordinates the gene is held in, or ``None`` for a gene no operator
            carries past an end — a mask bit is flipped.
    """

    low: float
    high: float
    rule: str | None


def _significand_is_even(value: float) -> bool:
    """Whether the last bit of a positive double's significand is zero."""
    return struct.unpack("<q", struct.pack("<d", value))[0] % 2 == 0


def _correctly_rounded_root(square: Fraction) -> float:
    """The double nearest the real square root of ``square``, ties to even."""
    below = math.sqrt(float(square))
    while Fraction(below) ** 2 > square:
        below = math.nextafter(below, 0.0)
    while Fraction(math.nextafter(below, math.inf)) ** 2 <= square:
        below = math.nextafter(below, math.inf)
    above = math.nextafter(below, math.inf)
    midpoint = (Fraction(below) + Fraction(above)) / 2
    if square < midpoint**2:
        return below
    if square > midpoint**2:
        return above
    return below if _significand_is_even(below) else above


@cache
def weight_bound(learning: LearningConfig, genes: int) -> float:
    """c_F: the per-gene bound of ``genes`` weight genes under ``learning``.

    Raises:
        ValueError: If ``genes`` is below one.
    """
    if genes < 1:
        raise ValueError(f"a weight bound is per gene of at least one, not {genes}")
    bound = learning.weight_norm_bound
    value = _correctly_rounded_root(Fraction(bound) ** 2 / genes)
    while _norms(np.full((1, genes), value, dtype=np.float64))[0] > bound:
        value = math.nextafter(value, 0.0)
    return value


def entry_bound(learning: LearningConfig) -> float:
    """± this: the largest a reading's spread can be over uncorrelated features."""
    return learning.weight_norm_bound


def declared(
    *, features: Sequence[str], learning: LearningConfig
) -> Mapping[str, GeneRange]:
    """Every gene a genome over ``features`` may carry, with its range and rule."""
    ranges: dict[str, GeneRange] = {}
    if features:
        weight = weight_bound(learning, len(features))
        for name in features:
            ranges[f"{MASK_PREFIX}{name}"] = GeneRange(0.0, 1.0, None)
            ranges[f"{WEIGHT_PREFIX}{name}"] = GeneRange(-weight, weight, REFLECTION)
    entry = entry_bound(learning)
    ranges[ENTRY_THRESHOLD] = GeneRange(-entry, entry, REFLECTION)
    ranges[FORGETTING_FACTOR] = GeneRange(
        learning.forgetting_minimum, learning.forgetting_maximum, REFLECTION
    )
    ranges[MUTATION_SCALE] = GeneRange(SCALE_MINIMUM, SCALE_MAXIMUM, SCALE_RULE)
    return MappingProxyType(ranges)


def _folded(unit: float) -> float:
    """``unit`` folded into [0, 1]: modulo two, mirrored about one."""
    folded = math.fmod(unit, 2.0)
    if folded < 0.0:
        folded += 2.0
    return 2.0 - folded if folded > 1.0 else folded


def _reflected(value: float, low: float, high: float) -> tuple[float, int]:
    """``value`` reflected into [low, high] in unit coordinates, and whether it fired."""
    unit = (value - low) / (high - low)
    if 0.0 <= unit <= 1.0:
        return value, 0
    unit = _folded(unit)
    if unit == 0.0:
        return low, 1
    if unit == 1.0:
        return high, 1
    return low + unit * (high - low), 1


def _reflected_in_log_space(value: float, low: float, high: float) -> tuple[float, int]:
    """``value`` reflected between the logarithms of [low, high], and whether it fired.

    A fold landing exactly on an end's logarithm returns that end itself.
    """
    if low <= value <= high:
        return value, 0
    floor, ceiling = math.log(low), math.log(high)
    folded, fired = _reflected(math.log(value), floor, ceiling)
    if folded == floor:
        return low, fired
    if folded == ceiling:
        return high, fired
    return math.exp(folded), fired


def bounded(
    gene: str,
    value: float,
    *,
    features: Sequence[str],
    learning: LearningConfig,
    seed: int,
) -> tuple[float, int]:
    """What ``gene``'s declared rule makes of ``value``, and how often it fired.

    ``seed`` is accepted because a rule may draw; reflection does not.

    Raises:
        KeyError: If ``gene`` is not declared for a genome over ``features``.
        ValueError: If ``value`` is not finite, or is not positive for a gene
            held in log space, or lies outside the range of a gene no rule
            acts on.
    """
    del seed
    if not math.isfinite(value):
        raise ValueError(f"{gene} = {value!r} cannot be brought into any range")
    declaration = declared(features=features, learning=learning)[gene]
    if declaration.rule == SCALE_RULE:
        if value <= 0.0:
            raise ValueError(
                f"{gene} = {value!r} cannot be brought into any range held in log space"
            )
        return _reflected_in_log_space(value, declaration.low, declaration.high)
    if declaration.rule is None:
        if not declaration.low <= value <= declaration.high:
            raise ValueError(
                f"{gene} = {value!r} lies outside [{declaration.low}, "
                f"{declaration.high}], and no rule acts on it"
            )
        return value, 0
    return _reflected(value, declaration.low, declaration.high)


def recorded(
    *, features: Sequence[str], learning: LearningConfig
) -> Mapping[str, object]:
    """The declaration as a run records it: version, policies and every range."""
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "weight_policy": WEIGHT_POLICY,
            "scale_rule": SCALE_RULE,
            "founding_scale": FOUNDING_SCALE,
            "strategy_genes": tuple(sorted(STRATEGY_GENES)),
            "ranges": MappingProxyType(
                {
                    name: (gene.low, gene.high, gene.rule)
                    for name, gene in sorted(
                        declared(features=features, learning=learning).items()
                    )
                }
            ),
        }
    )
