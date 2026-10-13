"""What the genome schema declares, beyond what the gene-range facts read from it.

The facts hold the ranges and the reflection to their oracles. What is left for
here is the schema's own edges: the refusals, the reflection's exact landings on
an end and past the far one, the entry range's rule, the record a run keeps
of the declaration, and that one version names one declaration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from fractions import Fraction

import pytest

from liq.evolution.ecology import genome_schema
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    MASK_PREFIX,
    WEIGHT_PREFIX,
)
from liq.evolution.ecology.agent_genes import MUTATION_SCALE
from liq.evolution.ecology.config import LearningConfig

FEATURES = ("level", "step")


def _bounded(gene: str, value: float) -> tuple[float, int]:
    return genome_schema.bounded(
        gene, value, features=FEATURES, learning=LearningConfig(), seed=0
    )


def test_every_class_is_declared_and_only_moved_genes_carry_a_rule() -> None:
    """Masks carry no rule; every moved gene reflects; the scale gene in log space."""
    ranges = genome_schema.declared(features=FEATURES, learning=LearningConfig())

    assert set(ranges) == {
        *(f"{MASK_PREFIX}{name}" for name in FEATURES),
        *(f"{WEIGHT_PREFIX}{name}" for name in FEATURES),
        ENTRY_THRESHOLD,
        FORGETTING_FACTOR,
        MUTATION_SCALE,
    }
    assert ranges[f"{MASK_PREFIX}level"].rule is None
    assert ranges[MUTATION_SCALE] == genome_schema.GeneRange(
        genome_schema.SCALE_MINIMUM,
        genome_schema.SCALE_MAXIMUM,
        genome_schema.SCALE_RULE,
    )
    for name in (f"{WEIGHT_PREFIX}level", ENTRY_THRESHOLD, FORGETTING_FACTOR):
        assert ranges[name].rule == genome_schema.REFLECTION
    assert {MUTATION_SCALE} == genome_schema.STRATEGY_GENES


def test_a_genome_reading_no_feature_declares_no_mask_or_weight() -> None:
    """With no features there is no weight bound to derive."""
    ranges = genome_schema.declared(features=(), learning=LearningConfig())

    assert set(ranges) == {ENTRY_THRESHOLD, FORGETTING_FACTOR, MUTATION_SCALE}


def test_the_entry_range_is_the_largest_spread_a_reading_can_have() -> None:
    """± the weight-norm bound, moving with it."""
    for bound in (1.0, 100.0, 250.0):
        learning = LearningConfig(weight_norm_bound=bound)
        entry = genome_schema.declared(features=FEATURES, learning=learning)[
            ENTRY_THRESHOLD
        ]
        assert (entry.low, entry.high) == (-bound, bound)


def test_the_weight_bound_refuses_no_genes() -> None:
    """A per-gene bound of zero genes is not a bound of anything."""
    with pytest.raises(ValueError, match="at least one"):
        genome_schema.weight_bound(LearningConfig(), 0)


def test_the_root_ties_to_the_even_neighbour() -> None:
    """The exact midpoint between two doubles rounds to the one with an even last bit."""
    one_ulp = math.ulp(1.0)
    root = genome_schema._correctly_rounded_root
    assert root(Fraction(4)) == 2.0
    assert root((Fraction(1) + Fraction(one_ulp) / 2) ** 2) == 1.0
    assert root((Fraction(1) + 3 * Fraction(one_ulp) / 2) ** 2) == 1.0 + 2 * one_ulp
    assert root((Fraction(1) + Fraction(3 * one_ulp) / 4) ** 2) == 1.0 + one_ulp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, (0.5, 0)),
        (1.25, (0.75, 1)),
        (-0.25, (0.25, 1)),
        (2.0, (0.0, 1)),
        (-1.0, (1.0, 1)),
        (3.5, (0.5, 1)),
    ],
)
def test_reflection_folds_in_unit_coordinates(
    value: float, expected: tuple[float, int]
) -> None:
    """Inside untouched; past an end folded back, an exact end returned as itself."""
    assert genome_schema._reflected(value, 0.0, 1.0) == expected


def test_a_value_no_rule_acts_on_must_already_be_inside() -> None:
    """A mask outside [0, 1] is refused; one inside is returned unfired."""
    assert _bounded(f"{MASK_PREFIX}level", 1.0) == (1.0, 0)
    with pytest.raises(ValueError, match="no rule acts"):
        _bounded(f"{MASK_PREFIX}level", 2.0)


def _log_folded(log_value: float) -> float:
    """``exp(log_value)``, the scale a fold landing at that logarithm returns."""
    return math.exp(log_value)


def test_the_scale_gene_is_reflected_in_log_space() -> None:
    """Inside untouched; past either end folded about that end's logarithm, once."""
    low = math.log(genome_schema.SCALE_MINIMUM)
    high = math.log(genome_schema.SCALE_MAXIMUM)
    width = high - low
    inside = _log_folded(low + width / 2)
    assert _bounded(MUTATION_SCALE, inside) == (inside, 0)
    above, fired = _bounded(MUTATION_SCALE, _log_folded(high + width / 4))
    assert fired == 1
    assert math.isclose(math.log(above), high - width / 4, abs_tol=1e-12)
    below, fired = _bounded(MUTATION_SCALE, _log_folded(low - width / 4))
    assert fired == 1
    assert math.isclose(math.log(below), low + width / 4, abs_tol=1e-12)


def test_a_scale_folded_exactly_onto_an_end_returns_the_end() -> None:
    """A fold landing on an end's logarithm returns the declared end itself.

    Over [1, e] the logarithms of the ends, e² and e³ are exact, so two and
    three widths past the low end land exactly on the low end and the high.
    """
    low, high = 1.0, math.e
    assert (math.log(math.e**2), math.log(math.e**3)) == (2.0, 3.0)
    assert genome_schema._reflected_in_log_space(math.e**2, low, high) == (low, 1)
    assert genome_schema._reflected_in_log_space(math.e**3, low, high) == (high, 1)


def test_a_scale_that_is_not_positive_is_refused() -> None:
    """No fold in log space brings zero or a negative scale inside."""
    for value in (0.0, -0.1):
        with pytest.raises(ValueError, match="cannot be brought"):
            _bounded(MUTATION_SCALE, value)


def test_a_value_that_is_not_finite_is_refused() -> None:
    """No fold brings an infinity or a NaN inside."""
    for value in (math.inf, math.nan):
        with pytest.raises(ValueError, match="cannot be brought"):
            _bounded(ENTRY_THRESHOLD, value)


def test_an_undeclared_gene_is_refused() -> None:
    """A gene the schema does not declare has no rule to apply."""
    with pytest.raises(KeyError):
        _bounded("weight.unread", 0.0)


def test_the_record_names_the_version_the_policies_and_every_range() -> None:
    """What a run keeps: enough to say which ranges its genes were held to."""
    learning = LearningConfig()
    record = genome_schema.recorded(features=FEATURES, learning=learning)
    weight = genome_schema.weight_bound(learning, len(FEATURES))

    assert record["schema_version"] == genome_schema.SCHEMA_VERSION
    assert record["weight_policy"] == genome_schema.WEIGHT_POLICY
    assert record["scale_rule"] == genome_schema.SCALE_RULE
    assert record["founding_scale"] == genome_schema.FOUNDING_SCALE
    assert record["strategy_genes"] == (MUTATION_SCALE,)
    assert record["ranges"][f"{WEIGHT_PREFIX}step"] == (  # type: ignore[index]
        -weight,
        weight,
        genome_schema.REFLECTION,
    )


#: The declaration each genome-schema version names, as the SHA-256 of its
#: canonical record (:func:`_canonical`) over :data:`FEATURES` and
#: :data:`PINNED_LEARNING`. A changed range, rule, class, founding scale or
#: strategy-gene set is a new version and a new key here; an existing key's
#: digest is never edited.
PINNED: dict[str, str] = {
    "array-genome-ranges-1": (
        "5c1a3f5ee199db7fac453601d182309ce7163039b3dbd7c45bf9beff6862b266"
    ),
}

#: The three settings the ranges are shaped by, written out so that moving a
#: configuration default does not move the pin.
PINNED_LEARNING = LearningConfig(
    weight_norm_bound=100.0, forgetting_minimum=0.9, forgetting_maximum=0.9999
)


def _plain(value: object) -> object:
    """``value`` with every mapping a dict and every tuple a list, for JSON."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _canonical(record: Mapping[str, object]) -> str:
    """The record as sorted, compact JSON: one text per declaration."""
    return json.dumps(
        _plain(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def test_the_version_names_one_declaration() -> None:
    """The ranges are part of the version: one version, one declaration.

    The record of the declaration under fixed features and range-shaping
    settings is digested and compared with the digest pinned for the version it
    names, so a range, rule or constant changed without a new version is seen.
    """
    record = genome_schema.recorded(features=FEATURES, learning=PINNED_LEARNING)
    digest = hashlib.sha256(_canonical(record).encode()).hexdigest()

    assert record["schema_version"] == genome_schema.SCHEMA_VERSION
    assert digest == PINNED[genome_schema.SCHEMA_VERSION]
