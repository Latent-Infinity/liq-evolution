"""Evidence that the initial population covers the declared gene ranges better than chance.

One claim. The founders an initial population is designed as cover the ranges
their genes are declared over more evenly than an independent uniform draw of
the same size, and the measured coverage is reported.

The measure and its quantifier are the oracle's
(``tests/support/design_coverage.py``): P1 and P2, each below the median of 32
seeded uniform draws, at each N ∈ {16, 50, 64, 1024}, over the founders' designed
genes mapped to unit coordinates through the ranges the genome schema declares
(read through ``tests/support/gene_range_probe.py``). Masks are bits and are
excluded; their on-fraction is reported.

Nothing here is market data: the points are configuration-space samples of the
genome, and the comparators are seeded draws over the same unit cube.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from liq.evolution import ecology
from liq.evolution.ecology.adapters import array_genome
from liq.evolution.ecology.agent import MASK_PREFIX
from liq.evolution.ecology.config import LearningConfig
from tests.support import design_coverage as oracle
from tests.support import gene_range_probe as probe

#: The production genome's feature count, and so its designed dimension 2F + 2.
FEATURES = tuple(f"feature_{index:02d}" for index in range(64))

#: The seed each design is drawn under, and the comparators' seed.
DESIGN_SEED = 0
COMPARATOR_SEED = 20261001

#: Reference CD² values, computed once with ``scipy.stats.qmc.discrepancy(…,
#: method="CD")`` (scipy 1.18.1) on the declared arrays below, and committed.
SMALL = np.array([[0.1, 0.2], [0.4, 0.9], [0.7, 0.35], [0.95, 0.6], [0.25, 0.05]])
GRID = np.array(
    [
        [(index * 0.37 + column * 0.11) % 1.0 for column in range(3)]
        for index in range(8)
    ]
)
CORNERS = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5], [0.5, 0.5, 0.5]])
REFERENCE = {
    "SMALL": 0.029402986111110526,
    "SMALL[:, :1]": 0.006333333333332978,
    "GRID": 0.03718914152728714,
    "GRID[:, 0]": 0.007833333333333803,
    "GRID[:, 1]": 0.0023333333333341866,
    "GRID[:, 2]": 0.0016583333333342054,
    "GRID[:, 01]": 0.021491461111111,
    "GRID[:, 02]": 0.014584633611111508,
    "GRID[:, 12]": 0.009949141111111448,
    "CORNERS": 0.19502314814814795,
}
RELATIVE = 1.0e-12


def _designed_units(count: int) -> tuple[np.ndarray, float]:
    """The designed founders' non-mask genes in unit coordinates, and the masks' on-fraction."""
    learning = LearningConfig()
    ranges = probe.declared_ranges(
        ecology.genome_schema, features=FEATURES, learning=learning
    )
    births = array_genome.design_founders(
        features=FEATURES, learning=learning, count=count, seed=DESIGN_SEED
    )
    genes = [birth.genome.genes for birth in births]
    measured = sorted(
        name
        for name, declared in ranges.items()
        if declared.rule is not None
        and not name.startswith(MASK_PREFIX)
        and name not in ecology.genome_schema.STRATEGY_GENES
    )
    units = np.array(
        [
            [
                (each[name] - ranges[name].low) / (ranges[name].high - ranges[name].low)
                for name in measured
            ]
            for each in genes
        ]
    )
    masks = [
        value
        for each in genes
        for name, value in each.items()
        if name.startswith(MASK_PREFIX)
    ]
    return units, float(np.mean(masks))


# ---------------------------------------------------------------------------
# The oracle's arithmetic, against the committed reference values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "points"),
    [
        ("SMALL", SMALL),
        ("SMALL[:, :1]", SMALL[:, :1]),
        ("GRID", GRID),
        ("GRID[:, 0]", GRID[:, [0]]),
        ("GRID[:, 1]", GRID[:, [1]]),
        ("GRID[:, 2]", GRID[:, [2]]),
        ("GRID[:, 01]", GRID[:, [0, 1]]),
        ("GRID[:, 02]", GRID[:, [0, 2]]),
        ("GRID[:, 12]", GRID[:, [1, 2]]),
        ("CORNERS", CORNERS),
    ],
)
def test_the_closed_form_agrees_with_the_reference(
    name: str, points: np.ndarray
) -> None:
    """The closed-form CD² is scipy's to 10⁻¹² relative on each declared array."""
    assert math.isclose(
        oracle.centred_discrepancy(points), REFERENCE[name], rel_tol=RELATIVE
    )


def test_p1_and_p2_are_the_means_of_the_projections() -> None:
    """P1 over the three 1-D projections, P2 over the three pairs, full CD² beside."""
    measured = oracle.coverage(GRID)
    p1 = (
        REFERENCE["GRID[:, 0]"] + REFERENCE["GRID[:, 1]"] + REFERENCE["GRID[:, 2]"]
    ) / 3
    p2 = (
        REFERENCE["GRID[:, 01]"] + REFERENCE["GRID[:, 02]"] + REFERENCE["GRID[:, 12]"]
    ) / 3
    assert math.isclose(measured.p1, p1, rel_tol=RELATIVE)
    assert math.isclose(measured.p2, p2, rel_tol=RELATIVE)
    assert math.isclose(measured.full, REFERENCE["GRID"], rel_tol=RELATIVE)


def test_the_oracle_refuses_points_it_cannot_measure() -> None:
    """Outside the cube, not a matrix, or one dimension where no pair exists."""
    for refused in (np.array([[1.5, 0.0]]), np.array([0.1, 0.2]), GRID[:, :1]):
        with pytest.raises(ValueError):
            oracle.coverage(refused)


# ---------------------------------------------------------------------------
# The claim.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", oracle.SIZES)
def test_initial_design_beats_a_uniform_draw(
    count: int, record_property: Callable[[str, object], None]
) -> None:
    """P1 and P2 of the designed founders each below the uniform draws' medians."""
    units, on_fraction = _designed_units(count)
    beats, figures = oracle.beats_uniform(units, seed=COMPARATOR_SEED)
    for key, value in {**figures, "mask_on_fraction": on_fraction}.items():
        record_property(key, value)

    assert units.shape == (count, len(FEATURES) + 2)
    assert beats, f"N = {count}: the design does not beat a uniform draw: {figures}"


def test_a_uniform_draw_in_the_designs_place_does_not_beat_one() -> None:
    """Non-vacuity: a seeded uniform draw substituted for the design fails the quantifier.

    The same dimension and sizes as the claim; the claim requires every N, so
    the substitute must miss at one of them at least.
    """
    verdicts = {
        count: oracle.beats_uniform(
            np.random.default_rng([DESIGN_SEED, count]).random(
                (count, len(FEATURES) + 2)
            ),
            seed=COMPARATOR_SEED,
        )[0]
        for count in oracle.SIZES
    }
    assert not all(verdicts.values()), verdicts
