"""The oracle for one claim: an initial population covers its gene ranges better than chance.

This module decides what "its coverage of the declared gene ranges exceeds that
of an independent uniform draw of the same size" is *measured as*, under
``plans/oracle-v3-operator-ablation-resume-2026-10-01.md`` §3.

**The measure is P1 and P2.** Points are the founders' designed genes, each
mapped to the unit interval through its declared range. P1 is the mean, over
genes, of the squared centred L2 discrepancy (CD²) of the points' 1-D
projection onto that gene; P2 is the mean, over unordered pairs of genes, of
the CD² of the 2-D projection onto the pair. Lower is better covered. Mask
genes are bits — held at {0, 1}, with no spread to measure — so they are
excluded from both, and their on-fraction is reported instead. The full CD² of
the points is reported, and never asserted above six dimensions, where it no
longer tells a design from a draw.

**The quantifier, fixed before any run.** The design's P1 **and** its P2 are
each below the median of the same measure over **32** seeded independent
uniform draws of the same size, at **each** N ∈ {16, 50, 64, 1024}, at the
production genome's designed dimension: F = 64 features, 2F + 2 designed genes,
of which the F + 2 that are not masks are measured. A uniform draw substituted
for the design is expected to fail it.

**The arithmetic is closed form, in ``numpy`` only.** CD² of n points
x₁ … xₙ in [0, 1]^d, with a = |x − ½|, is (Hickernell 1998)

    (13/12)^d − (2/n) Σᵢ Πₖ (1 + ½aᵢₖ − ½aᵢₖ²)
              + (1/n²) Σᵢ Σⱼ Πₖ (1 + ½aᵢₖ + ½aⱼₖ − ½|xᵢₖ − xⱼₖ|).

For one and two dimensions the products have one and two factors, so P1 and P2
over every gene and every pair come from per-gene single and pairwise terms and
their inner products, accumulated in blocks of rows. It agrees with
``scipy.stats.qmc.discrepancy(…, method="CD")`` on the declared arrays in the
evidence module, whose reference values were computed with it once and are
committed there. ``scipy`` is not imported here: ``liq-evolution`` does not
declare it.

**Seam.** Nothing here reads the genome schema or calls the implementation:
the caller hands in unit coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How many seeded uniform draws the design is compared against.
UNIFORM_DRAWS = 32

#: The founding sizes the comparison is made at.
SIZES = (16, 50, 64, 1024)

#: The largest dimension at which the full CD² is meaningful to assert.
FULL_CD_ASSERTED_UP_TO = 6

#: Rows of the pairwise terms held at once.
_BLOCK = 128


@dataclass(frozen=True)
class Coverage:
    """How evenly one set of unit points covers its cube.

    Attributes:
        p1: Mean 1-D projection CD² over the dimensions.
        p2: Mean 2-D projection CD² over the unordered pairs of dimensions.
        full: The full CD² of the points.
    """

    p1: float
    p2: float
    full: float


def _unit(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"points must be an n × d array, got shape {x.shape}")
    if not np.all((x >= 0.0) & (x <= 1.0)):
        raise ValueError("every coordinate must lie in the unit interval")
    return x


def centred_discrepancy(points: np.ndarray) -> float:
    """The squared centred L2 discrepancy of ``points``, closed form."""
    x = _unit(points)
    n, d = x.shape
    a = np.abs(x - 0.5)
    single = np.prod(1.0 + 0.5 * a - 0.5 * a * a, axis=1).sum()
    double = 0.0
    for start in range(0, n, _BLOCK):
        xi, ai = x[start : start + _BLOCK, None, :], a[start : start + _BLOCK, None, :]
        terms = 1.0 + 0.5 * ai + 0.5 * a[None, :, :] - 0.5 * np.abs(xi - x[None, :, :])
        double += np.prod(terms, axis=2).sum()
    return float((13.0 / 12.0) ** d - 2.0 / n * single + double / (n * n))


def coverage(points: np.ndarray) -> Coverage:
    """P1, P2 and the full CD² of ``points``.

    Raises:
        ValueError: If the points are not an n × d array in the unit cube, or
            d is below two, where no pair exists.
    """
    x = _unit(points)
    n, d = x.shape
    if d < 2:
        raise ValueError(f"P2 needs at least two dimensions, got {d}")
    a = np.abs(x - 0.5)
    single = 1.0 + 0.5 * a - 0.5 * a * a  # n × d
    single_sums = single.sum(axis=0)  # Σᵢ per dimension
    single_gram = single.T @ single  # Σᵢ per pair
    double_sums = np.zeros(d)
    double_gram = np.zeros((d, d))
    for start in range(0, n, _BLOCK):
        xi, ai = x[start : start + _BLOCK, None, :], a[start : start + _BLOCK, None, :]
        terms = 1.0 + 0.5 * ai + 0.5 * a[None, :, :] - 0.5 * np.abs(xi - x[None, :, :])
        flat = terms.reshape(-1, d)
        double_sums += flat.sum(axis=0)
        double_gram += flat.T @ flat
    one = 13.0 / 12.0 - 2.0 / n * single_sums + double_sums / (n * n)
    two = (13.0 / 12.0) ** 2 - 2.0 / n * single_gram + double_gram / (n * n)
    upper = np.triu_indices(d, k=1)
    return Coverage(
        p1=float(one.mean()),
        p2=float(two[upper].mean()),
        full=centred_discrepancy(x),
    )


def uniform_draws(n: int, d: int, *, seed: int) -> list[np.ndarray]:
    """The :data:`UNIFORM_DRAWS` seeded uniform comparators of size ``n`` in ``d``.

    Configuration-space samples of the genome's unit cube, not data of any
    kind; seeded so the comparison is reproducible.
    """
    return [
        np.random.default_rng([seed, n, draw]).random((n, d))
        for draw in range(UNIFORM_DRAWS)
    ]


def beats_uniform(points: np.ndarray, *, seed: int) -> tuple[bool, dict[str, float]]:
    """Whether ``points``' P1 and P2 are each below the uniform draws' medians.

    Returns the verdict and the figures it was read from, for the report.
    """
    n, d = np.asarray(points).shape
    design = coverage(points)
    draws = [coverage(draw) for draw in uniform_draws(n, d, seed=seed)]
    median_p1 = float(np.median([draw.p1 for draw in draws]))
    median_p2 = float(np.median([draw.p2 for draw in draws]))
    figures = {
        "p1": design.p1,
        "p2": design.p2,
        "full_cd2": design.full,
        "uniform_median_p1": median_p1,
        "uniform_median_p2": median_p2,
    }
    return design.p1 < median_p1 and design.p2 < median_p2, figures
