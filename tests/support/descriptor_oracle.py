"""The four ecology descriptor axes, predicted in plain ``math`` from their declared text.

An oracle the implementation can be compared with only if it shares nothing
with it. So this module restates each computation from the descriptor-axes
record (``plans/oracle-v4-descriptor-axes-2026-10-02.md`` §1–§2) and imports
nothing from the ecology, from ``liq.sim`` or from ``liq.evolution.fitness``:
no helper, no constant, no type. Its inputs are plain floats, and the one
structured input — an account's published states — is read by attribute
(``as_of``, ``equity``, ``exposures``), never through an ecology type.

**What it reads, and why that is independent.** The implementation reads the
population store's three bounded histories. The oracle reads none of them: an
agent's exposures and outcomes are re-derived from the states its evaluation
account published, and the reference from the bars of the stream, by
instant. A store that wrote a history at the wrong decision point, or a
descriptor that paired two windows out of step, therefore disagrees with this
module rather than with a shifted copy of itself.

**Truncation is absolute.** :func:`window_at` keeps only what was published at
or before the reading's instant, so the oracle's value is the declared
computation over the window that existed when the reading was taken. A
descriptor that read one decision point ahead would equal the declared
computation over a *later* window, and that is a different number from this
one, whatever a uniformly-ahead reader does to both sides of a comparison.

Nothing here is market data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

#: The declared activity threshold: a bar is active when ``|x| > 1e-12``.
ACTIVITY_THRESHOLD = 1e-12


class PublishedState(Protocol):
    """What the oracle reads of one published account state."""

    @property
    def as_of(self) -> datetime: ...

    @property
    def equity(self) -> float: ...

    @property
    def exposures(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class Window:
    """One agent's declared window, re-derived from its account and the bars.

    Attributes:
        exposures: The realised exposure at each of the window's decision
            points, oldest first.
        outcomes: The account's outcome at each of them,
            ``later.equity / earlier.equity - 1``.
        reference: The traded instrument's close-to-close return over each.
    """

    exposures: tuple[float, ...]
    outcomes: tuple[float, ...]
    reference: tuple[float, ...]


@dataclass(frozen=True)
class Predicted:
    """The four axes as declared, normalised and raw.

    Attributes:
        values: Axis name to normalised value, unaltered where it is excluded.
        raw: Axis name to raw value.
    """

    values: Mapping[str, float]
    raw: Mapping[str, float]


def window_at(
    states: Sequence[PublishedState],
    closes: Mapping[datetime, float],
    instants: Sequence[datetime],
    *,
    at: datetime,
    window: int,
    instrument: str,
) -> Window | None:
    """The agent's last ``window`` outcome decision points at or before ``at``.

    ``closes`` maps each decision instant of the stream to its bar's close and
    ``instants`` lists those instants in order, so the previous close of a
    decision point is the close of the instant before it in the stream.

    Returns ``None`` when fewer than ``window`` outcomes had been published by
    ``at``: the agent is undescribed, and the oracle predicts nothing for it.
    """
    published = [state for state in states if state.as_of <= at]
    if len(published) < window + 1:
        return None
    last = published[-(window + 1) :]
    previous_instant = {
        later: earlier for earlier, later in zip(instants, instants[1:], strict=False)
    }
    return Window(
        exposures=tuple(float(state.exposures[instrument]) for state in last[1:]),
        outcomes=tuple(
            later.equity / earlier.equity - 1.0
            for earlier, later in zip(last, last[1:], strict=False)
        ),
        reference=tuple(
            closes[state.as_of] / closes[previous_instant[state.as_of]] - 1.0
            for state in last[1:]
        ),
    )


def horizon_raw(exposures: Sequence[float]) -> float:
    """Mean length of the maximal active runs, in decision points; 0 if none."""
    runs: list[int] = []
    length = 0
    for value in exposures:
        if abs(value) > ACTIVITY_THRESHOLD:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    if not runs:
        return 0.0
    return sum(runs) / len(runs)


def turnover_raw(exposures: Sequence[float]) -> float:
    """Mean absolute change in exposure between consecutive decision points."""
    changes = [
        abs(later - earlier)
        for earlier, later in zip(exposures, exposures[1:], strict=False)
    ]
    return math.fsum(changes) / len(changes)


def beta_raw(outcomes: Sequence[float], reference: Sequence[float]) -> float:
    """Least-squares slope through the origin of the outcomes on the reference.

    ``nan`` where it is undefined: any input not finite; every reference
    return exactly zero; or a computed sum of squares that is zero or not
    finite.
    """
    if not all(math.isfinite(value) for value in (*outcomes, *reference)):
        return math.nan
    if all(value == 0.0 for value in reference):
        return math.nan
    spread = math.fsum(value * value for value in reference)
    if spread == 0.0 or not math.isfinite(spread):
        return math.nan
    return (
        math.fsum(
            outcome * value for outcome, value in zip(outcomes, reference, strict=True)
        )
        / spread
    )


def beta_normalised(raw: float, beta_max: float) -> float:
    """``clip(raw, 0, beta_max) / beta_max``; ``nan`` stays ``nan``."""
    if math.isnan(raw):
        return math.nan
    return min(max(raw, 0.0), beta_max) / beta_max


def predict(
    window: Window,
    *,
    length: int,
    turnover_cap: float,
    beta_max: float,
    class_ordinal: int,
    classes: int,
) -> Predicted:
    """Every axis of one agent, as the record declares it."""
    horizon = horizon_raw(window.exposures)
    turnover = turnover_raw(window.exposures)
    beta = beta_raw(window.outcomes, window.reference)
    return Predicted(
        values={
            "asset_class": (class_ordinal + 0.5) / classes,
            "holding_horizon": horizon / length,
            "turnover": turnover / turnover_cap,
            "beta": beta_normalised(beta, beta_max),
        },
        raw={
            "asset_class": float(class_ordinal),
            "holding_horizon": horizon,
            "turnover": turnover,
            "beta": beta,
        },
    )
