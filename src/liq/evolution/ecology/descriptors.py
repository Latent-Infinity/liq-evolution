"""Where an agent sits in behaviour space, under the ecology's own descriptor vocabulary.

The vocabulary is ``ecology-descriptors-2``: four axes, in this order, each
normalised onto the unit interval as ``plans/oracle-v4-descriptor-axes-2026-10-02.md``
§1–§2 declares them, with beta's estimator as
``plans/oracle-v4-beta-estimator-2026-10-03.md`` §2 supersedes it. Every axis is read over one window: the agent's last
``W`` decision points with an outcome, where ``W`` is the run's fitness window.

* ``asset_class`` — the declared asset class of the instrument the agent's
  account trades, ``(k + 0.5) / K`` for the class's ordinal ``k`` among the
  ``K`` the universe declares. One class is declared, so the axis is one cell.
* ``holding_horizon`` — the mean length of the maximal runs of non-flat
  realised exposure inside the window, divided by ``W``.
* ``turnover`` — the mean absolute change in realised exposure between
  consecutive decision points of the window, divided by ``c_t``, the largest
  change one decision point can produce under the mandate.
* ``beta`` — the least-squares slope through the origin, ``Σ r_a·r / Σ r²``,
  of the agent's outcomes on the traded instrument's close-to-close return
  over the same decision points, as ``clip(beta, 0, beta_max) / beta_max``
  with ``beta_max`` the mandate's gross-exposure bound. With no intercept, the
  reference's whole move, drift included, is what the account is measured
  against: for an account that carries only its exposure it is the
  ``r²``-weighted mean of that exposure, inside ``[0, beta_max]`` for every
  reference. The reference is **provisional**: it is re-pointed, under a new
  vocabulary, when the run's matched-exposure reference is declared.

**Exclusion, not clamping.** Beta's clip is the only clip in the ecology, it
acts on a finite estimate, and the raw estimate is kept beside the clipped
value. Every other value outside [0, 1], or not finite — an undefined beta
included — is carried unaltered and marked excluded with its reason, so the
diversity record's out-of-bounds refusal sees it and nothing makes it binnable.

**Undescribed is not out of range.** An agent the store holds fewer than ``W``
outcomes for has no descriptor: :func:`describe` raises
:class:`DescriptorUnavailable` and :func:`read` lists it as undescribed with
what it held. No value is substituted — no zero, no ``nan``, no partial window.

**What it reads.** Only the population store: each agent's outcome and
realised-exposure histories and the population's reference-return history,
all written by the accounting step before that decision point's turnover. A
reading is a value, computed when it is taken, so no later decision point can
change it. Nothing here reads bars, and nothing here imports the simulator or
the GP stream's descriptor helpers; the GP stream's ``turnover`` and
``holding_period_proxy`` are the same definitions, held equal by a conformance
test rather than by an import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

import numpy as np

from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import AgentId, Descriptor, InstrumentId

__all__ = [
    "ACTIVITY_THRESHOLD",
    "ASSET_CLASS",
    "AXES",
    "BEHAVIOURAL_AXES",
    "BETA",
    "BETA_ESTIMATOR",
    "HOLDING_HORIZON",
    "MAXIMUM_BINS",
    "MINIMUM_BINS",
    "PROVISIONAL_VOCABULARY",
    "REFERENCE",
    "REFERENCE_PROVISIONAL",
    "TURNOVER",
    "VOCABULARY",
    "AgentDescriptor",
    "DescriptorDeclaration",
    "DescriptorUnavailable",
    "DescriptorsMisaligned",
    "Drift",
    "Exclusion",
    "Mandate",
    "Reading",
    "describe",
    "drift",
    "marginal_quantiles",
    "read",
]

#: The descriptor vocabulary this module computes.
#: Retired by plans/oracle-v4-beta-estimator-2026-10-03.md; refused like any vocabulary this module does not compute.
VOCABULARY = "ecology-descriptors-2"

#: The vocabulary read off the genome and the current wish before this one.
#: Refused: no coordinate of it is carried into a record of this one.
PROVISIONAL_VOCABULARY = "ecology-provisional-descriptors-1"

ASSET_CLASS = "asset_class"
HOLDING_HORIZON = "holding_horizon"
TURNOVER = "turnover"
BETA = "beta"

#: The axes, in the order the diversity record bins them.
AXES = (ASSET_CLASS, HOLDING_HORIZON, TURNOVER, BETA)

#: The axes measured from behaviour, which drift and whose bins are free.
BEHAVIOURAL_AXES = (HOLDING_HORIZON, TURNOVER, BETA)

#: A decision point is active when ``|exposure|`` exceeds this: the GP stream's
#: holding-period threshold, fixed for conformance.
ACTIVITY_THRESHOLD = 1e-12

#: The beta reference, and that it is provisional, as provenance records them.
REFERENCE = "traded_instrument_close_to_close"
REFERENCE_PROVISIONAL = True

#: Beta's estimator, as provenance records it under ``descriptor_beta_estimator``.
BETA_ESTIMATOR = "least_squares_through_origin"

#: The admissible range of each behavioural axis's bin count.
MINIMUM_BINS = 2
MAXIMUM_BINS = 16


class Exclusion(StrEnum):
    """Why an axis's value is excluded from every binning."""

    OUTSIDE_UNIT_INTERVAL = "outside_unit_interval"
    NOT_FINITE = "not_finite"
    UNDEFINED = "undefined"


class DescriptorUnavailable(EcologyError):
    """An agent holds fewer outcomes than the descriptor window, so it has no descriptor.

    Attributes:
        agent_id: Who was asked for.
        held: How many outcomes the store holds for it.
        window: How many the descriptor reads.
    """

    def __init__(self, agent_id: AgentId, *, held: int, window: int) -> None:
        super().__init__(
            f"{agent_id!r} holds {held} outcomes and its descriptor reads a window "
            f"of {window}; an agent short of the window is undescribed, and no "
            "value is substituted for it"
        )
        self.agent_id = agent_id
        self.held = held
        self.window = window


class DescriptorsMisaligned(EcologyError):
    """An agent's histories do not line up over its window, so none is read.

    Every living agent records an exposure and an outcome at every decision
    point after its first, and the population a reference return; a history
    shorter than the outcomes it should accompany is a defect, refused rather
    than truncated into a window that pairs two different decision points.
    """


@dataclass(frozen=True)
class Mandate:
    """The four mandate bounds the descriptors' normalisation is derived from.

    Attributes:
        max_position: Largest exposure in any one instrument.
        max_gross_exposure: Largest gross exposure across the account.
        max_weight_delta: Largest change of exposure one decision may ask for.
        max_turnover: Largest turnover one decision may ask for.

    Raises:
        ValueError: If a bound is not positive and finite.
    """

    max_position: float
    max_gross_exposure: float
    max_weight_delta: float
    max_turnover: float

    def __post_init__(self) -> None:
        for name in (
            "max_position",
            "max_gross_exposure",
            "max_weight_delta",
            "max_turnover",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"{name} was {value}; a descriptor is normalised by the "
                    "mandate's bounds, so each must be positive and finite"
                )

    @property
    def turnover_cap(self) -> float:
        """``c_t``: the largest change in exposure one decision point can produce."""
        return min(
            self.max_position,
            self.max_gross_exposure,
            self.max_weight_delta,
            self.max_turnover,
        )

    @property
    def beta_max(self) -> float:
        """The bound beta is clipped to under the provisional reference."""
        return self.max_gross_exposure


@dataclass(frozen=True)
class DescriptorDeclaration:
    """What a population's descriptors are computed under, refused if the record rules it out.

    Attributes:
        window: ``W``, the run's fitness window.
        history_capacity: How many outcomes the store keeps per agent.
        asset_classes: The asset classes the universe declares, in order.
        universe: Each traded instrument and its declared asset class.
        mandate: The bounds ``c_t`` and ``beta_max`` derive from.
        vocabulary: The descriptor vocabulary; only :data:`VOCABULARY`.

    Raises:
        ValueError: If the window is shorter than two decision points or longer
            than the store keeps; if the universe trades other than one
            instrument (the provisional reference is that instrument's own
            return), declares other than one asset class, or names an
            instrument with no declared class; or if the vocabulary is the
            provisional one or any other this module does not compute.
    """

    window: int
    history_capacity: int
    asset_classes: tuple[str, ...]
    universe: Mapping[InstrumentId, str]
    mandate: Mandate
    vocabulary: str = VOCABULARY

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(
                f"a descriptor window of {self.window} has no change of exposure "
                "and no slope to estimate; the window is at least two"
            )
        if self.history_capacity < self.window:
            raise ValueError(
                f"history_capacity {self.history_capacity} cannot retain a "
                f"descriptor window of {self.window}; a window the store cannot "
                "keep is a configuration error, not a truncation"
            )
        if self.vocabulary == PROVISIONAL_VOCABULARY:
            raise ValueError(
                f"{PROVISIONAL_VOCABULARY!r} is the provisional vocabulary; no "
                f"coordinate of it is computed or carried under {VOCABULARY!r}"
            )
        if self.vocabulary != VOCABULARY:
            raise ValueError(
                f"descriptor vocabulary {self.vocabulary!r} is not one this module "
                f"computes; it computes {VOCABULARY!r}"
            )
        if not self.asset_classes:
            raise ValueError("the universe declares no asset class")
        if len(self.asset_classes) > 1:
            raise ValueError(
                f"the universe declares more than one asset class "
                f"{self.asset_classes}; what the axis means for an agent whose "
                "window spans classes is not decided"
            )
        if len(self.universe) != 1:
            raise ValueError(
                f"the universe trades {sorted(self.universe)}; the provisional "
                "beta reference is the traded instrument's own return, so it is "
                "defined only for exactly one traded instrument"
            )
        for instrument, asset_class in self.universe.items():
            if asset_class not in self.asset_classes:
                raise ValueError(
                    f"{instrument!r} is of class {asset_class!r}, and the universe "
                    f"declares no class of that name among {self.asset_classes}"
                )

    @property
    def asset_class_count(self) -> int:
        """``K``, the number of declared asset classes."""
        return len(self.asset_classes)

    @property
    def asset_class_ordinal(self) -> int:
        """``k``, the ordinal of the traded instrument's class in the declared order."""
        (asset_class,) = self.universe.values()
        return self.asset_classes.index(asset_class)

    def archive_bins(
        self, *, holding_horizon: int, turnover: int, beta: int
    ) -> tuple[int, int, int, int]:
        """The bins per axis, in :data:`AXES` order, for counts ruled elsewhere.

        The asset-class axis has one bin per declared class. The three
        behavioural counts have no default here: they are chosen by the
        descriptor-parameter ruling, each in [2, 16].

        Raises:
            ValueError: If a behavioural count is outside [2, 16].
        """
        counts = {HOLDING_HORIZON: holding_horizon, TURNOVER: turnover, BETA: beta}
        for axis, count in counts.items():
            if not MINIMUM_BINS <= count <= MAXIMUM_BINS:
                raise ValueError(
                    f"{count} bins for {axis!r} is outside "
                    f"[{MINIMUM_BINS}, {MAXIMUM_BINS}]"
                )
        return (self.asset_class_count, holding_horizon, turnover, beta)


@dataclass(frozen=True)
class AgentDescriptor:
    """One described agent's four axes, normalised and raw.

    Attributes:
        agent_id: Who is described.
        values: Axis to normalised value, in :data:`AXES` order. An excluded
            value is carried unaltered, never clamped.
        raw: Axis to raw value: the class ordinal, the mean run length in
            decision points, the mean absolute exposure change, and the raw
            beta estimate — never overwritten by the clipped one.
        excluded: Axis to why its value is excluded from every binning; empty
            for an agent whose every value can be binned.
        vocabulary: The descriptor vocabulary.
    """

    agent_id: AgentId
    values: Mapping[str, float]
    raw: Mapping[str, float]
    excluded: Mapping[str, Exclusion]
    vocabulary: str

    def descriptor(self) -> Descriptor:
        """The port's descriptor: the normalised values, under the vocabulary."""
        return Descriptor(values=self.values, schema_version=self.vocabulary)


@dataclass(frozen=True)
class Reading:
    """Every living agent's descriptor at one instant, or why it has none.

    Attributes:
        window: The descriptor window read over.
        described: Each agent with a descriptor, excluded axes included.
        undescribed: Each agent short of the window, with the outcomes held.
    """

    window: int
    described: Mapping[AgentId, AgentDescriptor]
    undescribed: Mapping[AgentId, int]


@dataclass(frozen=True)
class Drift:
    """Within-life drift on one axis between two consecutive readings.

    Attributes:
        count: How many agents were described at both, with the axis binnable
            at both.
        median: The median absolute change, or ``None`` with no such agent.
        q90: The 90th percentile, likewise.
        maximum: The largest, likewise.
    """

    count: int
    median: float | None
    q90: float | None
    maximum: float | None


class DescribedStore(Protocol):
    """What the descriptors read of the population store."""

    def agent_ids(self) -> tuple[AgentId, ...]:
        """The living agents, in the order held."""
        ...

    def history(self, agent_id: AgentId) -> tuple[float, ...]:
        """One agent's retained outcomes, oldest first."""
        ...

    def exposures(self, agent_id: AgentId) -> tuple[float, ...]:
        """One agent's retained realised exposures, oldest first."""
        ...

    def reference_returns(self) -> tuple[float, ...]:
        """The population's retained reference returns, oldest first."""
        ...


def read(store: DescribedStore, declaration: DescriptorDeclaration) -> Reading:
    """Describe every living agent at once.

    Raises:
        DescriptorsMisaligned: If an agent's histories do not line up.
    """
    return _read(store, declaration, store.agent_ids())


def describe(
    store: DescribedStore, declaration: DescriptorDeclaration, agent_id: AgentId
) -> AgentDescriptor:
    """Describe one living agent.

    Raises:
        DescriptorUnavailable: If the store holds fewer than ``W`` outcomes for it.
        DescriptorsMisaligned: If its histories do not line up.
    """
    reading = _read(store, declaration, (agent_id,))
    if agent_id in reading.undescribed:
        raise DescriptorUnavailable(
            agent_id, held=reading.undescribed[agent_id], window=declaration.window
        )
    return reading.described[agent_id]


def _read(
    store: DescribedStore,
    declaration: DescriptorDeclaration,
    agent_ids: Sequence[AgentId],
) -> Reading:
    window = declaration.window
    references = store.reference_returns()
    undescribed: dict[AgentId, int] = {}
    described_ids: list[AgentId] = []
    outcomes: list[tuple[float, ...]] = []
    exposures: list[tuple[float, ...]] = []
    for agent_id in agent_ids:
        history = store.history(agent_id)
        if len(history) < window:
            undescribed[agent_id] = len(history)
            continue
        held = store.exposures(agent_id)
        if len(history) > len(references):
            raise DescriptorsMisaligned(
                f"{agent_id!r} holds {len(history)} outcomes and the population "
                f"{len(references)} reference returns; its window would pair "
                "outcomes with reference returns of other decision points"
            )
        if len(held) < len(history):
            raise DescriptorsMisaligned(
                f"{agent_id!r} holds {len(history)} outcomes and {len(held)} "
                "exposures; every outcome is recorded with the exposure held "
                "over it, so a shorter exposure history is a defect"
            )
        described_ids.append(agent_id)
        outcomes.append(history[-window:])
        exposures.append(held[-window:])
    described: dict[AgentId, AgentDescriptor] = {}
    if described_ids:
        described = _describe(
            declaration,
            described_ids,
            np.asarray(exposures, dtype=np.float64),
            np.asarray(outcomes, dtype=np.float64),
            np.asarray(references[-window:], dtype=np.float64),
        )
    return Reading(
        window=window,
        described=MappingProxyType(described),
        undescribed=MappingProxyType(undescribed),
    )


def _describe(
    declaration: DescriptorDeclaration,
    agent_ids: Sequence[AgentId],
    exposures: np.ndarray,
    outcomes: np.ndarray,
    reference: np.ndarray,
) -> dict[AgentId, AgentDescriptor]:
    """The four axes for every agent at once, over their ``(agents, W)`` windows."""
    window = declaration.window
    mandate = declaration.mandate
    horizon_raw = _mean_active_run_length(np.abs(exposures) > ACTIVITY_THRESHOLD)
    turnover_raw = _mean_absolute_change(exposures)
    beta_raw = _beta(outcomes, reference)
    beta = np.clip(beta_raw, 0.0, mandate.beta_max) / mandate.beta_max
    ordinal = declaration.asset_class_ordinal
    asset_class = (ordinal + 0.5) / declaration.asset_class_count

    columns = {
        HOLDING_HORIZON: (horizon_raw / window, horizon_raw),
        TURNOVER: (turnover_raw / mandate.turnover_cap, turnover_raw),
        BETA: (beta, beta_raw),
    }
    described: dict[AgentId, AgentDescriptor] = {}
    for row, agent_id in enumerate(agent_ids):
        values = {ASSET_CLASS: asset_class}
        raw = {ASSET_CLASS: float(ordinal)}
        excluded: dict[str, Exclusion] = {}
        for axis, (normalised, unnormalised) in columns.items():
            value = float(normalised[row])
            values[axis] = value
            raw[axis] = float(unnormalised[row])
            reason = _exclusion(axis, value)
            if reason is not None:
                excluded[axis] = reason
        described[agent_id] = AgentDescriptor(
            agent_id=agent_id,
            values=MappingProxyType(values),
            raw=MappingProxyType(raw),
            excluded=MappingProxyType(excluded),
            vocabulary=declaration.vocabulary,
        )
    return described


def _exclusion(axis: str, value: float) -> Exclusion | None:
    """Why ``value`` cannot be binned on ``axis``, or ``None`` if it can."""
    if math.isnan(value) and axis == BETA:
        return Exclusion.UNDEFINED
    if not math.isfinite(value):
        return Exclusion.NOT_FINITE
    if not 0.0 <= value <= 1.0:
        return Exclusion.OUTSIDE_UNIT_INTERVAL
    return None


def _mean_active_run_length(active: np.ndarray) -> np.ndarray:
    """Per row, the mean length of the maximal runs of ``True``; 0 where there is none.

    A run touching either edge of the window counts at its in-window length.
    """
    starts = active[:, 0].astype(np.int64) + np.sum(
        active[:, 1:] & ~active[:, :-1], axis=1
    )
    total = np.sum(active, axis=1)
    return np.divide(
        total, starts, out=np.zeros(len(active), dtype=np.float64), where=starts > 0
    )


def _mean_absolute_change(exposures: np.ndarray) -> np.ndarray:
    """Per row, the mean absolute change between consecutive columns.

    Accumulated one column at a time, left to right, so each row's sum is the
    same floating-point sum a sequential reduction of that row gives.
    """
    total = np.zeros(len(exposures), dtype=np.float64)
    for column in range(1, exposures.shape[1]):
        total = total + np.abs(exposures[:, column] - exposures[:, column - 1])
    return total / (exposures.shape[1] - 1)


def _beta(outcomes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per row, the least-squares slope through the origin of the outcomes on the reference.

    ``Σ r_a·r / Σ r²``, with no intercept: the reference's whole move, its
    drift included, is the move the account is measured against. For an
    account whose outcome is its carried exposure times the reference's
    return this is the ``r²``-weighted mean of that exposure, inside
    [0, max exposure] for every reference; only costs, repositioning inside
    a bar and rounding can take it outside.

    ``nan`` where it is undefined: any input in the window that is not
    finite; a reference window every return of which is exactly ``0.0``,
    the only window whose sum of squares is zero; or a computed sum of
    squares that is zero or not finite (underflow or overflow of non-zero
    returns). ``nan`` is the IEEE result of an undefined quotient, carried
    so the exclusion path sees it, never a placeholder.
    """
    if not np.all(np.isfinite(reference)) or np.all(reference == 0.0):
        return np.full(len(outcomes), np.nan)
    spread = float(np.sum(reference**2))
    if spread == 0.0 or not math.isfinite(spread):
        return np.full(len(outcomes), np.nan)
    slope = outcomes @ reference / spread
    finite = np.all(np.isfinite(outcomes), axis=1)
    return np.where(finite, slope, np.nan)


def drift(earlier: Reading, later: Reading) -> Mapping[str, Drift]:
    """Within-life drift per behavioural axis between two consecutive readings.

    For each agent described at both with the axis binnable at both, the
    absolute change of its normalised value, summarised as count, median,
    90th percentile and maximum.
    """
    summaries: dict[str, Drift] = {}
    for axis in BEHAVIOURAL_AXES:
        changes = [
            abs(described.values[axis] - earlier.described[agent_id].values[axis])
            for agent_id, described in later.described.items()
            if agent_id in earlier.described
            and axis not in described.excluded
            and axis not in earlier.described[agent_id].excluded
        ]
        if not changes:
            summaries[axis] = Drift(count=0, median=None, q90=None, maximum=None)
            continue
        deltas = np.asarray(changes, dtype=np.float64)
        summaries[axis] = Drift(
            count=len(changes),
            median=float(np.median(deltas)),
            q90=float(np.quantile(deltas, 0.9)),
            maximum=float(deltas.max()),
        )
    return MappingProxyType(summaries)


def marginal_quantiles(
    reading: Reading,
) -> Mapping[str, tuple[float, float, float] | None]:
    """Per behavioural axis, the 10th, 50th and 90th percentiles of the described.

    ``None`` for an axis no described agent has a binnable value on.
    """
    quantiles: dict[str, tuple[float, float, float] | None] = {}
    for axis in BEHAVIOURAL_AXES:
        values = [
            described.values[axis]
            for described in reading.described.values()
            if axis not in described.excluded
        ]
        if not values:
            quantiles[axis] = None
            continue
        low, middle, high = np.quantile(np.asarray(values), (0.1, 0.5, 0.9))
        quantiles[axis] = (float(low), float(middle), float(high))
    return MappingProxyType(quantiles)
