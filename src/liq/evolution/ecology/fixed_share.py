"""The fixed-share aggregator: one weight per living agent, updated once per decision point.

This module implements the frozen aggregator contract record,
``liq-docs/plans/agent-ecology-aggregator-contract.md`` at sha256
:data:`CONTRACT_SHA256`, and nothing else. In outline (the record's §2):

* **Experts** are exactly the living agents. The order is ascending
  :data:`~liq.evolution.ecology.types.AgentId` in Python ``str`` order, and
  every reduction is a float64 NumPy reduction over the contiguous array in
  that order. Nothing is ranked, sorted by value or selected.
* **Loss.** An agent's loss is ``-r / sigma``: its own recorded per-bar return
  over the root-mean-square of the reference returns given strictly before
  this point. It is defined only once :data:`SCALE_WARMUP` reference returns
  have been given and their scale is positive; until then every agent sleeps.
* **The update**, from the previous vector: survivors; a log-domain loss step
  over the awake that preserves their total mass, the asleep carried
  unchanged; the dead's mass returned pro rata; newborns at zero; then mixing
  ``(1 - s) v + s / P``, last. Every weight is therefore at least ``s / P``
  exactly in float64, and nothing is clipped.
* **Refusals.** A non-finite outcome, reference return or weight is refused,
  never repaired. A parameter other than the record's is refused.

The weights and the reference scale's ``(count, sum_of_squares)`` are the
aggregator's whole state, held in :class:`FixedShareState` as plain values a
run can snapshot. There is one aggregator; nothing names, registers or swaps
it, and the ``[allocation]`` table it is configured from carries only
:data:`ALLOCATION_KEYS`. The same code runs whatever produced the outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import AgentId

__all__ = [
    "ALLOCATION_KEYS",
    "CONTRACT_ID",
    "CONTRACT_SHA256",
    "ETA",
    "SCALE_WARMUP",
    "SHARE_RATE",
    "AllocationRefused",
    "FixedShare",
    "FixedShareState",
    "NonFiniteAllocationInput",
    "UnusableLiving",
    "from_allocation",
]

#: The frozen contract record's identity and sha256.
CONTRACT_ID = "fixed-share-hedge.v1"
CONTRACT_SHA256 = "e25af4f6069ace503ff8c2d97b6ecd8e8676ed838032c949262f0cb9032da0bb"

#: The record's learning rate and share rate (its ``contract.eta`` and
#: ``contract.share_rate``). The floor's alpha is the share rate.
ETA = 0.19952623149688797
SHARE_RATE = 0.07943282347242814

#: How many reference returns the loss waits for (the record's m).
SCALE_WARMUP = 32

#: The closed ``[allocation]`` key set.
ALLOCATION_KEYS = ("contract_sha256", "eta", "share_rate")


class AllocationRefused(EcologyError):
    """An ``[allocation]`` table or a parameter other than the frozen record's."""


class NonFiniteAllocationInput(EcologyError):
    """A non-finite outcome, reference return or weight; the run stops."""


class UnusableLiving(EcologyError):
    """A living set the update cannot be stated over."""


@dataclass(frozen=True)
class FixedShareState:
    """The aggregator's whole state at one decision point.

    Attributes:
        agent_ids: The living, ascending.
        weights: One float64 weight per agent of ``agent_ids``, aligned.
        count: How many reference returns have been given.
        sum_of_squares: The sum of their squares.
    """

    agent_ids: tuple[AgentId, ...]
    weights: np.ndarray
    count: int
    sum_of_squares: float


def _ids(living: Sequence[AgentId]) -> tuple[np.ndarray, np.ndarray]:
    """The living as an ascending array, and the order that sorts them."""
    ids = np.asarray(tuple(living), dtype=np.str_)
    if ids.size == 0:
        raise UnusableLiving("there are no living agents to weight")
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    repeated = ordered[1:][ordered[1:] == ordered[:-1]]
    if repeated.size:
        raise UnusableLiving(f"{sorted(set(repeated.tolist()))} appear twice")
    return ordered, order


def _same(value: object, frozen: object) -> bool:
    return type(value) is type(frozen) and value == frozen


@dataclass(frozen=True)
class FixedShare:
    """The aggregator, at the frozen record's parameters and no others.

    Raises:
        AllocationRefused: If any argument differs from the record.
    """

    contract_sha256: str
    eta: float
    share_rate: float

    def __post_init__(self) -> None:
        """Refuse every value but the frozen record's."""
        for name, frozen in (
            ("contract_sha256", CONTRACT_SHA256),
            ("eta", ETA),
            ("share_rate", SHARE_RATE),
        ):
            value = getattr(self, name)
            if not _same(value, frozen):
                raise AllocationRefused(
                    f"{name} = {value!r} is not the frozen record's {frozen!r}"
                )

    def start(self, founders: Sequence[AgentId]) -> FixedShareState:
        """The state before the first update: uniform over the founders."""
        ids, _ = _ids(founders)
        return FixedShareState(
            agent_ids=tuple(ids.tolist()),
            weights=np.full(ids.size, 1.0 / ids.size),
            count=0,
            sum_of_squares=0.0,
        )

    def update(
        self,
        state: FixedShareState,
        *,
        living: Sequence[AgentId],
        returns: np.ndarray,
        recorded: np.ndarray,
        reference: float | None,
    ) -> FixedShareState:
        """One decision point's update, over exactly ``living`` after turnover.

        Args:
            state: The state the previous decision point left.
            living: The living after this point's turnover, in any order.
            returns: Each living agent's own recorded return at this point,
                aligned with ``living``; read only where ``recorded``.
            recorded: Whether each living agent has an outcome at this point.
            reference: The reference return recorded at this point, added to
                the scale after the loss step, or ``None`` if none was.

        Raises:
            UnusableLiving: If ``living`` is empty, repeats an agent, is not
                aligned with ``returns`` and ``recorded``, or has no survivor.
            NonFiniteAllocationInput: If a recorded return, the reference
                return or a resulting weight is not finite.
        """
        ids, order = _ids(living)
        returns = np.asarray(returns, dtype=np.float64)
        recorded = np.asarray(recorded, dtype=bool)
        if returns.shape != ids.shape or recorded.shape != ids.shape:
            raise UnusableLiving(
                "returns and recorded must be aligned with the living, one each"
            )
        returns, recorded = returns[order], recorded[order]
        if not np.all(np.isfinite(returns[recorded])):
            raise NonFiniteAllocationInput("a recorded outcome is not finite")
        if reference is not None and not math.isfinite(reference):
            raise NonFiniteAllocationInput(f"the reference return {reference!r}")

        before = np.asarray(state.agent_ids, dtype=np.str_)
        at = np.minimum(np.searchsorted(before, ids), before.size - 1)
        survives = before[at] == ids
        if not survives.any():
            raise UnusableLiving(
                "no survivor from the previous point: no living agent was weighted before"
            )
        held = state.weights[at[survives]]

        mass = held.copy()
        defined = state.count >= SCALE_WARMUP and state.sum_of_squares > 0.0
        awake = recorded[survives] & defined
        if awake.any():
            sigma = math.sqrt(state.sum_of_squares / state.count)
            with np.errstate(all="ignore"):
                logs = np.log(held[awake]) + self.eta * (
                    returns[survives][awake] / sigma
                )
                top = logs.max()
                shifted = logs - top
                mass[awake] = np.exp(
                    shifted - np.log(np.sum(np.exp(shifted)))
                ) * np.sum(held[awake])

        pre_mix = np.zeros(ids.size)
        with np.errstate(all="ignore"):
            pre_mix[survives] = mass / np.sum(mass)
            weights = (1.0 - self.share_rate) * pre_mix + self.share_rate / ids.size
        if not np.all(np.isfinite(weights)):
            raise NonFiniteAllocationInput("a weight is not finite")

        count, squares = state.count, state.sum_of_squares
        if reference is not None:
            count, squares = count + 1, squares + reference * reference
        return FixedShareState(
            agent_ids=tuple(ids.tolist()),
            weights=weights,
            count=count,
            sum_of_squares=squares,
        )


def from_allocation(table: Mapping[str, object] | None) -> FixedShare | None:
    """The aggregator an ``[allocation]`` table configures, or ``None`` without one.

    Without a table the run produces no weights.

    Raises:
        AllocationRefused: If the table carries any key outside
            :data:`ALLOCATION_KEYS`, lacks one, or holds a value other than
            the frozen record's.
    """
    if table is None:
        return None
    extra = sorted(set(table) - set(ALLOCATION_KEYS))
    if extra:
        raise AllocationRefused(f"[allocation] does not take {extra}")
    missing = sorted(set(ALLOCATION_KEYS) - set(table))
    if missing:
        raise AllocationRefused(f"[allocation] is missing {missing}")
    return FixedShare(
        contract_sha256=table["contract_sha256"],  # ty: ignore[invalid-argument-type]
        eta=table["eta"],  # ty: ignore[invalid-argument-type]
        share_rate=table["share_rate"],  # ty: ignore[invalid-argument-type]
    )
