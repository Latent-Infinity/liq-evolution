"""The aggregate book's ledger: netting, the rebalance band, and attribution.

The book is one continuing account beside the evaluation accounts, keyed by
:data:`~liq.evolution.ecology.types.BOOK_ACCOUNT_ID`, opened at the evaluation
accounts' notional and moved only by its own profit and loss. This module keeps
its ledger. It does arithmetic and nothing else: the bounds on the book are the
portfolio-risk port's, reached through whatever adapter the run is composed
with, and the ledger only records what that port decided.

**What is netted.** At each decision point every living agent's *share* is its
weight times the exposure its mandate permitted it — a fraction of book equity,
because the weight is a fraction of the book's capital and the permitted target
a fraction of the capital allocated to the agent. The shares are summed into one
netted target **before** any bound is applied, and the port is handed the whole
netted book at once. The ledger never reads an agent's wish, its realised
holding, its fills, its equity or its costs: its only inputs are the weights,
the permitted targets, the bar and the book's own account.

**The band.** What the port leaves is traded to exactly when it is more than
the declared band away from what the book holds, when it is flat, or when a
de-risk action is in force; otherwise the book sends nothing and keeps what it
holds. A flat book is therefore always sent, never deferred.

**Attribution.** Each row also accounts for the bar the book was just carried
over, in five lines, each priced on the book's own equity and prices: every
agent's contribution (its share traded alone), the netting effect (what the
shares' own trades would have cost beyond the book's), the constraint effect,
the rebalance effect and the execution effect. Together they come to the change
in the book's equity, and a bar that does not reconcile within :data:`TAU_BAR`
stops the ledger rather than being written down.

**Defined every bar.** One row per decision point, whether or not the book
trades. A bar on which every agent is flat is a flat book, recorded.

**A batch.** Agents are put into a canonical order by identity and every sum
over them is exactly rounded, so the order they are handed in changes no
figure. The arithmetic is array-batched over the population.

**A plain value.** :class:`BookLedger` is frozen and holds tuples and floats
only, with no random state, so the loop holding it can snapshot and restore it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.ports import PortfolioRisk
from liq.evolution.ecology.types import (
    BOOK_ACCOUNT_ID,
    AccountState,
    AgentId,
    Bar,
    ConstrainedBook,
    CostProvenance,
    CostScenarioId,
    InstrumentId,
    NettedBook,
    PositionTarget,
    UtcTimestamp,
)

__all__ = [
    "TAU_BAR",
    "BarAttribution",
    "BookDecision",
    "BookDoesNotReconcile",
    "BookLedger",
    "BookRow",
]

#: The per-bar reconciliation tolerance, relative to ``max(1, equity)`` at the
#: bar's start: the accounting oracle's agreement figure, so "the same" keeps
#: one meaning across the evaluation accounts and the book.
TAU_BAR: Final = 1.0e-12

_SIDES_PER_ROUND_TRIP: Final = 2
_BASIS_POINTS_PER_UNIT: Final = 10_000.0
_FLAT: Final = 0.0


class BookDoesNotReconcile(EcologyError):
    """The five lines of a bar do not come to the book's change in equity."""


@dataclass(frozen=True)
class _Decided:
    """The series one decision point left: shares, netted, bounded and sent."""

    agent_ids: tuple[AgentId, ...]
    shares: tuple[float, ...]
    netted: float
    constrained: float
    sent: float


_NOTHING_DECIDED: Final = _Decided(
    agent_ids=(), shares=(), netted=_FLAT, constrained=_FLAT, sent=_FLAT
)


@dataclass(frozen=True)
class BarAttribution:
    """The bar the book was just carried over, in five lines of book equity.

    Attributes:
        agent_ids: Every agent with a share at either decision point the bar
            spans, in canonical order.
        contributions: Each agent's share traded alone at the book's equity,
            net of what that share's own trades would have been charged.
        agent_costs: What each share's own trades would have been charged: its
            held share where it died, and nothing where the book held none.
        netting: What the shares' own trades would have cost beyond what the
            netted target's trade costs. Never negative.
        constraint: What the portfolio-risk port's bounds changed.
        rebalance: What the band changed.
        execution: What acting on the sent target changed, including a target
            not reached and the charge actually made.
        book_cost: What the execution model charged the book over the bar.
        equity_change: The change in the book's equity over the bar.
        residual: The lines' sum less ``equity_change``.
    """

    agent_ids: tuple[AgentId, ...]
    contributions: tuple[float, ...]
    agent_costs: tuple[float, ...]
    netting: float
    constraint: float
    rebalance: float
    execution: float
    book_cost: float
    equity_change: float
    residual: float


@dataclass(frozen=True)
class BookDecision:
    """What the book decided at one decision point.

    Attributes:
        agent_ids: The living agents, in canonical order.
        shares: Each one's share of the book, as a fraction of book equity.
        netted: The shares' sum: the book's target before any bound.
        constrained: The portfolio-risk port's outcome for the netted book.
        held: What the book held at the decision point.
        sent: What the book holds after the next bar if what it was sent is
            reached: the bounded target where it trades, ``held`` where not.
        target: The target sent, acted on at the next bar, or ``None``.
    """

    agent_ids: tuple[AgentId, ...]
    shares: tuple[float, ...]
    netted: float
    constrained: ConstrainedBook
    held: float
    sent: float
    target: PositionTarget | None


@dataclass(frozen=True)
class BookRow:
    """One decision point of the book's ledger.

    Attributes:
        as_of: The decision instant (UTC).
        attribution: The bar the book was just carried over.
        decision: What the book decided at ``as_of``.
    """

    as_of: UtcTimestamp
    attribution: BarAttribution
    decision: BookDecision


def _float64(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _identities(agent_ids: Sequence[AgentId]) -> NDArray[np.str_]:
    return np.asarray(agent_ids, dtype=np.str_)


def _on(union: NDArray[np.str_], decided: _Decided) -> NDArray[np.float64]:
    """``decided``'s shares laid out over ``union``, nothing where it had none."""
    laid_out = np.zeros(union.size, dtype=np.float64)
    laid_out[np.searchsorted(union, _identities(decided.agent_ids))] = _float64(
        decided.shares
    )
    return laid_out


def _sum(values: NDArray[np.float64]) -> float:
    return math.fsum(values.tolist())


@dataclass(frozen=True)
class BookLedger:
    """The book's ledger state between decision points: a plain value.

    Build one with :meth:`open`; advance it with :meth:`decide`, which returns
    the next state and the row it wrote and changes nothing in place.

    Attributes:
        instrument: The single instrument the book follows.
        band: The rebalance band, a fraction of book equity in ``[0, 1)``.
        cost_scenario_id: The scenario the one-side rate was resolved from.
        one_side_rate: What one side of a trade is charged per unit of traded
            notional, as the scenario effectively charges it.
        equity: The book's equity at the last decision point.
        held: What the book held at the last decision point.
        costs_charged: What the book had been charged by then.
        close: The last decision point's close, or ``None`` before the first.
        high_water_mark: The most the book has been worth over the pass.
        de_risk_latched: Whether a de-risk action has fired in this pass.

    Raises:
        ValueError: If the band is outside ``[0, 1)`` or the book opens with
            nothing.
    """

    instrument: InstrumentId
    band: float
    cost_scenario_id: CostScenarioId
    one_side_rate: float
    equity: float
    held: float
    costs_charged: float
    close: float | None
    high_water_mark: float
    de_risk_latched: bool
    _last: _Decided = _NOTHING_DECIDED
    _before: _Decided = _NOTHING_DECIDED

    def __post_init__(self) -> None:
        if not 0.0 <= self.band < 1.0:
            raise ValueError(
                f"the rebalance band is a fraction of book equity in [0, 1), not "
                f"{self.band}: a band of one or more would never trade a change"
            )
        if not self.equity > 0.0:
            raise ValueError(
                f"the book must open with something: {self.equity} leaves no "
                "equity for its exposure to be a fraction of"
            )

    @classmethod
    def open(
        cls,
        *,
        instrument: InstrumentId,
        band: float,
        cost: CostProvenance,
        opening_equity: float,
    ) -> BookLedger:
        """Open the book flat at ``opening_equity``, charging what ``cost`` states."""
        return cls(
            instrument=instrument,
            band=band,
            cost_scenario_id=cost.cost_scenario_id,
            one_side_rate=cost.effective_round_trip_bps
            / _SIDES_PER_ROUND_TRIP
            / _BASIS_POINTS_PER_UNIT,
            equity=opening_equity,
            held=_FLAT,
            costs_charged=0.0,
            close=None,
            high_water_mark=opening_equity,
            de_risk_latched=False,
        )

    def decide(
        self,
        *,
        bar: Bar,
        account: AccountState,
        agent_ids: Sequence[AgentId],
        weights: ArrayLike,
        permitted: ArrayLike,
        risk: PortfolioRisk,
    ) -> tuple[BookLedger, BookRow]:
        """Attribute the bar just accounted, then net, bound and band the book.

        Args:
            bar: The bar the book was just carried over, priced in the book's
                instrument.
            account: The book's own account as published at the decision point.
            agent_ids: The living agents.
            weights: Each one's weight, aligned with ``agent_ids``.
            permitted: The exposure each one's mandate permits it, as a
                fraction of its allocated capital, aligned with ``agent_ids``.
            risk: The portfolio-risk port the netted book is handed to.

        Returns:
            The ledger after this decision point, and the row it wrote.

        Raises:
            ValueError: If the account is not the book's, the bar is in another
                instrument, the batch is not one weight and one permitted
                target per distinct agent, or the port answers for another book.
            BookDoesNotReconcile: If the bar's lines do not come to the book's
                change in equity within :data:`TAU_BAR`.
        """
        if account.agent_id != BOOK_ACCOUNT_ID:
            raise ValueError(
                f"the book ledger keeps the account reserved as "
                f"{BOOK_ACCOUNT_ID!r} and no other, not {account.agent_id!r}'s"
            )
        if bar.instrument != self.instrument:
            raise ValueError(
                f"the book follows {self.instrument!r}; a bar in "
                f"{bar.instrument!r} does not price it"
            )
        held = account.exposures.get(self.instrument, _FLAT)
        attribution = self._attribute(bar, account, held)
        decided = self._net(agent_ids, weights, permitted)
        high_water_mark = max(self.high_water_mark, account.equity)
        book = NettedBook(
            as_of=account.as_of,
            instrument=self.instrument,
            shares=MappingProxyType(
                dict(zip(decided.agent_ids, decided.shares, strict=True))
            ),
            netted=decided.netted,
            account=account,
            high_water_mark=high_water_mark,
            de_risk_latched=self.de_risk_latched,
        )
        constrained = risk.constrain(book)
        if (constrained.as_of, constrained.instrument, constrained.netted) != (
            book.as_of,
            book.instrument,
            book.netted,
        ):
            raise ValueError(
                "the portfolio-risk port answered for another book: netted "
                f"{constrained.netted} in {constrained.instrument!r} at "
                f"{constrained.as_of.isoformat()}, not {book.netted} in "
                f"{book.instrument!r} at {book.as_of.isoformat()}"
            )
        latched = self.de_risk_latched or constrained.de_risk is not None
        bounded = constrained.target
        trades = bounded == _FLAT or latched or abs(bounded - held) > self.band
        sent = bounded if trades else held
        target = (
            PositionTarget(
                agent_id=BOOK_ACCOUNT_ID,
                instrument=self.instrument,
                target_exposure=bounded,
                as_of=account.as_of,
            )
            if trades
            else None
        )
        row = BookRow(
            as_of=account.as_of,
            attribution=attribution,
            decision=BookDecision(
                agent_ids=decided.agent_ids,
                shares=decided.shares,
                netted=decided.netted,
                constrained=constrained,
                held=held,
                sent=sent,
                target=target,
            ),
        )
        following = replace(
            self,
            equity=account.equity,
            held=held,
            costs_charged=account.costs_charged,
            close=bar.close,
            high_water_mark=high_water_mark,
            de_risk_latched=latched,
            _last=replace(decided, constrained=bounded, sent=sent),
            _before=self._last,
        )
        return following, row

    @staticmethod
    def _net(
        agent_ids: Sequence[AgentId], weights: ArrayLike, permitted: ArrayLike
    ) -> _Decided:
        """Each living agent's share, in canonical order, and their exact sum."""
        identities = _identities(agent_ids)
        weight = _float64(weights)
        wanted = _float64(permitted)
        if (
            weight.shape != identities.shape
            or wanted.shape != identities.shape
            or np.unique(identities).size != identities.size
        ):
            raise ValueError(
                f"a book is netted from one weight and one permitted target per "
                f"distinct living agent: {identities.size} agent ids, "
                f"{weight.size} weights, {wanted.size} permitted targets"
            )
        order = np.argsort(identities, kind="stable")
        shares = weight[order] * wanted[order]
        return _Decided(
            agent_ids=tuple(identities[order].tolist()),
            shares=tuple(shares.tolist()),
            netted=_sum(shares),
            constrained=_FLAT,
            sent=_FLAT,
        )

    def _attribute(
        self, bar: Bar, account: AccountState, held: float
    ) -> BarAttribution:
        """The bar just accounted, in five lines that come to the change in equity.

        ``before`` is what was decided two decision points ago, held over the
        bar's gap; ``last`` what was decided one ago, acted on at the bar's open
        and held to its close.
        """
        before, last = self._before, self._last
        start = self.equity
        gap = _FLAT if self.close is None else bar.open / self.close - 1.0
        move = bar.close / bar.open - 1.0
        opened = start * (1.0 + self.held * gap)
        book_cost = account.costs_charged - self.costs_charged
        traded_at = opened - book_cost
        rate = self.one_side_rate

        def leg(earlier: float, later: float) -> float:
            return start * earlier * gap + traded_at * later * move

        def cost(earlier: float, later: float) -> float:
            return rate * opened * abs(later - earlier)

        union = np.union1d(_identities(before.agent_ids), _identities(last.agent_ids))
        earlier, later = _on(union, before), _on(union, last)
        agent_costs = rate * opened * np.abs(later - earlier)
        contributions = start * earlier * gap + traded_at * later * move - agent_costs
        netted_cost = cost(before.netted, last.netted)
        constrained_cost = cost(before.constrained, last.constrained)
        sent_cost = cost(before.sent, last.sent)
        netting = _sum(agent_costs) - netted_cost
        constraint = leg(
            before.constrained - before.netted, last.constrained - last.netted
        ) - (constrained_cost - netted_cost)
        rebalance = leg(
            before.sent - before.constrained, last.sent - last.constrained
        ) - (sent_cost - constrained_cost)
        execution = leg(self.held - before.sent, held - last.sent) - (
            book_cost - sent_cost
        )
        equity_change = account.equity - start
        residual = (
            math.fsum((_sum(contributions), netting, constraint, rebalance, execution))
            - equity_change
        )
        if abs(residual) > TAU_BAR * max(1.0, start):
            raise BookDoesNotReconcile(
                f"the book's bar to {account.as_of.isoformat()} does not reconcile: "
                f"its lines leave {residual} of a change of {equity_change} "
                f"unexplained, beyond {TAU_BAR} of max(1, {start})"
            )
        return BarAttribution(
            agent_ids=tuple(union.tolist()),
            contributions=tuple(contributions.tolist()),
            agent_costs=tuple(agent_costs.tolist()),
            netting=netting,
            constraint=constraint,
            rebalance=rebalance,
            execution=execution,
            book_cost=book_cost,
            equity_change=equity_change,
            residual=residual,
        )
