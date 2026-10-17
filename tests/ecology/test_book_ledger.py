"""The aggregate book's ledger: it nets before any bound, attributes, and is defined every bar.

The book is one continuing account beside the evaluation accounts. At each
decision point the ledger nets every living agent's share — its weight times
what its mandate permitted it, as a fraction of the book's own equity — into one
target, hands that netted book to the portfolio-risk port, applies the declared
rebalance band to what the port leaves, and writes one row. The row also
attributes the bar the book has just been accounted for, in five lines that sum
to the change in the book's equity: each agent's contribution, the netting
effect, the constraint effect, the rebalance effect and the execution effect.

**Nothing here is market data.** The walks are over the declared arithmetic
stream S1 in ``tests/support/turnover_stream.py`` and over declared price pairs
written below; the cost rate is the one ``test_accounts.py`` declares for its own
checks, read back through the stand-in execution model's own provenance, never
written here. The rebalance bands below are declared for these checks and are
nobody's choice of band.

The ledger module is imported inside each case, so a missing module fails every
case where it is used rather than the collection of the file.
"""

from __future__ import annotations

import ast
import importlib
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np
import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent
from liq.evolution.ecology.adapters import (
    NullExecutionSimulator,
    NullPortfolioRisk,
    NullRiskSizer,
)
from liq.evolution.ecology.config import EcologyConfig, LearningConfig
from liq.evolution.ecology.types import (
    BOOK_ACCOUNT_ID,
    AccountState,
    Bar,
    BarWindow,
    ConstrainedBook,
    DeRiskEvent,
    Fill,
    NettedBook,
    PositionTarget,
    Rejection,
)
from tests.ecology.test_accounts import DECLARED_ROUND_TRIP_BPS
from tests.support import turnover_stream as streams

#: The module under test.
LEDGER_MODULE = "liq.evolution.ecology.book_ledger"

#: The per-bar tolerance the aggregate-book record fixes: the accounting
#: oracle's agreement figure, so "the same" keeps one meaning.
DECLARED_TAU_BAR = 1.0e-12

#: A band declared for these checks: wide enough that a small move sits inside
#: it and a large one does not. Not the run's band, which is chosen elsewhere.
BAND = 0.25

#: The trade-every-bar band, a mandatory arm of any band.
NO_BAND = 0.0

#: What the book and every account open with.
OPENING_EQUITY = 1.0

#: How many decision points a population walk covers.
DECISION_POINTS = 40

#: The decision point the population walk's turnover happens at.
TURNOVER_AT = 20

#: Who is admitted at the turnover point.
NEWBORN = "newborn"

INSTRUMENT = streams.INSTRUMENT
ORIGIN = datetime(2001, 1, 1, tzinfo=UTC)
BAR_DURATION = timedelta(minutes=1)


def _ledger() -> ModuleType:
    """The ledger module, imported where a case uses it."""
    return importlib.import_module(LEDGER_MODULE)


def _charging() -> NullExecutionSimulator:
    return NullExecutionSimulator(effective_round_trip_bps=DECLARED_ROUND_TRIP_BPS)


def _one_side_rate(simulator: NullExecutionSimulator) -> float:
    """The scenario's one-side rate, as the record defines it, from the model's own provenance."""
    return simulator.cost_provenance().effective_round_trip_bps / 2 * 1.0e-4


def _open(
    band: float = NO_BAND, simulator: NullExecutionSimulator | None = None
) -> Any:
    return _ledger().BookLedger.open(
        instrument=INSTRUMENT,
        band=band,
        cost=(_charging() if simulator is None else simulator).cost_provenance(),
        opening_equity=OPENING_EQUITY,
    )


def _bar(index: int, opening: float, close: float, instrument: str = INSTRUMENT) -> Bar:
    period_start = ORIGIN + index * BAR_DURATION
    return Bar(
        instrument=instrument,
        period_start=period_start,
        period_end=period_start + BAR_DURATION,
        open=opening,
        high=max(opening, close),
        low=min(opening, close),
        close=close,
        volume=1.0,
    )


def _book_account(
    as_of: datetime,
    equity: float,
    held: float,
    charged: float = 0.0,
    agent_id: str = BOOK_ACCOUNT_ID,
) -> AccountState:
    return AccountState(
        agent_id=agent_id,
        as_of=as_of,
        equity=equity,
        exposures=MappingProxyType({INSTRUMENT: held}),
        costs_charged=charged,
    )


# ---------------------------------------------------------------------------
# Stand-ins for the portfolio-risk port. Each is a declared rule of the test,
# not a model of any bound.
# ---------------------------------------------------------------------------


@dataclass
class _Recording:
    """Apply nothing, and keep every netted book handed in."""

    handed: list[NettedBook] = field(default_factory=list)

    def constrain(self, book: NettedBook) -> ConstrainedBook:
        self.handed.append(book)
        return NullPortfolioRisk().constrain(book)

    def bound(self) -> tuple[str, ...]:
        return ()

    def not_bound(self) -> tuple[tuple[str, str], ...]:
        return NullPortfolioRisk().not_bound()


@dataclass
class _LongAndUnlevered(_Recording):
    """Hold the netted target inside [0, 1], naming the bound when it binds."""

    def constrain(self, book: NettedBook) -> ConstrainedBook:
        self.handed.append(book)
        target = min(max(book.netted, 0.0), 1.0)
        rejections = (
            ()
            if target == book.netted
            else (
                Rejection(
                    agent_id=BOOK_ACCOUNT_ID,
                    instrument=book.instrument,
                    as_of=book.as_of,
                    reason="declared_book_bound",
                ),
            )
        )
        return ConstrainedBook(
            as_of=book.as_of,
            instrument=book.instrument,
            netted=book.netted,
            target=target,
            rejections=rejections,
        )


@dataclass
class _FiresAt(_Recording):
    """Fire a declared de-risk event at one decision point, leaving the target as netted."""

    fires_at: datetime = ORIGIN

    def constrain(self, book: NettedBook) -> ConstrainedBook:
        self.handed.append(book)
        event = (
            DeRiskEvent(
                as_of=book.as_of,
                metric="declared_probe_metric",
                value=0.0,
                threshold=0.0,
                high_water_mark=book.high_water_mark,
                book_equity=book.account.equity,
                action="declared_probe_action",
                target_before=book.netted,
                target_after=book.netted,
                reason="declared_probe_fired",
            )
            if book.as_of == self.fires_at
            else None
        )
        return ConstrainedBook(
            as_of=book.as_of,
            instrument=book.instrument,
            netted=book.netted,
            target=book.netted,
            rejections=(),
            de_risk=event,
        )


@dataclass
class _AnswersAnotherBook(_Recording):
    """Answer for a book other than the one handed in."""

    def constrain(self, book: NettedBook) -> ConstrainedBook:
        return ConstrainedBook(
            as_of=book.as_of,
            instrument=book.instrument,
            netted=book.netted + 0.125,
            target=book.netted + 0.125,
            rejections=(),
        )


# ---------------------------------------------------------------------------
# A declared walk of the book alone: the ledger driven over declared prices,
# with the book's account rebuilt here in the accounts' own order — carry into
# the open, act on the target formed a bar ago, carry to the close — through the
# stand-in execution model. The rebuild is the test's, independent of the
# ledger's arithmetic.
# ---------------------------------------------------------------------------


@dataclass
class _Walk:
    ledger: Any
    risk: Any = field(default_factory=_Recording)
    simulator: NullExecutionSimulator = field(default_factory=_charging)
    equity: float = OPENING_EQUITY
    held: float = 0.0
    charged: float = 0.0
    close: float | None = None
    pending: PositionTarget | None = None
    rows: list[Any] = field(default_factory=list)
    accounts: list[AccountState] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)

    def step(
        self,
        opening: float,
        close: float,
        ids: Sequence[str],
        weights: Sequence[float],
        permitted: Sequence[float],
    ) -> Any:
        bar = _bar(len(self.rows), opening, close)
        if self.close is not None and self.held != 0.0:
            self.equity *= 1.0 + self.held * (opening / self.close - 1.0)
        if self.pending is not None:
            _, reached = self.simulator.execute(
                self.pending,
                bar,
                _book_account(bar.period_start, self.equity, self.held, self.charged),
            )
            self.equity = reached.equity
            self.held = reached.exposures.get(INSTRUMENT, 0.0)
            self.charged = reached.costs_charged
            self.pending = None
        if self.held != 0.0:
            self.equity *= 1.0 + self.held * (close / opening - 1.0)
        self.close = close
        account = _book_account(bar.period_end, self.equity, self.held, self.charged)
        self.ledger, row = self.ledger.decide(
            bar=bar,
            account=account,
            agent_ids=tuple(ids),
            weights=np.asarray(weights, dtype=np.float64),
            permitted=np.asarray(permitted, dtype=np.float64),
            risk=self.risk,
        )
        self.pending = row.decision.target
        self.rows.append(row)
        self.accounts.append(account)
        self.bars.append(bar)
        return row


def _opened_value(walk: _Walk, index: int) -> float:
    """The book's equity at bar ``index``'s open, before anything traded there."""
    before = walk.accounts[index - 1]
    gap = walk.bars[index].open / walk.bars[index - 1].close - 1.0
    return before.equity * (1.0 + before.exposures[INSTRUMENT] * gap)


def _lines(row: Any) -> float:
    attribution = row.attribution
    return math.fsum(
        (
            math.fsum(attribution.contributions),
            attribution.netting,
            attribution.constraint,
            attribution.rebalance,
            attribution.execution,
        )
    )


# ---------------------------------------------------------------------------
# A population walk: the ledger driven through the accounts' own allocation
# hook, over S1, with turnover among the agents.
# ---------------------------------------------------------------------------


class _Ledgered:
    """An allocation hook that forms the book through the ledger, with declared weights."""

    def __init__(
        self,
        ledger: Any,
        weigh: Callable[[tuple[str, ...]], np.ndarray],
        *,
        reverse: bool = False,
    ) -> None:
        self.ledger = ledger
        self.weigh = weigh
        self.reverse = reverse
        self.risk = _Recording()
        self.rows: list[Any] = []
        self.permitted: list[dict[str, float]] = []

    def allocate(self, window: BarWindow, living: tuple[str, ...]) -> None:
        del window, living

    def book(
        self,
        window: BarWindow,
        permitted: Mapping[str, float],
        account: AccountState,
    ) -> PositionTarget | None:
        ids = tuple(permitted)
        if self.reverse:
            ids = ids[::-1]
        self.permitted.append(dict(permitted))
        self.ledger, row = self.ledger.decide(
            bar=window.bars[INSTRUMENT],
            account=account,
            agent_ids=ids,
            weights=self.weigh(ids),
            permitted=np.asarray([permitted[i] for i in ids], dtype=np.float64),
            risk=self.risk,
        )
        self.rows.append(row)
        return row.decision.target


def _equal(ids: tuple[str, ...]) -> np.ndarray:
    return np.full(len(ids), 1.0 / len(ids))


def _alone(chosen: str) -> Callable[[tuple[str, ...]], np.ndarray]:
    """The injected single-weight vector: all of the book to one agent."""

    def weigh(ids: tuple[str, ...]) -> np.ndarray:
        return np.asarray([1.0 if i == chosen else 0.0 for i in ids])

    return weigh


def _population_walk(
    founders: int,
    weigh: Callable[[tuple[str, ...]], np.ndarray],
    *,
    band: float = NO_BAND,
    dying: str | None = None,
    reverse: bool = False,
):
    state = agent.PopulationState.founded(
        streams.founders(founders), learning=LearningConfig()
    )
    simulator = _charging()
    accounts = ecology.population_accounts.PopulationAccounts(
        state,
        instrument=INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=simulator,
        opening_equity=OPENING_EQUITY,
    )
    seen = {"index": -1}

    def turnover(window: BarWindow) -> None:
        seen["index"] += 1
        if dying is None or seen["index"] != TURNOVER_AT:
            return
        state.cull(dying, died_at=window.as_of)
        accounts.close(dying, as_of=window.as_of)
        state.admit(
            replace(
                streams.founder(7),
                agent_id=NEWBORN,
                parents=(dying,),
                born_at=window.as_of,
            )
        )
        accounts.open(NEWBORN, as_of=window.as_of)

    allocation = _Ledgered(_open(band, simulator), weigh, reverse=reverse)
    stream = streams.priced(count=DECISION_POINTS)
    result = accounts.run(
        streams.Watched(stream, state),
        EcologyConfig(run_id="book-ledger"),
        turnover=turnover,
        allocation=allocation,
    )
    return result, allocation


FOUNDER = "founder-000"
OTHER = "founder-001"


# ---------------------------------------------------------------------------
# The netting point.
# ---------------------------------------------------------------------------


def test_the_book_nets_every_living_share_into_one_target_before_any_bound() -> None:
    """Each share is weight times permitted target; the netted book is what the port is handed."""
    walk = _Walk(_open())
    row = walk.step(100.0, 100.0, ("a", "b", "c"), (0.5, 0.25, 0.25), (1.0, 0.5, 0.0))

    (handed,) = walk.risk.handed
    assert dict(handed.shares) == {"a": 0.5, "b": 0.125, "c": 0.0}
    assert handed.netted == 0.625
    assert handed.instrument == INSTRUMENT
    assert handed.account is walk.accounts[0]
    assert handed.account.agent_id == BOOK_ACCOUNT_ID
    assert row.decision.agent_ids == ("a", "b", "c")
    assert row.decision.shares == (0.5, 0.125, 0.0)
    assert row.decision.netted == 0.625
    assert row.decision.constrained.target == 0.625


@pytest.mark.parametrize(
    ("weights", "permitted", "netted", "bound"),
    [
        # A long and a short share offset before the bound sees them: netted
        # first the book is 0.25; bounded agent by agent first it would be 0.5.
        ((0.5, 0.5), (1.0, -0.5), 0.25, 0.25),
        # A net-short book is bounded as a book: netted first the bound takes
        # -0.625 to flat; bounded agent by agent first it would be 0.125.
        ((0.25, 0.75), (0.5, -1.0), -0.625, 0.0),
    ],
    ids=["offsetting-shares", "net-short-book"],
)
def test_a_bound_acts_on_the_netted_book_never_on_an_agents_share(
    weights: tuple[float, float],
    permitted: tuple[float, float],
    netted: float,
    bound: float,
) -> None:
    """The book nets at the permitted targets, then the port bounds the netted book."""
    walk = _Walk(_open(), risk=_LongAndUnlevered())
    row = walk.step(100.0, 100.0, ("a", "b"), weights, permitted)

    (handed,) = walk.risk.handed
    assert handed.netted == netted
    assert row.decision.netted == netted
    assert row.decision.constrained.target == bound
    assert row.decision.target is not None
    assert row.decision.target.target_exposure == bound
    assert row.decision.target.agent_id == BOOK_ACCOUNT_ID


# ---------------------------------------------------------------------------
# The netting line: no phantom round-trip cost.
# ---------------------------------------------------------------------------


def test_agents_trading_against_each_other_trade_more_than_the_book_and_it_is_measured() -> (
    None
):
    """Sum of |per-agent trade| exceeds |book trade|, and the excess is the netting line."""
    walk = _Walk(_open())
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (1.0, 0.0))
    walk.step(100.0, 101.0, ("a", "b"), (0.5, 0.5), (0.0, 1.0))
    row = walk.step(102.0, 103.0, ("a", "b"), (0.5, 0.5), (0.0, 1.0))

    before, last = walk.rows[0].decision, walk.rows[1].decision
    agents_trade = math.fsum(
        abs(now - then) for now, then in zip(last.shares, before.shares, strict=True)
    )
    book_trade = abs(last.netted - before.netted)
    assert agents_trade > book_trade
    rate = _one_side_rate(walk.simulator)
    opened = _opened_value(walk, 2)
    expected = rate * opened * (agents_trade - book_trade)
    assert row.attribution.netting > 0.0
    assert abs(row.attribution.netting - expected) <= DECLARED_TAU_BAR
    assert math.fsum(row.attribution.agent_costs) > row.attribution.book_cost


def test_agents_trading_the_same_way_trade_exactly_what_the_book_does() -> None:
    """With no two shares moving against each other the netting line is nothing."""
    walk = _Walk(_open())
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (0.5, 0.0))
    walk.step(100.0, 101.0, ("a", "b"), (0.5, 0.5), (1.0, 0.5))
    row = walk.step(102.0, 103.0, ("a", "b"), (0.5, 0.5), (1.0, 0.5))

    before, last = walk.rows[0].decision, walk.rows[1].decision
    agents_trade = math.fsum(
        abs(now - then) for now, then in zip(last.shares, before.shares, strict=True)
    )
    assert agents_trade == abs(last.netted - before.netted)
    assert abs(row.attribution.netting) <= DECLARED_TAU_BAR
    assert row.attribution.book_cost > 0.0


def test_the_netting_line_is_never_negative_over_a_walk_with_turnover() -> None:
    """Triangle inequality, per bar, on the population walk."""
    _, allocation = _population_walk(3, _equal, dying=OTHER)
    nettings = [row.attribution.netting for row in allocation.rows]
    assert all(netting >= -DECLARED_TAU_BAR for netting in nettings)


# ---------------------------------------------------------------------------
# The attribution identity.
# ---------------------------------------------------------------------------


def test_the_declared_tolerance_is_the_records() -> None:
    """The per-bar tolerance is fixed by the record, never re-set."""
    assert _ledger().TAU_BAR == DECLARED_TAU_BAR


@pytest.mark.parametrize("band", [NO_BAND, BAND], ids=["no-band", "declared-band"])
def test_the_five_lines_sum_to_the_books_equity_change_at_every_bar(
    band: float,
) -> None:
    """Contributions, netting, constraint, rebalance and execution reconcile per bar."""
    result, allocation = _population_walk(3, _equal, band=band, dying=OTHER)
    book = result.book
    assert book is not None
    assert len(allocation.rows) == len(book.states) == DECISION_POINTS

    for index in range(1, DECISION_POINTS):
        opened_with = book.states[index - 1].equity
        change = book.states[index].equity - opened_with
        row = allocation.rows[index]
        assert row.attribution.equity_change == change
        residual = _lines(row) - change
        assert abs(residual) <= DECLARED_TAU_BAR * max(1.0, opened_with), (
            f"bar {index} does not reconcile: residual {residual}"
        )
        assert row.attribution.residual == pytest.approx(residual, abs=1e-15)
    assert any(row.attribution.book_cost > 0.0 for row in allocation.rows), (
        "the book never paid anything, so the cost terms were never exercised"
    )
    assert any(row.attribution.equity_change != 0.0 for row in allocation.rows)


def test_the_first_row_attributes_nothing_because_nothing_was_held() -> None:
    """The book opens flat at the first decision point; its first row is all zero."""
    walk = _Walk(_open())
    row = walk.step(100.0, 101.0, ("a",), (1.0,), (1.0,))
    attribution = row.attribution
    assert attribution.contributions == ()
    assert (
        attribution.netting,
        attribution.constraint,
        attribution.rebalance,
        attribution.execution,
        attribution.book_cost,
        attribution.equity_change,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_a_bound_that_binds_is_attributed_to_the_constraint_line() -> None:
    """What the port took away appears in the constraint line, not in any agent's."""
    walk = _Walk(_open(), risk=_LongAndUnlevered())
    walk.step(100.0, 100.0, ("a", "b"), (0.25, 0.75), (1.0, 1.0))
    walk.step(100.0, 101.0, ("a", "b"), (0.25, 0.75), (0.5, -1.0))
    row = walk.step(102.0, 104.0, ("a", "b"), (0.25, 0.75), (0.5, -1.0))

    assert walk.rows[1].decision.constrained.target == 0.0
    assert row.attribution.constraint != 0.0
    assert abs(_lines(row) - row.attribution.equity_change) <= DECLARED_TAU_BAR


def test_a_target_not_reached_is_attributed_to_the_execution_line() -> None:
    """A target the execution model could not reach leaves its difference in execution."""
    unreachable = NullExecutionSimulator(
        effective_round_trip_bps=DECLARED_ROUND_TRIP_BPS, reachable_exposure=0.5
    )
    walk = _Walk(_open(simulator=unreachable), simulator=unreachable)
    walk.step(100.0, 100.0, ("a",), (1.0,), (1.0,))
    row = walk.step(100.0, 102.0, ("a",), (1.0,), (1.0,))

    assert walk.accounts[1].exposures[INSTRUMENT] == 0.0
    assert row.attribution.execution != 0.0
    assert row.attribution.contributions[0] != 0.0
    assert abs(_lines(row) - row.attribution.equity_change) <= DECLARED_TAU_BAR


def test_an_account_whose_change_the_lines_do_not_explain_is_refused() -> None:
    """A bar that does not reconcile within the tolerance stops the ledger loudly."""
    ledger = _open()
    bar = _bar(0, 100.0, 100.0)
    ledger, _ = ledger.decide(
        bar=bar,
        account=_book_account(bar.period_end, OPENING_EQUITY, 0.0),
        agent_ids=("a",),
        weights=np.asarray([1.0]),
        permitted=np.asarray([0.0]),
        risk=_Recording(),
    )
    later = _bar(1, 100.0, 100.0)
    with pytest.raises(_ledger().BookDoesNotReconcile, match="reconcile"):
        ledger.decide(
            bar=later,
            account=_book_account(later.period_end, OPENING_EQUITY + 0.5, 0.0),
            agent_ids=("a",),
            weights=np.asarray([1.0]),
            permitted=np.asarray([0.0]),
            risk=_Recording(),
        )


def _flat_bar_then(opening_equity: float, equity: float) -> Any:
    """Open at ``opening_equity``, decide one flat bar, then hand in ``equity``.

    Nothing is held and nothing is charged, so every line is nothing and the
    residual is the account's whole change.
    """
    ledger = _ledger().BookLedger.open(
        instrument=INSTRUMENT,
        band=NO_BAND,
        cost=_charging().cost_provenance(),
        opening_equity=opening_equity,
    )
    flat = {
        "agent_ids": ("a",),
        "weights": np.asarray([1.0]),
        "permitted": np.asarray([0.0]),
        "risk": _Recording(),
    }
    bar = _bar(0, 100.0, 100.0)
    ledger, _ = ledger.decide(
        bar=bar, account=_book_account(bar.period_end, opening_equity, 0.0), **flat
    )
    later = _bar(1, 100.0, 100.0)
    _, row = ledger.decide(
        bar=later, account=_book_account(later.period_end, equity, 0.0), **flat
    )
    return row


def test_a_residual_beyond_the_tolerance_on_a_book_worth_less_than_two_is_refused() -> (
    None
):
    """On a book worth one the tolerance is the declared figure itself, not twice it."""
    equity = OPENING_EQUITY + 1.5 * DECLARED_TAU_BAR
    residual = abs(equity - OPENING_EQUITY)
    assert DECLARED_TAU_BAR < residual < 2.0 * DECLARED_TAU_BAR
    with pytest.raises(_ledger().BookDoesNotReconcile, match="reconcile"):
        _flat_bar_then(OPENING_EQUITY, equity)


def test_a_residual_within_the_tolerance_on_a_book_worth_more_than_one_is_accepted() -> (
    None
):
    """The tolerance scales up with the book's equity above one, never down."""
    worth = 4.0
    equity = worth + DECLARED_TAU_BAR
    residual = abs(equity - worth)
    assert DECLARED_TAU_BAR / worth < residual < DECLARED_TAU_BAR * worth
    row = _flat_bar_then(worth, equity)
    assert row.attribution.equity_change == equity - worth
    assert abs(row.attribution.residual) == residual


# ---------------------------------------------------------------------------
# A cost appears in the book only as far as the book held the position.
# ---------------------------------------------------------------------------


def test_a_dying_agent_is_charged_exactly_its_held_share_and_nothing_if_none_was_held() -> (
    None
):
    """Its attributed charge is the unwind of the share the book held of it."""
    walk = _Walk(_open())
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (1.0, 0.0))
    walk.step(100.0, 101.0, ("a", "b"), (0.5, 0.5), (1.0, 0.0))
    # Both die; a newborn holds nothing yet.
    walk.step(102.0, 103.0, ("c",), (1.0,), (0.0,))
    row = walk.step(103.0, 104.0, ("c",), (1.0,), (0.0,))

    attribution = row.attribution
    assert attribution.agent_ids == ("a", "b", "c")
    costs = dict(zip(attribution.agent_ids, attribution.agent_costs, strict=True))
    rate = _one_side_rate(walk.simulator)
    held_share = walk.rows[1].decision.shares[0]
    assert held_share == 0.5
    assert costs["a"] == pytest.approx(
        rate * _opened_value(walk, 3) * held_share, rel=1e-12
    )
    assert costs["b"] == 0.0
    assert costs["c"] == 0.0
    contributions = dict(
        zip(attribution.agent_ids, attribution.contributions, strict=True)
    )
    assert contributions["b"] == 0.0
    assert contributions["c"] == 0.0


def test_the_charge_rate_is_the_scenarios_it_is_handed() -> None:
    """Twice the scenario's rate charges every share twice as much; no rate is the ledger's."""
    rows = []
    for factor in (1.0, 2.0):
        simulator = NullExecutionSimulator(
            effective_round_trip_bps=factor * DECLARED_ROUND_TRIP_BPS
        )
        walk = _Walk(_open(simulator=simulator), simulator=simulator)
        walk.step(100.0, 100.0, ("a",), (1.0,), (1.0,))
        rows.append(walk.step(100.0, 100.0, ("a",), (1.0,), (1.0,)))
    once, twice = (row.attribution.agent_costs[0] for row in rows)
    assert once > 0.0
    assert twice == pytest.approx(2.0 * once, rel=1e-12)


def test_the_ledger_carries_the_scenario_it_was_opened_on() -> None:
    """The scenario's id is provenance: the ledger keeps the one it was handed."""
    cost = _charging().cost_provenance()
    ledger = _ledger().BookLedger.open(
        instrument=INSTRUMENT, band=NO_BAND, cost=cost, opening_equity=OPENING_EQUITY
    )
    assert ledger.cost_scenario_id == cost.cost_scenario_id


def test_a_book_opened_on_a_scenario_that_charges_nothing_charges_no_share() -> None:
    """The stand-in's default scenario charges nothing, and neither does any line."""
    free = NullExecutionSimulator()
    walk = _Walk(_open(simulator=free), simulator=free)
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (1.0, 0.0))
    row = walk.step(100.0, 101.0, ("a", "b"), (0.5, 0.5), (0.0, 1.0))
    assert row.attribution.agent_costs == (0.0, 0.0)
    assert row.attribution.netting == 0.0


# ---------------------------------------------------------------------------
# Defined every bar: an all-flat bar is a realised flat book, never nothing.
# ---------------------------------------------------------------------------


def test_a_bar_on_which_every_agent_is_flat_records_a_flat_book() -> None:
    """The band never delays a flat book, and a flat book is a row, not an absence."""
    walk = _Walk(_open(BAND))
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (0.5, 0.5))
    # Every agent flat, inside the band of what the book holds: it still goes flat.
    walk.step(100.0, 101.0, ("a", "b"), (0.5, 0.5), (0.0, 0.0))
    walk.step(101.0, 101.0, ("a", "b"), (0.5, 0.5), (0.0, 0.0))
    walk.step(101.0, 102.0, ("a", "b"), (0.5, 0.5), (0.0, 0.0))

    assert len(walk.rows) == 4
    assert walk.rows[1].decision.held == 0.5
    for row in walk.rows[1:]:
        assert row.decision.netted == 0.0
        assert row.decision.target is not None
        assert row.decision.target.target_exposure == 0.0
        assert row.decision.sent == 0.0
    assert walk.rows[2].decision.held == 0.0
    assert walk.rows[3].decision.held == 0.0


def test_a_population_walk_writes_one_row_per_decision_point() -> None:
    """One row per decision point, whether or not the book trades."""
    result, allocation = _population_walk(3, _equal, band=BAND, dying=OTHER)
    assert result.book is not None
    assert [row.as_of for row in allocation.rows] == [
        state.as_of for state in result.book.states
    ]
    assert any(row.decision.target is None for row in allocation.rows), (
        "the band never held the book, so the no-trade row was never exercised"
    )


# ---------------------------------------------------------------------------
# The rebalance band.
# ---------------------------------------------------------------------------


def test_a_move_inside_the_band_sends_nothing_and_one_beyond_it_trades_to_the_target() -> (
    None
):
    """Inside the band the book holds what it holds; beyond it, it trades to the target exactly."""
    walk = _Walk(_open(BAND))
    first = walk.step(100.0, 100.0, ("a",), (1.0,), (0.5,))
    inside = walk.step(100.0, 101.0, ("a",), (1.0,), (0.625,))
    beyond = walk.step(101.0, 102.0, ("a",), (1.0,), (1.0,))

    assert first.decision.target is not None
    assert first.decision.target.target_exposure == 0.5
    assert inside.decision.held == 0.5
    assert inside.decision.target is None
    assert inside.decision.sent == 0.5
    assert beyond.decision.target is not None
    assert beyond.decision.target.target_exposure == 1.0
    assert beyond.decision.sent == 1.0


def test_with_no_band_any_move_trades_and_no_move_sends_nothing() -> None:
    """The trade-every-bar arm: every change is traded, and a target already held is not resent."""
    walk = _Walk(_open(NO_BAND))
    walk.step(100.0, 100.0, ("a",), (1.0,), (0.5,))
    small = walk.step(100.0, 101.0, ("a",), (1.0,), (0.5078125,))
    still = walk.step(101.0, 102.0, ("a",), (1.0,), (0.5078125,))

    assert small.decision.target is not None
    assert small.decision.target.target_exposure == 0.5078125
    assert still.decision.target is None
    assert still.decision.sent == still.decision.held == 0.5078125


def test_a_de_risk_in_force_trades_inside_the_band_and_stays_in_force() -> None:
    """A de-risk firing is never delayed by the band, and its latch carries forward."""
    walk = _Walk(_open(BAND))
    walk.risk = _FiresAt(fires_at=_bar(1, 1.0, 1.0).period_end)
    walk.step(100.0, 100.0, ("a",), (1.0,), (0.5,))
    fired = walk.step(100.0, 101.0, ("a",), (1.0,), (0.5625,))
    latched = walk.step(101.0, 102.0, ("a",), (1.0,), (0.5625,))

    assert [book.de_risk_latched for book in walk.risk.handed] == [False, False, True]
    assert fired.decision.target is not None
    assert fired.decision.target.target_exposure == 0.5625
    assert latched.decision.target is not None
    assert walk.ledger.de_risk_latched is True


def test_the_high_water_mark_handed_to_the_port_is_the_most_the_book_has_been_worth() -> (
    None
):
    """The book's high-water mark is its own, over the pass so far."""
    walk = _Walk(_open())
    walk.step(100.0, 100.0, ("a",), (1.0,), (1.0,))
    for opening, close in ((100.0, 110.0), (110.0, 120.0), (120.0, 90.0), (90.0, 95.0)):
        walk.step(opening, close, ("a",), (1.0,), (1.0,))
    marks = [book.high_water_mark for book in walk.risk.handed]
    equities = [account.equity for account in walk.accounts]
    assert marks == [
        max(OPENING_EQUITY, *equities[: index + 1]) for index in range(len(equities))
    ]
    assert marks[-1] > equities[-1]


# ---------------------------------------------------------------------------
# Single-agent identity: the injected single-weight vector, and P = 1.
# ---------------------------------------------------------------------------


def _assert_the_book_is_the_agents_own_account(result, allocation, chosen: str) -> None:
    book = result.book
    assert book is not None
    own = result.accounts[chosen]
    assert len(book.states) == len(own.states) == len(allocation.rows)
    for row, permitted in zip(allocation.rows, allocation.permitted, strict=True):
        decision = row.decision
        shares = dict(zip(decision.agent_ids, decision.shares, strict=True))
        assert decision.netted == permitted[chosen]
        assert shares[chosen] == permitted[chosen]
        assert all(share == 0.0 for i, share in shares.items() if i != chosen)
        attribution = row.attribution
        contributions = dict(
            zip(attribution.agent_ids, attribution.contributions, strict=True)
        )
        assert all(value == 0.0 for i, value in contributions.items() if i != chosen)
        assert attribution.netting == 0.0
        assert attribution.constraint == 0.0
        assert attribution.rebalance == 0.0
        assert abs(attribution.execution) <= DECLARED_TAU_BAR * max(1.0, OPENING_EQUITY)
    for mine, theirs in zip(book.states, own.states, strict=True):
        assert mine.as_of == theirs.as_of
        assert mine.equity == theirs.equity
        assert dict(mine.exposures) == dict(theirs.exposures)
        assert mine.costs_charged == theirs.costs_charged
    assert any(isinstance(outcome, Fill) for outcome in book.fills), (
        "the chosen agent never traded, so the identity compares two idle accounts"
    )
    assert book.states[-1].costs_charged > 0.0


def test_the_injected_single_weight_vector_reproduces_that_agents_own_account() -> None:
    """All of the book to one founder, with turnover among the others: the book is its account."""
    result, allocation = _population_walk(3, _alone(FOUNDER), dying=OTHER)
    assert NEWBORN in result.accounts, "no turnover happened among the others"
    _assert_the_book_is_the_agents_own_account(result, allocation, FOUNDER)


def test_a_population_of_one_is_its_own_book() -> None:
    """P = 1: the whole book to the only agent, which is its account bitwise."""
    result, allocation = _population_walk(1, _alone(FOUNDER))
    _assert_the_book_is_the_agents_own_account(result, allocation, FOUNDER)


# ---------------------------------------------------------------------------
# A batch: the order agents are handed in is not part of the question.
# ---------------------------------------------------------------------------


def test_permuting_the_agents_changes_no_figure() -> None:
    """The same agents handed in another order give the same rows and the same state, bitwise."""
    ids = ("c", "a", "d", "b")
    weights = (0.125, 0.375, 0.25, 0.25)
    path = (
        ((100.0, 100.0), (1.0, 0.25, 0.5, 0.0)),
        ((100.0, 101.0), (0.0, 1.0, 0.5, 0.75)),
        ((101.0, 99.0), (0.5, 0.5, 0.0, 1.0)),
        ((99.5, 100.5), (0.5, 0.5, 0.0, 1.0)),
    )
    walks = []
    for order in (slice(None), slice(None, None, -1)):
        walk = _Walk(_open())
        for (opening, close), permitted in path:
            walk.step(opening, close, ids[order], weights[order], permitted[order])
        walks.append(walk)
    forward, backward = walks
    assert forward.rows == backward.rows
    assert forward.ledger == backward.ledger


def test_a_population_walk_handed_in_reverse_order_writes_the_same_rows() -> None:
    """The population's own order is not part of the book."""
    _, forward = _population_walk(3, _equal, band=BAND, dying=OTHER)
    _, backward = _population_walk(3, _equal, band=BAND, dying=OTHER, reverse=True)
    assert forward.rows == backward.rows


# ---------------------------------------------------------------------------
# What the ledger refuses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "band", [-0.125, 1.0, math.nan], ids=["negative", "one", "nan"]
)
def test_a_band_outside_zero_to_one_is_refused(band: float) -> None:
    """The band is a fraction of book equity in [0, 1); no value is defaulted."""
    with pytest.raises(ValueError, match="band"):
        _open(band)


def test_a_book_that_opens_with_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="open"):
        _ledger().BookLedger.open(
            instrument=INSTRUMENT,
            band=NO_BAND,
            cost=_charging().cost_provenance(),
            opening_equity=0.0,
        )


def test_the_ledger_keeps_only_the_books_own_account() -> None:
    """Any account but the reserved identity's is refused, naming the reservation."""
    bar = _bar(0, 100.0, 100.0)
    with pytest.raises(ValueError, match=BOOK_ACCOUNT_ID):
        _open().decide(
            bar=bar,
            account=_book_account(
                bar.period_end, OPENING_EQUITY, 0.0, agent_id=FOUNDER
            ),
            agent_ids=("a",),
            weights=np.asarray([1.0]),
            permitted=np.asarray([1.0]),
            risk=_Recording(),
        )


def test_a_bar_in_another_instrument_is_refused() -> None:
    bar = _bar(0, 100.0, 100.0, instrument="SOMEWHERE-ELSE")
    with pytest.raises(ValueError, match="SOMEWHERE-ELSE"):
        _open().decide(
            bar=bar,
            account=_book_account(bar.period_end, OPENING_EQUITY, 0.0),
            agent_ids=("a",),
            weights=np.asarray([1.0]),
            permitted=np.asarray([1.0]),
            risk=_Recording(),
        )


@pytest.mark.parametrize(
    ("ids", "weights", "permitted"),
    [
        (("a", "b"), (1.0,), (1.0, 0.0)),
        (("a", "b"), (0.5, 0.5), (1.0,)),
        (("a", "a"), (0.5, 0.5), (1.0, 0.0)),
    ],
    ids=["weights-misaligned", "permitted-misaligned", "duplicate-agent"],
)
def test_a_batch_that_is_not_one_figure_per_living_agent_is_refused(
    ids: tuple[str, ...], weights: tuple[float, ...], permitted: tuple[float, ...]
) -> None:
    bar = _bar(0, 100.0, 100.0)
    with pytest.raises(ValueError, match="agent"):
        _open().decide(
            bar=bar,
            account=_book_account(bar.period_end, OPENING_EQUITY, 0.0),
            agent_ids=ids,
            weights=np.asarray(weights),
            permitted=np.asarray(permitted),
            risk=_Recording(),
        )


def test_the_reserved_identity_holds_no_share_of_the_book() -> None:
    bar = _bar(0, 100.0, 100.0)
    with pytest.raises(ValueError, match=BOOK_ACCOUNT_ID):
        _open().decide(
            bar=bar,
            account=_book_account(bar.period_end, OPENING_EQUITY, 0.0),
            agent_ids=(BOOK_ACCOUNT_ID,),
            weights=np.asarray([1.0]),
            permitted=np.asarray([1.0]),
            risk=_Recording(),
        )


def test_a_port_answer_for_another_book_is_refused() -> None:
    """What the port returns must be about the book it was handed."""
    walk = _Walk(_open(), risk=_AnswersAnotherBook())
    with pytest.raises(ValueError, match="book"):
        walk.step(100.0, 100.0, ("a",), (1.0,), (0.5,))


# ---------------------------------------------------------------------------
# A plain value the loop holds; what the module may reach.
# ---------------------------------------------------------------------------


def test_the_ledger_state_is_a_plain_frozen_value_with_no_hidden_state() -> None:
    """Deciding twice from one state gives one answer; the state survives a round trip."""
    walk = _Walk(_open(BAND))
    walk.step(100.0, 100.0, ("a", "b"), (0.5, 0.5), (0.0, 0.0))
    state = walk.ledger
    restored = pickle.loads(pickle.dumps(state))
    assert restored == state
    with pytest.raises(FrozenInstanceError):
        state.band = NO_BAND

    # The book went flat, so the account it is handed next is the opening one.
    bar = _bar(1, 100.0, 101.0)
    account = _book_account(bar.period_end, OPENING_EQUITY, 0.0)
    answers = [
        ledger.decide(
            bar=bar,
            account=account,
            agent_ids=("a", "b"),
            weights=np.asarray([0.5, 0.5]),
            permitted=np.asarray([1.0, 0.5]),
            risk=_Recording(),
        )
        for ledger in (state, restored, state)
    ]
    assert answers[0] == answers[1] == answers[2]
    assert walk.ledger == restored


def test_the_decision_reads_no_evaluation_account() -> None:
    """The ledger is handed the book's account, the weights and the permitted targets only."""
    import inspect

    parameters = inspect.signature(_ledger().BookLedger.decide).parameters
    assert set(parameters) == {
        "self",
        "bar",
        "account",
        "agent_ids",
        "weights",
        "permitted",
        "risk",
    }


def _module_tree() -> ast.Module:
    module = _ledger()
    assert module.__file__ is not None
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_ledger_reaches_no_risk_authority_and_no_evaluation_account() -> None:
    """No ``liq.risk`` import; no module that keeps an agent's account."""
    imported: set[str] = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    forbidden = {
        "account_state",
        "accounts",
        "population_accounts",
        "steady_state",
    }
    assert not any(name.startswith("liq.risk") for name in imported), imported
    assert not any(name.rsplit(".", 1)[-1] in forbidden for name in imported), imported


def test_the_ledger_is_batched_over_the_population_with_no_loop_per_agent() -> None:
    """Array-batched: no ``for`` statement and no comprehension anywhere in the module."""
    looping = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp)
    looping += (ast.DictComp, ast.GeneratorExp)
    found = [
        f"{type(node).__name__} at line {node.lineno}"
        for node in ast.walk(_module_tree())
        if isinstance(node, looping)
    ]
    assert found == []
