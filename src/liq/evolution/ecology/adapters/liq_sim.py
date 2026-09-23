"""Acting on a permitted target through the platform's execution simulator.

This is the only module in this library that imports :mod:`liq.sim`. Everything
about whether an order fills, at what price, and what it is charged is asked of
the simulator and its models; nothing about it is decided here. What is decided
here is translation — an exposure the ecology permits becomes an order in the
simulator's vocabulary, and what the simulator answers becomes an outcome in the
ecology's.

**Why the simulator's own event loop is not what is driven.** Its loop takes a
sequence of bars and decides eligibility from indices into that sequence, so a
call carrying a single bar can never satisfy a delay of one bar: the difference
between an order's origin index and the current index is structurally zero, and
the only way to obtain a fill from a one-bar call is to set the minimum delay to
zero — which fills the order on the bar it was decided on, at prices that
printed before the decision. That is measured look-ahead, and it is why
:attr:`LiqSimExecutionAdapter.minimum_delay_bars` refuses a value below one:
the setting that would produce it is not expressible through this adapter. The
simulator's matching, pricing, fee and constraint functions are driven directly
instead, and its own eligibility rule decides the delay.

**What crosses a bar boundary: nothing.** The port promises no hidden per-agent
state between calls, and this adapter keeps none, because the continuation state
a multi-bar run would need is either recoverable from what each call is handed
or not applicable to a call that opens no run:

* the absolute index an order's origin must be measured against is the bar's own
  UTC period, and a timestamp is absolute by construction — the difference in
  bars between the instant a target was formed and the instant a bar completes
  is recovered by dividing by the bar's own width, so no index has to be carried
  and none can drift out of a slice's frame of reference;
* the eligibility latch a multi-bar run keeps per order, keyed by object
  identity, is not needed: no order outlives the call that built it, and each is
  named deterministically from the target it came from;
* the running totals a multi-bar run accumulates — equity, exposures, charges —
  are the account state that is handed in and handed back;
* raw slippage observations are not reduced here at all. Percentiles do not
  compose across calls, so this adapter computes none and reports none.

**Which of the simulator's bounds are applied, and which deliberately are not.**
Applied: the position bound, which refuses an order that would take a position
past a declared fraction of equity; and the short permission, which refuses to
create a short at a venue configured without them. Not applied, each for a
reason rather than by omission: buying power, margin and gross leverage all need
a cash balance or a mark for every instrument held, and a call is handed one
bar, so a portfolio-level bound computed here would be computed from marks this
call was not given; the pattern-day-trade counter, the frequency cap, the equity
floor and the kill switch are all session- or run-scoped state, which is
precisely the state this adapter is documented above as not keeping. None of
them substitutes for the mandate: what an agent may hold is decided at the risk
boundary, before anything reaches here.

**Costs.** No cost number is written down in this module. The scenario is named
by id and resolved, inside this adapter, against the catalogue of scenarios it
is configured with, and every figure charged comes from that scenario. The
scenario is applied *as written*, including a leg this book does not trade: a
hedge charge in the scenario is charged even though no hedge is traded, because
a harness that silently drops a scenario parameter has chosen the cost itself.
What that adds up to is readable through :meth:`cost_provenance` rather than
left to be reasoned about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal, DivisionByZero, InvalidOperation
from types import MappingProxyType
from uuid import NAMESPACE_URL, uuid5

from liq.core import (
    Bar as ProviderBar,
)
from liq.core import (
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioState,
    Position,
)
from liq.evolution.ecology.errors import EcologyError, translating
from liq.evolution.ecology.types import (
    AccountState,
    Bar,
    CostProvenance,
    CostScenarioId,
    Fill,
    NotFilled,
    PositionTarget,
)
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.sim.constraints import (
    ConstraintViolation,
    check_position_limit,
    check_short_permission,
)
from liq.sim.exceptions import LookAheadBiasError
from liq.sim.execution import match_order
from liq.sim.providers import fee_model_from_config, slippage_model_from_config
from liq.sim.validation import assert_no_lookahead, is_order_eligible

__all__ = [
    "BAR_DID_NOT_MATCH_THE_ORDER",
    "DEFAULT_MAXIMUM_POSITION_EXPOSURE",
    "DEFAULT_MINIMUM_DELAY_BARS",
    "HEDGE_CHARGED_NOT_TRADED",
    "HEDGE_NOT_IN_SCENARIO",
    "HEDGE_PER_SIDE_BPS",
    "NOTHING_TO_TRADE",
    "ROUND_TRIP_BPS",
    "SIDES_PER_ROUND_TRIP",
    "VENUE_POSITION_BOUND_BREACHED",
    "VENUE_SHORT_NOT_PERMITTED",
    "CostScenarioUnusable",
    "LiqSimExecutionAdapter",
    "OrderNotExpressible",
    "TargetNotYetActionable",
]

#: The scenario parameter naming what a round trip costs, in basis points.
ROUND_TRIP_BPS = "round_trip_bps"

#: The scenario parameter naming what one side of a hedge costs, in basis
#: points. This book trades no hedge; see the module docstring for why it is
#: charged all the same when a scenario carries it.
HEDGE_PER_SIDE_BPS = "hedge_per_side_bps"

#: Unit arithmetic, not costs: a target is reached by moving to it once, so one
#: call trades one of a round trip's two sides.
SIDES_PER_ROUND_TRIP = 2

#: The scenario carried a hedge leg, and it was charged although no hedge was
#: traded.
HEDGE_CHARGED_NOT_TRADED = "charged_not_traded"

#: The scenario carried no hedge leg, so there was none to charge.
HEDGE_NOT_IN_SCENARIO = "not_in_scenario"

#: The exposure asked for is the exposure already held, so nothing traded.
NOTHING_TO_TRADE = "target_already_held"

#: The venue's position bound refused the order this target would need.
VENUE_POSITION_BOUND_BREACHED = "venue_position_bound_breached"

#: The venue permits no shorts, and this target would have created one.
VENUE_SHORT_NOT_PERMITTED = "venue_short_not_permitted"

#: The bar did not match the order the target would need.
BAR_DID_NOT_MATCH_THE_ORDER = "bar_did_not_match_the_order"

# The simulator's own configuration defaults, read from it rather than restated,
# so that a change to them arrives here rather than diverging silently. Reading
# a default is not declaring a bound, and the two below are not this adapter's
# bounds: they are the simulator's numbers for a book, inherited by whatever
# composes this adapter and states nothing of its own. A composition that has to
# defend what its venue accepts supplies its own, and one that leaves these
# standing has chosen nothing at the level it is actually running at.
_SIMULATOR_DEFAULTS = SimulatorConfig()

#: Bars a target must age before it can be acted on, as the simulator's own
#: configuration requires by default.
DEFAULT_MINIMUM_DELAY_BARS = _SIMULATOR_DEFAULTS.min_order_delay_bars

#: The largest position, as a fraction of equity, the venue will accept an order
#: for, as the simulator's own configuration bounds it by default.
DEFAULT_MAXIMUM_POSITION_EXPOSURE = _SIMULATOR_DEFAULTS.max_position_pct

# The venue is configured to charge the resolved scenario as a commission and to
# add nothing else. The fee model turns basis points of notional into a charge,
# which is what a scenario states; the slippage model is given nothing to add,
# because a second charge on top would be a number the scenario does not supply.
_FEE_MODEL = "TieredMakerTaker"
_MAKER_BPS = "maker_bps"
_TAKER_BPS = "taker_bps"
_SLIPPAGE_MODEL = "PFOF"
_ADVERSE_BPS = "adverse_bps"
_NO_SEPARATE_SLIPPAGE = 0.0

# Order identities are derived from the target rather than drawn at random, so
# that the same target acted on against the same bar produces the same order.
_ORDER_NAMESPACE = NAMESPACE_URL


class CostScenarioUnusable(EcologyError):
    """The named cost scenario is absent, or states nothing this adapter can charge."""


class TargetNotYetActionable(EcologyError):
    """The target has not aged the delay the execution model requires.

    Acting on it against this bar would trade at prices that printed before the
    decision was taken. It is raised rather than reported as an outcome, because
    it says the caller handed over the wrong bar: an outcome would record a
    market fact, and this is a wiring defect.
    """


class OrderNotExpressible(EcologyError):
    """The target or bar could not be described to the execution model at all."""


@dataclass(frozen=True)
class _ResolvedScenario:
    """One cost scenario, resolved to what it charges per side and per round trip."""

    per_side_bps: float
    effective_round_trip_bps: float
    hedge_leg: str


@dataclass(frozen=True)
class LiqSimExecutionAdapter:
    """Act on permitted targets through the platform's execution simulator.

    Attributes:
        cost_scenario_id: The scenario every charge is drawn from, resolved by
            this adapter against ``scenarios``.
        scenarios: The cost book's scenarios, by id, each mapping parameter
            names to values. Supplied as configuration: this library does not
            reach the book itself, and a catalogue that is not the book's is a
            composition error rather than something this adapter can detect.
        venue: Name recorded as the venue the outcome came from.
        maximum_position_exposure: The largest position, as a fraction of
            equity, the venue accepts an order for.
        minimum_delay_bars: How many bars a target must age before it can be
            acted on. Values below one are refused: a target acted on inside the
            bar it was formed on trades at prices that printed before it.
    """

    cost_scenario_id: CostScenarioId
    scenarios: Mapping[CostScenarioId, Mapping[str, float]]
    venue: str = "liq-sim"
    maximum_position_exposure: float = DEFAULT_MAXIMUM_POSITION_EXPOSURE
    minimum_delay_bars: int = DEFAULT_MINIMUM_DELAY_BARS
    _scenario: _ResolvedScenario = field(init=False, repr=False, compare=False)
    _provider: ProviderConfig = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Resolve the named scenario and configure the venue it will be charged at."""
        if self.minimum_delay_bars < 1:
            raise ValueError(
                "minimum_delay_bars must be at least 1: a target acted on inside "
                "the bar it was formed on is filled at prices that printed "
                "before the decision, which is look-ahead and not a setting"
            )
        if self.maximum_position_exposure <= 0.0:
            raise ValueError("maximum_position_exposure must be positive")
        scenario = self._resolve(self.cost_scenario_id)
        object.__setattr__(self, "_scenario", scenario)
        object.__setattr__(
            self,
            "_provider",
            ProviderConfig(
                name=self.venue,
                asset_classes=["equity"],
                fee_model=_FEE_MODEL,
                fee_params={
                    _MAKER_BPS: scenario.per_side_bps,
                    _TAKER_BPS: scenario.per_side_bps,
                },
                slippage_model=_SLIPPAGE_MODEL,
                slippage_params={_ADVERSE_BPS: _NO_SEPARATE_SLIPPAGE},
                short_enabled=False,
            ),
        )

    def cost_provenance(self) -> CostProvenance:
        """Return what the named scenario effectively charges, as applied here."""
        return CostProvenance(
            cost_scenario_id=self.cost_scenario_id,
            effective_round_trip_bps=self._scenario.effective_round_trip_bps,
            hedge_leg=self._scenario.hedge_leg,
        )

    def execute(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
    ) -> tuple[Fill | NotFilled, AccountState]:
        """Act on ``target`` at ``bar`` and return the outcome and the new state."""
        self._refuse_look_ahead(target, bar)
        held = account.exposures.get(target.instrument, 0.0)
        wanted = target.target_exposure - held
        provider_bar = self._provider_bar(bar)
        mark = provider_bar.open
        quantity = self._quantity(wanted, account.equity, mark)
        if quantity is None:
            return self._nothing_traded(target, bar, account, held, NOTHING_TO_TRADE)
        order = self._order(target, quantity, wanted, mark)
        refused = self._venue_refusal(order, account, held, mark, provider_bar)
        if refused is not None:
            return self._nothing_traded(target, bar, account, held, *refused)
        filled = match_order(
            order,
            provider_bar,
            slippage=self._slippage(order, provider_bar),
            commission=self._commission(order, mark),
            provider=self.venue,
            timestamp=provider_bar.timestamp,
        )
        if filled is None:
            return self._nothing_traded(
                target, bar, account, held, BAR_DID_NOT_MATCH_THE_ORDER
            )
        return self._traded(
            target, bar, account, float(filled.price), float(filled.commission)
        )

    def _resolve(self, scenario_id: CostScenarioId) -> _ResolvedScenario:
        """Resolve ``scenario_id`` against the configured catalogue, or refuse to run."""
        if not scenario_id:
            raise CostScenarioUnusable(
                "cost scenario id is empty; every charge is drawn from a named "
                "scenario and none can be defaulted here"
            )
        parameters = self.scenarios.get(scenario_id)
        if parameters is None:
            raise CostScenarioUnusable(
                f"cost scenario '{scenario_id}' is not in the catalogue this "
                f"adapter was configured with; it carries {sorted(self.scenarios)}"
            )
        if ROUND_TRIP_BPS not in parameters:
            raise CostScenarioUnusable(
                f"cost scenario '{scenario_id}' states no '{ROUND_TRIP_BPS}', so "
                "this adapter cannot charge it; it carries "
                f"{sorted(parameters)}"
            )
        hedge = parameters.get(HEDGE_PER_SIDE_BPS)
        per_side = float(parameters[ROUND_TRIP_BPS]) / SIDES_PER_ROUND_TRIP
        if hedge is not None:
            per_side += float(hedge)
        return _ResolvedScenario(
            per_side_bps=per_side,
            effective_round_trip_bps=per_side * SIDES_PER_ROUND_TRIP,
            hedge_leg=(
                HEDGE_NOT_IN_SCENARIO if hedge is None else HEDGE_CHARGED_NOT_TRADED
            ),
        )

    def _refuse_look_ahead(self, target: PositionTarget, bar: Bar) -> None:
        """Refuse a target that has not aged the delay the execution model requires."""
        width = bar.period_end - bar.period_start
        if width <= timedelta(0):
            raise OrderNotExpressible(
                "a bar with no duration states no width to measure a delay in"
            )
        with translating(
            LookAheadBiasError,
            into=TargetNotYetActionable,
            action="checking the target against the bar that would act on it",
        ):
            assert_no_lookahead(target.as_of, bar.period_start)
        aged = (bar.period_end - target.as_of) // width
        if not is_order_eligible(0, aged, self.minimum_delay_bars):
            raise TargetNotYetActionable(
                f"the target aged {aged} bar(s) by the end of this bar, and the "
                f"execution model requires {self.minimum_delay_bars}: the bar "
                "that acts on a target is not the bar it was formed on"
            )

    def _provider_bar(self, bar: Bar) -> ProviderBar:
        """Describe ``bar`` in the execution model's vocabulary."""
        with translating(
            ValueError,
            into=OrderNotExpressible,
            action="describing the bar to the execution model",
        ):
            return ProviderBar(
                timestamp=bar.period_start,
                symbol=bar.instrument,
                open=_as_decimal(bar.open),
                high=_as_decimal(bar.high),
                low=_as_decimal(bar.low),
                close=_as_decimal(bar.close),
                volume=_as_decimal(bar.volume),
            )

    def _quantity(self, wanted: float, equity: float, mark: Decimal) -> Decimal | None:
        """Return the units that move exposure by ``wanted``, or ``None`` for none."""
        with translating(
            (ValueError, ArithmeticError),
            into=OrderNotExpressible,
            action="sizing the order the target would need",
        ):
            units = abs(_as_decimal(wanted)) * _as_decimal(equity) / mark
        return units if units > 0 else None

    def _order(
        self, target: PositionTarget, quantity: Decimal, wanted: float, mark: Decimal
    ) -> OrderRequest:
        """Describe the move ``target`` asks for as an order at the venue."""
        with translating(
            ValueError,
            into=OrderNotExpressible,
            action="describing the target to the execution model",
        ):
            return OrderRequest(
                client_order_id=uuid5(
                    _ORDER_NAMESPACE,
                    f"{target.agent_id}|{target.instrument}|{target.as_of.isoformat()}"
                    f"|{target.target_exposure!r}",
                ),
                symbol=target.instrument,
                side=OrderSide.BUY if wanted > 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                reference_price=mark,
                timestamp=target.as_of,
            )

    def _venue_refusal(
        self,
        order: OrderRequest,
        account: AccountState,
        held: float,
        mark: Decimal,
        provider_bar: ProviderBar,
    ) -> tuple[str, str] | None:
        """Ask the venue's own bounds about ``order``; report a refusal, or ``None``.

        Each bound is asked separately so that the code reported names the bound
        that bound, rather than collapsing every refusal into one.
        """
        portfolio = self._portfolio(account, held, mark, provider_bar)
        try:
            check_short_permission(
                order,
                portfolio,
                short_enabled=self._provider.short_enabled,
                locate_required=self._provider.locate_required,
            )
        except ConstraintViolation as refusal:
            return VENUE_SHORT_NOT_PERMITTED, type(refusal).__name__
        try:
            check_position_limit(
                order,
                portfolio,
                max_position_pct=self.maximum_position_exposure,
                mark_price=mark,
            )
        except ConstraintViolation as refusal:
            return VENUE_POSITION_BOUND_BREACHED, type(refusal).__name__
        return None

    def _portfolio(
        self,
        account: AccountState,
        held: float,
        mark: Decimal,
        provider_bar: ProviderBar,
    ) -> PortfolioState:
        """Describe the evaluation account in the execution model's vocabulary.

        Only the instrument this call is handed a bar for is described. Exposure
        held elsewhere has no mark in this call, and a portfolio restated from
        exposures without marks would be an accounting model invented here; the
        bounds that would need one are not applied, which is why none is built.
        """
        equity = _as_decimal(account.equity)
        positions: dict[str, Position] = {}
        cash = equity
        if held != 0.0:
            units = _as_decimal(held) * equity / mark
            positions[provider_bar.symbol] = Position(
                symbol=provider_bar.symbol,
                quantity=units,
                average_price=mark,
                realized_pnl=Decimal(0),
                timestamp=provider_bar.timestamp,
            )
            cash = equity - abs(units) * mark
        with translating(
            ValueError,
            into=OrderNotExpressible,
            action="describing the evaluation account to the execution model",
        ):
            return PortfolioState(
                cash=cash,
                positions=positions,
                timestamp=provider_bar.timestamp,
            )

    def _slippage(self, order: OrderRequest, provider_bar: ProviderBar) -> Decimal:
        """Ask the venue's slippage model what this order slips by."""
        return slippage_model_from_config(self._provider).calculate(order, provider_bar)

    def _commission(self, order: OrderRequest, mark: Decimal) -> Decimal:
        """Ask the venue's fee model what the resolved scenario charges this order."""
        return fee_model_from_config(self._provider).calculate(
            order, mark, is_maker=False
        )

    def _traded(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
        price: float,
        cost: float,
    ) -> tuple[Fill, AccountState]:
        """Report the whole order filled, which is the only fill this model produces.

        The order was sized to reach the target and the execution model fills an
        order in full or not at all, so a fill reaches the target exactly. There
        is no quantity between nothing and all of it for this model to return,
        and none is constructed here.
        """
        fill = Fill(
            agent_id=target.agent_id,
            instrument=target.instrument,
            as_of=bar.period_end,
            requested_exposure=target.target_exposure,
            filled_exposure=target.target_exposure,
            price=price,
            cost=cost,
            cost_scenario_id=self.cost_scenario_id,
        )
        exposures = dict(account.exposures)
        exposures[target.instrument] = fill.filled_exposure
        return fill, AccountState(
            agent_id=account.agent_id,
            as_of=fill.as_of,
            equity=account.equity - cost,
            exposures=MappingProxyType(exposures),
            costs_charged=account.costs_charged + cost,
        )

    def _nothing_traded(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
        held: float,
        reason: str,
        detail: str = "",
    ) -> tuple[NotFilled, AccountState]:
        """Report that nothing traded, leaving the account exactly as handed in."""
        outcome = NotFilled(
            agent_id=target.agent_id,
            instrument=target.instrument,
            as_of=bar.period_end,
            requested_exposure=target.target_exposure,
            held_exposure=held,
            reason=reason,
            cost_scenario_id=self.cost_scenario_id,
            detail=detail,
        )
        return outcome, AccountState(
            agent_id=account.agent_id,
            as_of=outcome.as_of,
            equity=account.equity,
            exposures=MappingProxyType(dict(account.exposures)),
            costs_charged=account.costs_charged,
        )


def _as_decimal(value: float) -> Decimal:
    """Return ``value`` as the exact decimal the execution model works in."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, DivisionByZero) as unusable:  # pragma: no cover
        raise OrderNotExpressible(
            f"{value!r} is not a quantity the venue can hold"
        ) from unusable
