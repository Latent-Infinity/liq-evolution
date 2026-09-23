"""How the execution adapter behaves on inputs this repository can construct.

The claim that this adapter produces the right numbers on a real tape is
evidence and lives beside the approved fixtures, in the consumer repository.
What is checked here is the other half: that the adapter answers the port's
contract, that the two outcomes it can produce are both reachable, that the
scenario it is handed by name is the scenario it charges — including the leg
this book does not trade — and that the settings which would reintroduce
look-ahead are refused rather than available.

The bar below is the contract suite's declared probe bar, not market data. No
number computed from it says anything about any instrument; what it exercises
is the shape of the translation, which is what this repository can see.

The cost scenarios below are declared for these checks and are not the cost
book. This library does not reach the book — the catalogue arrives as
configuration from wherever a run is assembled — so what is asserted here is
that whatever the catalogue states is what gets charged, never that a particular
scenario states a particular number.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from liq.evolution.ecology.adapters import LiqSimExecutionAdapter
from liq.evolution.ecology.adapters.liq_sim import (
    BAR_DID_NOT_MATCH_THE_ORDER,
    HEDGE_CHARGED_NOT_TRADED,
    HEDGE_NOT_IN_SCENARIO,
    HEDGE_PER_SIDE_BPS,
    NOTHING_TO_TRADE,
    ROUND_TRIP_BPS,
    SIDES_PER_ROUND_TRIP,
    VENUE_POSITION_BOUND_BREACHED,
    VENUE_SHORT_NOT_PERMITTED,
    CostScenarioUnusable,
    OrderNotExpressible,
    TargetNotYetActionable,
)
from liq.evolution.ecology.types import (
    AccountState,
    Bar,
    Fill,
    NotFilled,
    PositionTarget,
)
from tests.support.contract.test_execution_simulator import (
    ACCOUNTS,
    BAR,
    INSTRUMENT,
    PERIOD_END,
    PERIOD_START,
    ExecutionSimulatorContract,
    UnreachableTarget,
    account_holding,
    permitted,
)

#: A declared scenario with a hedge leg this book does not trade, and one
#: without. Neither is the cost book's; see the module docstring.
WITH_HEDGE = "declared-with-hedge"
WITHOUT_HEDGE = "declared-without-hedge"
ROUND_TRIP = 6.0
HEDGE_PER_SIDE = 0.5

SCENARIOS = {
    WITH_HEDGE: {ROUND_TRIP_BPS: ROUND_TRIP, HEDGE_PER_SIDE_BPS: HEDGE_PER_SIDE},
    WITHOUT_HEDGE: {ROUND_TRIP_BPS: ROUND_TRIP},
    "declared-without-a-round-trip": {HEDGE_PER_SIDE_BPS: HEDGE_PER_SIDE},
}

#: Wide enough that every exposure the contract suite permits is within it.
WHOLE_ACCOUNT = 1.0

#: Narrow enough that the contract suite's largest permitted exposure is not.
NARROW_BOUND = 0.25

BASIS_POINTS_PER_UNIT = 10_000.0


def adapter(
    scenario: str = WITH_HEDGE,
    *,
    maximum_position_exposure: float = WHOLE_ACCOUNT,
) -> LiqSimExecutionAdapter:
    """An adapter over a declared scenario, bounded as the caller asks."""
    return LiqSimExecutionAdapter(
        cost_scenario_id=scenario,
        scenarios=SCENARIOS,
        maximum_position_exposure=maximum_position_exposure,
    )


def out_of_reach() -> UnreachableTarget:
    """The declared out-of-reach case: more position than the venue's bound allows.

    The refusal comes from the execution model's own position bound, not from
    the mandate: a target the risk boundary refused would never reach this port
    at all, so a case produced there would leave this one untested.
    """
    return UnreachableTarget(
        simulator=adapter(maximum_position_exposure=NARROW_BOUND),
        target=permitted(WHOLE_ACCOUNT),
        bar=BAR,
        account=ACCOUNTS[0],
    )


class TestLiqSimExecutionAdapter(ExecutionSimulatorContract):
    """The simulator-backed model, held to the contract every model answers."""

    adapter_factory = staticmethod(adapter)
    unreachable_target = staticmethod(out_of_reach)


def test_the_scenario_is_applied_as_written_including_the_leg_not_traded() -> None:
    """A parameter silently dropped would make the charge one the harness chose."""
    with_hedge = adapter(WITH_HEDGE).cost_provenance()
    without_hedge = adapter(WITHOUT_HEDGE).cost_provenance()
    assert without_hedge.effective_round_trip_bps == pytest.approx(ROUND_TRIP)
    assert without_hedge.hedge_leg == HEDGE_NOT_IN_SCENARIO
    assert with_hedge.effective_round_trip_bps == pytest.approx(
        ROUND_TRIP + SIDES_PER_ROUND_TRIP * HEDGE_PER_SIDE
    )
    assert with_hedge.hedge_leg == HEDGE_CHARGED_NOT_TRADED


def test_the_charge_is_the_effective_rate_on_the_notional_that_moved() -> None:
    """The reported effective rate is the rate the venue actually charges."""
    executed = adapter(WITH_HEDGE)
    account = ACCOUNTS[0]
    outcome, new_state = executed.execute(permitted(0.5), BAR, account)
    assert isinstance(outcome, Fill)
    per_side = (
        executed.cost_provenance().effective_round_trip_bps / SIDES_PER_ROUND_TRIP
    )
    expected = 0.5 * account.equity * per_side / BASIS_POINTS_PER_UNIT
    assert outcome.cost == pytest.approx(expected)
    assert new_state.costs_charged == pytest.approx(account.costs_charged + expected)
    assert new_state.equity == pytest.approx(account.equity - expected)


def test_a_fill_prices_where_the_execution_model_prices_it() -> None:
    """The price is the simulator's, and it is inside the bar it was acted on."""
    outcome, _ = adapter().execute(permitted(0.5), BAR, ACCOUNTS[0])
    assert isinstance(outcome, Fill)
    assert outcome.price == pytest.approx(BAR.open)
    assert BAR.low <= outcome.price <= BAR.high
    assert outcome.as_of == BAR.period_end


def test_a_target_already_held_trades_nothing_and_says_so() -> None:
    """Reaching a target that is already held needs no trade and charges none."""
    account = account_holding(**{INSTRUMENT: 0.5})
    outcome, new_state = adapter().execute(permitted(0.5), BAR, account)
    assert isinstance(outcome, NotFilled)
    assert outcome.reason == NOTHING_TO_TRADE
    assert outcome.held_exposure == outcome.requested_exposure
    assert new_state.costs_charged == account.costs_charged
    assert dict(new_state.exposures) == dict(account.exposures)


def test_the_venue_position_bound_refuses_a_target_beyond_it() -> None:
    """The refusal is the execution model's own bound, named by a stable code."""
    outcome, new_state = adapter(maximum_position_exposure=NARROW_BOUND).execute(
        permitted(WHOLE_ACCOUNT), BAR, ACCOUNTS[0]
    )
    assert isinstance(outcome, NotFilled)
    assert outcome.reason == VENUE_POSITION_BOUND_BREACHED
    assert outcome.held_exposure != outcome.requested_exposure
    assert new_state.equity == ACCOUNTS[0].equity


def test_the_venue_refuses_to_create_a_short() -> None:
    """A venue configured without shorts refuses one, whatever the mandate allowed."""
    short = PositionTarget(
        agent_id="contract-agent",
        instrument=INSTRUMENT,
        target_exposure=-0.5,
        as_of=PERIOD_START,
    )
    outcome, _ = adapter().execute(short, BAR, ACCOUNTS[0])
    assert isinstance(outcome, NotFilled)
    assert outcome.reason == VENUE_SHORT_NOT_PERMITTED


def test_reducing_a_position_to_flat_is_not_a_short() -> None:
    """Selling what is held is permitted; only creating a short is not."""
    account = account_holding(**{INSTRUMENT: 0.3})
    outcome, _ = adapter().execute(permitted(0.0), BAR, account)
    assert isinstance(outcome, Fill)
    assert outcome.filled_exposure == 0.0


def test_a_target_acted_on_inside_the_bar_it_was_formed_on_is_refused() -> None:
    """This is the look-ahead the delay exists for, and it is refused, not absorbed."""
    formed_at_the_close = PositionTarget(
        agent_id="contract-agent",
        instrument=INSTRUMENT,
        target_exposure=0.5,
        as_of=PERIOD_END,
    )
    with pytest.raises(TargetNotYetActionable):
        adapter().execute(formed_at_the_close, BAR, ACCOUNTS[0])


def test_a_target_formed_after_the_bar_is_refused() -> None:
    """A target the bar could not have known about is the same defect, louder."""
    later = PositionTarget(
        agent_id="contract-agent",
        instrument=INSTRUMENT,
        target_exposure=0.5,
        as_of=PERIOD_END + timedelta(minutes=5),
    )
    with pytest.raises(TargetNotYetActionable):
        adapter().execute(later, BAR, ACCOUNTS[0])


def test_a_target_aged_longer_than_the_delay_is_actionable() -> None:
    """The delay is a minimum, so an older target is acted on rather than refused."""
    older = PositionTarget(
        agent_id="contract-agent",
        instrument=INSTRUMENT,
        target_exposure=0.5,
        as_of=PERIOD_START - timedelta(minutes=3),
    )
    outcome, _ = adapter().execute(older, BAR, ACCOUNTS[0])
    assert isinstance(outcome, Fill)


def test_a_target_that_has_not_aged_a_longer_declared_delay_is_refused() -> None:
    """The delay the model is configured with is the one enforced, not a fixed one."""
    slower = LiqSimExecutionAdapter(
        cost_scenario_id=WITH_HEDGE, scenarios=SCENARIOS, minimum_delay_bars=2
    )
    with pytest.raises(TargetNotYetActionable, match="requires 2"):
        slower.execute(permitted(0.5), BAR, ACCOUNTS[0])


def test_the_setting_that_would_fill_on_the_decision_bar_is_not_expressible() -> None:
    """A delay below one bar is the measured look-ahead, so it cannot be configured."""
    with pytest.raises(ValueError, match="minimum_delay_bars"):
        LiqSimExecutionAdapter(
            cost_scenario_id=WITH_HEDGE, scenarios=SCENARIOS, minimum_delay_bars=0
        )


def test_a_position_bound_that_permits_nothing_is_refused() -> None:
    """A bound of zero would refuse every order for a reason nobody configured."""
    with pytest.raises(ValueError, match="maximum_position_exposure"):
        LiqSimExecutionAdapter(
            cost_scenario_id=WITH_HEDGE,
            scenarios=SCENARIOS,
            maximum_position_exposure=0.0,
        )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("", "cost scenario id is empty"),
        ("not-in-the-catalogue", "not in the catalogue"),
        ("declared-without-a-round-trip", f"states no '{ROUND_TRIP_BPS}'"),
    ],
)
def test_a_scenario_that_cannot_be_charged_refuses_to_run(
    scenario: str, expected: str
) -> None:
    """A run that cannot resolve its costs does not start with a defaulted one."""
    with pytest.raises(CostScenarioUnusable, match=expected):
        LiqSimExecutionAdapter(cost_scenario_id=scenario, scenarios=SCENARIOS)


def test_a_bar_with_no_duration_states_no_delay_to_measure() -> None:
    """Without a width there is no number of bars between two instants."""
    instant = Bar(
        instrument=INSTRUMENT,
        period_start=PERIOD_START,
        period_end=PERIOD_START,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000.0,
    )
    with pytest.raises(OrderNotExpressible, match="no duration"):
        adapter().execute(permitted(0.5), instant, ACCOUNTS[0])


def test_an_instrument_the_venue_cannot_name_is_refused() -> None:
    """A provider's own validation failure travels out as an ecology error."""
    unnameable = Bar(
        instrument="not a symbol",
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1_000.0,
    )
    target = PositionTarget(
        agent_id="contract-agent",
        instrument="not a symbol",
        target_exposure=0.5,
        as_of=PERIOD_START,
    )
    with pytest.raises(OrderNotExpressible):
        adapter().execute(target, unnameable, ACCOUNTS[0])


def test_an_account_with_no_equity_has_nothing_to_move() -> None:
    """An order for no units is no order, and it is reported as nothing traded."""
    empty = AccountState(
        agent_id="contract-agent",
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        equity=0.0,
        exposures={},
        costs_charged=0.0,
    )
    outcome, _ = adapter().execute(permitted(0.5), BAR, empty)
    assert isinstance(outcome, NotFilled)
    assert outcome.reason == NOTHING_TO_TRADE


def test_a_bar_that_matches_no_order_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The execution model can decline to match, and declining is not a fill.

    The model matches every market order it is given, so the only way to see the
    outcome it would produce for one it declines is to make it decline. What is
    checked is this adapter's own translation of that answer, not the model's
    decision to give it.
    """
    from liq.evolution.ecology.adapters import liq_sim

    monkeypatch.setattr(liq_sim, "match_order", lambda *args, **kwargs: None)
    outcome, new_state = adapter().execute(permitted(0.5), BAR, ACCOUNTS[0])
    assert isinstance(outcome, NotFilled)
    assert outcome.reason == BAR_DID_NOT_MATCH_THE_ORDER
    assert new_state.costs_charged == ACCOUNTS[0].costs_charged
