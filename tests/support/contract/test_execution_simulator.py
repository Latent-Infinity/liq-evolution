"""The contract every execution model is run against.

Tier 2. Acting on a permitted target yields an outcome and a new account state,
and the pair has to agree: the realised exposure is what the account then holds,
the charge is the account's charge, and both name the cost scenario the adapter
was configured with. Account state goes in and comes back out, so a run can
resume from a state that was handed over rather than from a provider's private
memory.

**Two outcomes, and only two.** Either something traded — a fill, priced inside
the bar it was acted on and charged under the named scenario — or nothing did,
and the outcome says so and names why. Nothing here says what a fill should cost
or where inside the bar it prints; those belong to the scenario and the model.
What is contracted is that the outcome reports what happened rather than what
was asked, that a non-fill charges nothing and moves nothing, and that neither
kind is ever silent about its reason.

The non-fill half is asserted structurally rather than by naming a type, because
what makes it honest is what it *cannot* carry: no cost, and no price, because
no trade happened at one. A type check would pass a value that carried both.

Which targets a model can and cannot reach is the model's business, so a
registration may declare the case its own bound refuses. Declaring nothing is
not a way out: an adapter that declares no case of its own is held to a target
beyond any mandate, which every execution model must refuse. Either way the
declaration cannot become an opt-out, because a case the model actually fills
fails the suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import ClassVar

import pytest

from liq.evolution.ecology import (
    AccountState,
    Bar,
    ExecutionSimulator,
    Fill,
    PositionTarget,
)
from liq.evolution.ecology.adapters import NullExecutionSimulator

AGENT = "contract-agent"
INSTRUMENT = "CONTRACT-A"
OTHER_INSTRUMENT = "CONTRACT-B"
PERIOD_START = datetime(2000, 1, 1, tzinfo=UTC)
PERIOD_END = datetime(2000, 1, 1, 0, 1, tzinfo=UTC)

# A declared probe bar, not market data: a flat body inside a unit range at a
# constant volume. Only its ordering in time and its high-low bracket are read.
BAR = Bar(
    instrument=INSTRUMENT,
    period_start=PERIOD_START,
    period_end=PERIOD_END,
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1_000.0,
)

# Exposures a long-only mandate can permit. A short is deliberately absent: no
# mandate in this design permits one, so a short target reaching an execution
# model is a case the wired path cannot produce, and a suite that ran it would
# be reporting arithmetic as though it were evidence about execution.
PERMITTED_EXPOSURES = (0.0, 0.5, 1.0)

#: How much of a round trip one call to :meth:`execute` can trade. A target is
#: reached by moving to it once, so a call charges one side of a round trip.
SIDES_PER_ROUND_TRIP = 2

#: Basis points to a fraction of notional.
BASIS_POINTS_PER_UNIT = 10_000.0


def account_holding(**exposures: float) -> AccountState:
    """Return the probe account that already holds ``exposures``."""
    return AccountState(
        agent_id=AGENT,
        as_of=PERIOD_START,
        equity=1.0,
        exposures=MappingProxyType(dict(exposures)),
        costs_charged=0.0,
    )


ACCOUNTS = (
    account_holding(),
    account_holding(**{OTHER_INSTRUMENT: 0.4}),
    account_holding(**{INSTRUMENT: 0.3}),
)
ACCOUNT_IDS = ("flat", "elsewhere", "already-held")


def permitted(exposure: float) -> PositionTarget:
    """Return the probe target permitting ``exposure`` in the probe instrument.

    The target is stamped at the start of the bar it is handed with, not at its
    end. A target formed at a bar's close and acted on inside that same bar
    would be acted on at prices that printed before it was formed, which is the
    look-ahead this whole boundary exists to make impossible; the instant a
    decision is taken is therefore the instant the bar that acts on it opens.
    """
    return PositionTarget(
        agent_id=AGENT,
        instrument=INSTRUMENT,
        target_exposure=exposure,
        as_of=PERIOD_START,
    )


@dataclass(frozen=True)
class UnreachableTarget:
    """One case an adapter declares its model cannot reach.

    The whole case is declared, not just the target: which targets are out of
    reach depends on how the model is configured, so the registration hands over
    the configured model together with the target, bar and account that put it
    out of reach.

    Attributes:
        simulator: The model, configured so that ``target`` is out of its reach.
        target: The exposure that cannot be reached.
        bar: The bar the target would be acted on.
        account: The account state it would be acted against.
    """

    simulator: ExecutionSimulator
    target: PositionTarget
    bar: Bar
    account: AccountState


#: Twice the whole evaluation account in one instrument. No mandate in this
#: design permits it and no venue accepts an order for it, so it is out of reach
#: of any execution model worth the name — which is what makes it usable as the
#: case an adapter is held to when it declares no narrower one of its own.
BEYOND_ANY_MANDATE = 2.0


def beyond_any_mandate(simulator: ExecutionSimulator) -> UnreachableTarget:
    """The case every model is held to: more than the whole account, twice over."""
    return UnreachableTarget(
        simulator=simulator,
        target=permitted(BEYOND_ANY_MANDATE),
        bar=BAR,
        account=ACCOUNTS[0],
    )


def traded(outcome: object) -> bool:
    """Whether ``outcome`` says a trade happened."""
    return isinstance(outcome, Fill)


class ExecutionSimulatorContract:
    """What an execution model promises, whichever fills it models.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyExecutionSimulator(ExecutionSimulatorContract):
            adapter_factory = staticmethod(MyExecutionSimulator)

    An adapter that applies a bound of its own may also declare the case that
    bound refuses, which is what exercises the bound rather than the extreme::

        class TestMyExecutionSimulator(ExecutionSimulatorContract):
            adapter_factory = staticmethod(MyExecutionSimulator)
            unreachable_target = staticmethod(my_narrower_case)

    Declaring nothing is not a way out of the out-of-reach half of the contract:
    an adapter that declares no case of its own is held to
    :data:`BEYOND_ANY_MANDATE`, which every execution model must refuse.
    """

    adapter_factory: ClassVar[Callable[[], ExecutionSimulator]]
    unreachable_target: ClassVar[Callable[[], UnreachableTarget] | None] = None

    @pytest.fixture
    def adapter(self) -> ExecutionSimulator:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    @pytest.fixture
    def unreachable(self) -> UnreachableTarget:
        """The case this adapter cannot reach: its own, or the one every model shares."""
        declared = type(self).unreachable_target
        if declared is None:
            return beyond_any_mandate(type(self).adapter_factory())
        return declared()

    def test_the_adapter_satisfies_the_port(self, adapter: ExecutionSimulator) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, ExecutionSimulator)

    def test_the_adapter_names_the_cost_scenario_it_was_configured_with(
        self, adapter: ExecutionSimulator
    ) -> None:
        """The scenario is resolved outside the ecology and carried by name."""
        assert isinstance(adapter.cost_scenario_id, str)
        assert adapter.cost_scenario_id != ""

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_acting_yields_an_outcome_and_the_state_that_follows_it(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """Both halves of the outcome are returned; neither is left implicit."""
        outcome, new_state = adapter.execute(permitted(exposure), BAR, account)
        assert outcome is not None
        assert isinstance(new_state, AccountState)

    def test_the_declared_probe_targets_are_reachable_at_least_once(
        self, adapter: ExecutionSimulator
    ) -> None:
        """A model that never fills would make every fill case below vacuous."""
        outcomes = [
            adapter.execute(permitted(exposure), BAR, account)[0]
            for exposure in PERMITTED_EXPOSURES
            for account in ACCOUNTS
        ]
        assert any(traded(outcome) for outcome in outcomes)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_outcome_is_recorded_under_the_scenario_the_adapter_names(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A result can be re-costed because every outcome says what it was costed under."""
        outcome, _ = adapter.execute(permitted(exposure), BAR, account)
        assert outcome.cost_scenario_id == adapter.cost_scenario_id
        if traded(outcome):
            assert outcome.cost >= 0.0
            assert math.isfinite(outcome.cost)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_outcome_reports_what_was_asked_and_what_was_reached(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """Scoring reads the realised value, so both are kept and neither is lost."""
        target = permitted(exposure)
        outcome, _ = adapter.execute(target, BAR, account)
        assert outcome.agent_id == target.agent_id
        assert outcome.instrument == target.instrument
        assert outcome.requested_exposure == target.target_exposure
        reached = outcome.filled_exposure if traded(outcome) else outcome.held_exposure
        assert math.isfinite(reached)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_a_fill_is_priced_and_stamped_inside_the_bar_it_was_acted_on(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A price outside the bar, or an instant outside it, is not a fill of it."""
        outcome, _ = adapter.execute(permitted(exposure), BAR, account)
        if traded(outcome):
            assert BAR.low <= outcome.price <= BAR.high
        assert BAR.period_start <= outcome.as_of <= BAR.period_end

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_new_state_records_the_exposure_that_was_reached(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """The account holds what happened, not what the target asked for."""
        target = permitted(exposure)
        outcome, new_state = adapter.execute(target, BAR, account)
        held = account.exposures.get(target.instrument, 0.0)
        reached = outcome.filled_exposure if traded(outcome) else held
        assert new_state.exposures.get(target.instrument, 0.0) == reached
        untouched = {
            instrument: exposure
            for instrument, exposure in account.exposures.items()
            if instrument != target.instrument
        }
        for instrument, exposure_held in untouched.items():
            assert new_state.exposures[instrument] == exposure_held

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_charge_reaches_the_account_exactly_once(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A cost that is double-counted or dropped breaks every net number."""
        outcome, new_state = adapter.execute(permitted(exposure), BAR, account)
        charged = new_state.costs_charged - account.costs_charged
        expected = outcome.cost if traded(outcome) else 0.0
        assert charged == pytest.approx(expected)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_new_state_belongs_to_the_same_agent_and_is_not_stale(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """State is handed back for the agent it was handed in for, moved forward."""
        _, new_state = adapter.execute(permitted(exposure), BAR, account)
        assert new_state.agent_id == account.agent_id
        assert new_state.as_of >= account.as_of
        assert math.isfinite(new_state.equity)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_no_hidden_state_is_kept_between_calls(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A run resumes from a state that was handed out, not private memory."""
        first_outcome, first_state = adapter.execute(permitted(exposure), BAR, account)
        second_outcome, second_state = adapter.execute(
            permitted(exposure), BAR, account
        )
        assert first_outcome == second_outcome
        assert first_state == second_state

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    def test_acting_leaves_the_state_it_was_handed_alone(
        self, adapter: ExecutionSimulator, account: AccountState
    ) -> None:
        """The handed-in state stays readable as it was, for the record it is in."""
        exposures = dict(account.exposures)
        equity = account.equity
        costs_charged = account.costs_charged
        adapter.execute(permitted(0.5), BAR, account)
        assert dict(account.exposures) == exposures
        assert account.equity == equity
        assert account.costs_charged == costs_charged

    def test_an_unreachable_target_is_reported_as_such(
        self, unreachable: UnreachableTarget
    ) -> None:
        """A target out of reach is never reported as though it had been reached.

        This is the case the whole boundary exists for: an execution model that
        answered the request back would make every score a score of what was
        wanted rather than of what was held, and nothing downstream could tell
        the difference.
        """
        outcome, new_state = unreachable.simulator.execute(
            unreachable.target, unreachable.bar, unreachable.account
        )
        assert not traded(outcome)
        assert outcome.requested_exposure == unreachable.target.target_exposure
        held = unreachable.account.exposures.get(unreachable.target.instrument, 0.0)
        assert outcome.held_exposure == held

    def test_an_unreachable_target_charges_nothing_and_moves_nothing(
        self, unreachable: UnreachableTarget
    ) -> None:
        """Nothing traded, so nothing is charged and nothing is held differently."""
        _, new_state = unreachable.simulator.execute(
            unreachable.target, unreachable.bar, unreachable.account
        )
        account = unreachable.account
        assert new_state.costs_charged == pytest.approx(account.costs_charged)
        assert new_state.equity == pytest.approx(account.equity)
        assert dict(new_state.exposures) == dict(account.exposures)

    def test_an_unreachable_target_carries_no_price_and_no_charge(
        self, unreachable: UnreachableTarget
    ) -> None:
        """No trade happened, so there is no price it happened at and no cost of it."""
        outcome, _ = unreachable.simulator.execute(
            unreachable.target, unreachable.bar, unreachable.account
        )
        assert not hasattr(outcome, "price")
        assert not hasattr(outcome, "cost")

    def test_an_unreachable_target_names_a_stable_reason(
        self, unreachable: UnreachableTarget
    ) -> None:
        """Silence is the failure this replaces, so the reason is a code, not prose."""
        first, _ = unreachable.simulator.execute(
            unreachable.target, unreachable.bar, unreachable.account
        )
        second, _ = unreachable.simulator.execute(
            unreachable.target, unreachable.bar, unreachable.account
        )
        assert isinstance(first.reason, str)
        assert first.reason != ""
        assert first.reason == second.reason

    def test_the_adapter_reports_what_the_named_scenario_effectively_charges(
        self, adapter: ExecutionSimulator
    ) -> None:
        """A scenario parameter silently ignored is a partial inline cost decision.

        The effective rate is what is actually charged over a round trip, which
        is not always what the scenario's headline round trip says: a scenario
        whose legs this book does not trade is still applied as written, and the
        difference has to be readable rather than reasoned about.
        """
        provenance = adapter.cost_provenance()
        assert provenance.cost_scenario_id == adapter.cost_scenario_id
        assert math.isfinite(provenance.effective_round_trip_bps)
        assert provenance.effective_round_trip_bps >= 0.0
        assert isinstance(provenance.hedge_leg, str)
        assert provenance.hedge_leg != ""

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_a_fill_is_charged_at_the_effective_rate_the_adapter_reports(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """The reported rate is the one applied, so reporting it is not a claim apart."""
        target = permitted(exposure)
        outcome, _ = adapter.execute(target, BAR, account)
        if not traded(outcome):
            pytest.skip("nothing traded, so there is no charge to account for")
        per_side = (
            adapter.cost_provenance().effective_round_trip_bps / SIDES_PER_ROUND_TRIP
        )
        moved = abs(
            outcome.filled_exposure - account.exposures.get(target.instrument, 0.0)
        )
        expected = moved * account.equity * per_side / BASIS_POINTS_PER_UNIT
        assert outcome.cost == pytest.approx(expected)


class TestNullExecutionSimulator(ExecutionSimulatorContract):
    """The stand-in, held to the same contract as any provider-backed model."""

    adapter_factory = staticmethod(NullExecutionSimulator)
