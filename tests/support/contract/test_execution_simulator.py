"""The contract every execution model is run against.

Tier 2. Acting on a permitted target yields a fill and a new account state, and
the pair has to agree: the realised exposure is what the account then holds, the
charge is the account's charge, and both name the cost scenario the adapter was
configured with. Account state goes in and comes back out, so a run can resume
from a state that was handed over rather than from a provider's private memory.

Nothing here says what a fill should cost or where inside the bar it prints —
those belong to the scenario and the model. What is contracted is that a fill
is priced inside the bar it was formed on, reports what was reached rather than
what was asked, and is accounted exactly once.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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

PERMITTED_EXPOSURES = (0.0, 0.5, -0.5, 1.0)


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
    """Return the probe target permitting ``exposure`` in the probe instrument."""
    return PositionTarget(
        agent_id=AGENT,
        instrument=INSTRUMENT,
        target_exposure=exposure,
        as_of=PERIOD_END,
    )


class ExecutionSimulatorContract:
    """What an execution model promises, whichever fills it models.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyExecutionSimulator(ExecutionSimulatorContract):
            adapter_factory = staticmethod(MyExecutionSimulator)
    """

    adapter_factory: ClassVar[Callable[[], ExecutionSimulator]]

    @pytest.fixture
    def adapter(self) -> ExecutionSimulator:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

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
    def test_acting_yields_a_fill_and_the_state_that_follows_it(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """Both halves of the outcome are returned; neither is left implicit."""
        fill, new_state = adapter.execute(permitted(exposure), BAR, account)
        assert isinstance(fill, Fill)
        assert isinstance(new_state, AccountState)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_fill_is_charged_under_the_scenario_the_adapter_names(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A result can be re-costed because the fill says what it was costed under."""
        fill, _ = adapter.execute(permitted(exposure), BAR, account)
        assert fill.cost_scenario_id == adapter.cost_scenario_id
        assert fill.cost >= 0.0
        assert math.isfinite(fill.cost)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_fill_reports_what_was_asked_and_what_was_reached(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """Scoring reads the realised value, so both are kept and neither is lost."""
        target = permitted(exposure)
        fill, _ = adapter.execute(target, BAR, account)
        assert fill.agent_id == target.agent_id
        assert fill.instrument == target.instrument
        assert fill.requested_exposure == target.target_exposure
        assert math.isfinite(fill.filled_exposure)

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_fill_is_priced_and_stamped_inside_the_bar_it_was_formed_on(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A price outside the bar, or an instant outside it, is not a fill of it."""
        fill, _ = adapter.execute(permitted(exposure), BAR, account)
        assert BAR.low <= fill.price <= BAR.high
        assert BAR.period_start <= fill.as_of <= BAR.period_end

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_new_state_records_the_exposure_that_was_reached(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """The account holds what happened, not what the target asked for."""
        target = permitted(exposure)
        fill, new_state = adapter.execute(target, BAR, account)
        assert new_state.exposures[target.instrument] == fill.filled_exposure
        untouched = {
            instrument: held
            for instrument, held in account.exposures.items()
            if instrument != target.instrument
        }
        for instrument, held in untouched.items():
            assert new_state.exposures[instrument] == held

    @pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
    @pytest.mark.parametrize("exposure", PERMITTED_EXPOSURES)
    def test_the_charge_reaches_the_account_exactly_once(
        self, adapter: ExecutionSimulator, exposure: float, account: AccountState
    ) -> None:
        """A cost that is double-counted or dropped breaks every net number."""
        fill, new_state = adapter.execute(permitted(exposure), BAR, account)
        charged = new_state.costs_charged - account.costs_charged
        assert charged == pytest.approx(fill.cost)

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
        first_fill, first_state = adapter.execute(permitted(exposure), BAR, account)
        second_fill, second_state = adapter.execute(permitted(exposure), BAR, account)
        assert first_fill == second_fill
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


class TestNullExecutionSimulator(ExecutionSimulatorContract):
    """The stand-in, held to the same contract as any provider-backed model."""

    adapter_factory = staticmethod(NullExecutionSimulator)
