"""Every living agent keeps its own evaluation account, opened at birth and closed at death.

One agent's account over a whole pass is checked in ``test_accounts.py``. What is
checked here is what a population that turns over adds to it: an account opened
when an agent is admitted part-way through a walk, at the same normalised
notional every founder opened at; an account whose agent has died, closed at the
next bar through the execution model and charged there once; and each agent's
per-bar account outcomes going into its own outcome history and nobody else's.

**Nothing here is market data.** The walks are over the declared arithmetic
stream S1 in ``tests/support/turnover_stream.py``. The charging stand-in is
handed the rate ``test_accounts.py`` declares for its own checks; no rate is
written here, and "charged once" is read off the execution model's own outcome
records rather than re-derived.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent
from liq.evolution.ecology.adapters import NullExecutionSimulator, NullRiskSizer
from liq.evolution.ecology.config import EcologyConfig, LearningConfig
from liq.evolution.ecology.types import BarWindow, Fill, NotFilled
from tests.ecology.test_accounts import DECLARED_ROUND_TRIP_BPS
from tests.support import turnover_stream as streams

#: How many decision points each walk here covers: enough for a death and the
#: bar after it, and a birth that then trades.
DECISION_POINTS = 40

#: The decision point the hook below kills one agent and admits another at.
TURNOVER_AT = 20

#: What every account opens with. One, so an exposure and the equity it is a
#: fraction of read as the same number.
OPENING_EQUITY = 1.0

#: Who is admitted at the turnover point.
NEWBORN = "newborn"


def _charging() -> NullExecutionSimulator:
    return NullExecutionSimulator(effective_round_trip_bps=DECLARED_ROUND_TRIP_BPS)


def _population() -> agent.PopulationState:
    return agent.PopulationState.founded(streams.founders(3), learning=LearningConfig())


def _accounts(
    state: agent.PopulationState, simulator: NullExecutionSimulator | None = None
):
    return ecology.population_accounts.PopulationAccounts(
        state,
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=_charging() if simulator is None else simulator,
        opening_equity=OPENING_EQUITY,
    )


def _one_death_and_one_birth(
    state, accounts
) -> tuple[Callable[[BarWindow], None], dict[str, str]]:
    """A turnover hook: at ``TURNOVER_AT`` the agent holding most dies; one is born."""
    seen = {"index": -1}
    chosen: dict[str, str] = {}

    def hook(window: BarWindow) -> None:
        seen["index"] += 1
        if seen["index"] != TURNOVER_AT:
            return
        holding = max(state.agent_ids(), key=lambda agent_id: state.realised(agent_id))
        assert state.realised(holding) != 0.0, (
            "nobody holds anything at the turnover point, so a forced close would "
            "close nothing and the check would be vacuous"
        )
        chosen["dead"] = holding
        state.cull(holding, died_at=window.as_of)
        accounts.close(holding, as_of=window.as_of)
        state.admit(
            replace(
                streams.founder(7),
                agent_id=NEWBORN,
                parents=(holding,),
                born_at=window.as_of,
            )
        )
        accounts.open(NEWBORN, as_of=window.as_of)

    return hook, chosen


def _run(simulator: NullExecutionSimulator | None = None):
    state = _population()
    accounts = _accounts(state, simulator)
    hook, chosen = _one_death_and_one_birth(state, accounts)
    stream = streams.priced(count=DECISION_POINTS)
    result = accounts.run(
        streams.Watched(stream, state),
        EcologyConfig(run_id="population-accounts"),
        turnover=hook,
    )
    return state, stream, result, chosen["dead"]


def test_an_agent_born_mid_walk_opens_at_the_founders_notional() -> None:
    """The newborn's account opens at the birth instant, at what every founder opened at."""
    _, stream, result, _ = _run()

    newborn = result.accounts[NEWBORN]
    founders = [result.accounts[birth.agent_id] for birth in streams.founders(3)]
    assert newborn.opening_equity == OPENING_EQUITY
    assert all(account.opening_equity == OPENING_EQUITY for account in founders)
    assert result.opened_at[NEWBORN] == stream[TURNOVER_AT].as_of
    assert newborn.states[0].as_of == stream[TURNOVER_AT].as_of
    assert newborn.states[0].equity == OPENING_EQUITY
    assert any(isinstance(outcome, Fill) for outcome in newborn.fills), (
        "the newborn never traded, so its account was opened and never used"
    )


def test_a_death_is_closed_at_the_next_bar_and_charged_once() -> None:
    """The dead agent's exposure is closed at the bar after its death, charged once there."""
    simulator = _charging()
    _, stream, result, dead = _run(simulator)
    account = result.accounts[dead]
    closing_bar = stream[TURNOVER_AT + 1]

    assert account.states[-1].as_of == closing_bar.as_of
    assert result.closed_at[dead] == closing_bar.as_of
    closing = account.fills[-1]
    assert isinstance(closing, Fill)
    assert closing.requested_exposure == 0.0
    assert closing.filled_exposure == 0.0
    assert closing.as_of == closing_bar.bars[streams.INSTRUMENT].period_end
    assert closing.cost > 0.0
    assert closing.cost_scenario_id == simulator.cost_provenance().cost_scenario_id
    assert account.cost_scenario_id == simulator.cost_provenance().cost_scenario_id
    assert (
        account.states[-1].costs_charged
        == account.states[-2].costs_charged + closing.cost
    ), "the close was charged other than once, at what the execution model reported"
    traded = [outcome for outcome in account.fills if isinstance(outcome, Fill)]
    assert account.states[-1].costs_charged == sum(fill.cost for fill in traded)
    assert account.states[-1].exposures[streams.INSTRUMENT] == 0.0


def test_each_agents_account_outcomes_feed_its_own_history_only() -> None:
    """An agent's outcome history is its own account's per-bar outcomes, nobody else's."""
    state, _, result, _ = _run()

    histories = {}
    for agent_id in state.agent_ids():
        states = result.accounts[agent_id].states
        own = tuple(
            later.equity / earlier.equity - 1.0
            for earlier, later in zip(states, states[1:], strict=False)
        )
        assert state.history(agent_id) == own[-state.history_capacity :], agent_id
        histories[agent_id] = own
    assert len(histories[NEWBORN]) == DECISION_POINTS - 1 - TURNOVER_AT
    traded = {history for history in histories.values() if any(history)}
    assert len(traded) >= 2, (
        "fewer than two living agents earned anything, so the check cannot tell "
        "an agent's own outcomes from somebody else's"
    )


def test_a_dead_agents_account_is_final_and_readable() -> None:
    """After its closing bar a dead agent's account takes no further state."""
    _, _, result, dead = _run()
    account = result.accounts[dead]

    assert len(account.states) == TURNOVER_AT + 2
    assert account.agent_id == dead
    assert account.closing_equity == account.states[-1].equity
    assert result.history.visits[-1].as_of > account.states[-1].as_of


def test_a_population_settles_as_one_and_never_through_one_member() -> None:
    """A member's account is never asked to settle on its own."""
    state = _population()
    accounts = _accounts(state)
    member = accounts.member(state.agent_ids()[0])
    with pytest.raises(ecology.errors.EcologyError, match="settled"):
        member.observe(streams.priced(count=2)[1].as_of, 0.0)


def test_accounts_refuse_what_could_not_be_one() -> None:
    """Nothing to open with, a birth nobody admitted, and a death of the living are refused."""
    state = _population()
    with pytest.raises(ValueError, match="open with something"):
        ecology.population_accounts.PopulationAccounts(
            state,
            instrument=streams.INSTRUMENT,
            sizer=NullRiskSizer(),
            simulator=_charging(),
            opening_equity=0.0,
        )
    accounts = _accounts(state)
    as_of = streams.priced(count=1)[0].as_of
    with pytest.raises(KeyError, match="not alive"):
        accounts.open("nobody", as_of=as_of)
    with pytest.raises(ValueError, match="already"):
        accounts.open(state.agent_ids()[0], as_of=as_of)
    with pytest.raises(ValueError, match="still alive"):
        accounts.close(state.agent_ids()[0], as_of=as_of)
    with pytest.raises(KeyError, match="no account"):
        accounts.close("nobody", as_of=as_of)


def test_a_decision_point_that_does_not_price_the_instrument_is_refused() -> None:
    """An account population follows one instrument, and a bar without it is a defect."""
    state = _population()
    accounts = _accounts(state)
    window = streams.priced(count=1)[0]
    unpriced = replace(window, bars=MappingProxyType({}))
    with pytest.raises(ecology.accounts.DecisionPointNotPriced):
        accounts.visit(unpriced)


def test_a_flat_agents_death_trades_nothing_and_is_charged_nothing() -> None:
    """A death with nothing open takes the same exit, and the model says nothing traded."""
    state = _population()
    accounts = _accounts(state)
    stream = streams.priced(count=3)
    accounts.visit(stream[0])
    flat = state.agent_ids()[0]
    assert state.realised(flat) == 0.0

    def die(window: BarWindow) -> None:
        state.cull(flat, died_at=window.as_of)
        accounts.close(flat, as_of=window.as_of)
        with pytest.raises(ValueError, match="already closed"):
            accounts.close(flat, as_of=window.as_of)

    accounts.visit(stream[1], turnover=die)
    accounts.visit(stream[2])
    closed = accounts.evaluation(flat)
    assert isinstance(closed.fills[-1], NotFilled)
    assert closed.fills[-1].requested_exposure == 0.0
    assert closed.states[-1].costs_charged == 0.0
    assert closed.states[-1].as_of == stream[2].as_of
