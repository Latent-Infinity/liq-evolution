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

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import timedelta
from types import MappingProxyType

import pytest

from liq.evolution import ecology
from liq.evolution.ecology import agent
from liq.evolution.ecology.adapters import NullExecutionSimulator, NullRiskSizer
from liq.evolution.ecology.config import EcologyConfig, LearningConfig
from liq.evolution.ecology.types import (
    AccountState,
    BarWindow,
    Fill,
    NotFilled,
    PositionTarget,
    Rejection,
    SizingOutcome,
)
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


# The allocation hook. With an allocation the walk also keeps the aggregate
# book's account. Its weights are updated once per decision point, after
# settlement and the trigger's turnover and before the population decides; its
# target is formed after the decision, from what every living agent was
# permitted, and acted on at the next bar like any agent's target, never the
# same one. Without an allocation the walk is the one above, with no book.

#: The decision point the refusing sizer below refuses every wish at: one at
#: which some agent was last permitted a non-zero target, so a refusal that
#: flattened instead of holding would show.
REFUSED_AT = 6


def _book_target(window: BarWindow, index: int) -> PositionTarget | None:
    """The book's declared probe target at decision point ``index``.

    It varies with the index, so a target acted on at the wrong bar shows; and
    every third point sends none, so the book's holding is left alone there.
    """
    if index % 3 == 2:
        return None
    return PositionTarget(
        agent_id=ecology.BOOK_ACCOUNT_ID,
        instrument=streams.INSTRUMENT,
        target_exposure=0.25 * (1 + index % 3),
        as_of=window.as_of,
    )


class _RecordingAllocation:
    """An allocation that records what it is handed and sends declared targets."""

    def __init__(self, state, accounts) -> None:
        self.state = state
        self.accounts = accounts
        self.calls: list[tuple[str, object]] = []
        self.targets: list[PositionTarget | None] = []
        self.permitted: list[dict[str, float]] = []
        self.handed: list[AccountState] = []

    def allocate(self, window: BarWindow, living: tuple[str, ...]) -> None:
        self.calls.append(("allocate", window.as_of))
        assert living == self.state.agent_ids()
        for agent_id in living:
            states = self.accounts.evaluation(agent_id).states
            assert states[-1].as_of == window.as_of, (
                f"{agent_id} had not been accounted for the bar when weights updated"
            )
            if len(states) >= 2:
                earned = states[-1].equity / states[-2].equity - 1.0
                assert self.state.history(agent_id)[-1] == earned, (
                    f"{agent_id}'s outcome was not settled when weights updated"
                )
            try:
                formed = self.accounts.wish(agent_id).as_of
            except KeyError:
                formed = None
            assert formed is None or formed < window.as_of, (
                f"{agent_id} had already decided when weights updated"
            )
        self.living_at = getattr(self, "living_at", [])
        self.living_at.append(living)

    def book(
        self,
        window: BarWindow,
        permitted: Mapping[str, float],
        account: AccountState,
    ) -> PositionTarget | None:
        self.calls.append(("book", window.as_of))
        assert set(permitted) == set(self.state.agent_ids())
        for agent_id in permitted:
            assert self.accounts.wish(agent_id).as_of == window.as_of
        self.permitted.append(dict(permitted))
        self.handed.append(account)
        target = _book_target(window, len(self.targets))
        self.targets.append(target)
        return target


def _allocated_run(sizer=None):
    state = _population()
    accounts = ecology.population_accounts.PopulationAccounts(
        state,
        instrument=streams.INSTRUMENT,
        sizer=NullRiskSizer() if sizer is None else sizer,
        simulator=_charging(),
        opening_equity=OPENING_EQUITY,
    )
    hook, chosen = _one_death_and_one_birth(state, accounts)
    allocation = _RecordingAllocation(state, accounts)
    stream = streams.priced(count=DECISION_POINTS)
    result = accounts.run(
        streams.Watched(stream, state),
        EcologyConfig(run_id="population-accounts"),
        turnover=hook,
        allocation=allocation,
    )
    return state, stream, result, chosen["dead"], allocation


def test_without_an_allocation_the_walk_keeps_no_book() -> None:
    """No allocation, no book: the walk is the one every case above checks."""
    _, _, result, _ = _run()
    assert result.book is None


def test_an_allocation_changes_no_agents_account() -> None:
    """The book is kept beside the evaluation accounts and moves none of them."""
    _, _, plain, _ = _run()
    _, _, allocated, _, _ = _allocated_run()
    assert dict(allocated.accounts) == dict(plain.accounts)
    assert dict(allocated.opened_at) == dict(plain.opened_at)
    assert dict(allocated.closed_at) == dict(plain.closed_at)


def test_weights_update_once_per_bar_after_turnover_with_exactly_the_living() -> None:
    """Once per decision point, after settlement and turnover, before the decision."""
    _, stream, _, dead, allocation = _allocated_run()

    updates = [as_of for stage, as_of in allocation.calls if stage == "allocate"]
    assert updates == [window.as_of for window in stream]
    founders = tuple(birth.agent_id for birth in streams.founders(3))
    assert allocation.living_at[TURNOVER_AT - 1] == founders
    assert NEWBORN in allocation.living_at[TURNOVER_AT]
    assert dead not in allocation.living_at[TURNOVER_AT]


def test_the_book_is_formed_after_the_decision_from_what_the_living_were_permitted() -> (
    None
):
    """One book per decision point, after its update, over exactly the living."""
    _, stream, _, _, allocation = _allocated_run()

    assert allocation.calls == [
        (stage, window.as_of) for window in stream for stage in ("allocate", "book")
    ]
    assert NEWBORN not in allocation.permitted[TURNOVER_AT - 1]
    assert NEWBORN in allocation.permitted[TURNOVER_AT]
    for handed, window in zip(allocation.handed, stream, strict=True):
        assert handed.agent_id == ecology.BOOK_ACCOUNT_ID
        assert handed.as_of == window.as_of


def test_the_book_target_is_acted_on_at_the_next_bar_never_the_same_one() -> None:
    """What the book is sent at one decision point trades at the bar after it."""
    _, stream, result, _, allocation = _allocated_run()
    book = result.book
    assert book is not None

    assert book.agent_id == ecology.BOOK_ACCOUNT_ID
    assert book.opening_equity == OPENING_EQUITY
    assert [state.as_of for state in book.states] == [w.as_of for w in stream]
    sent = [
        (index, target)
        for index, target in enumerate(allocation.targets[:-1])
        if target is not None
    ]
    assert len(book.fills) == len(sent)
    for (index, target), outcome in zip(sent, book.fills, strict=True):
        acting = stream[index + 1].bars[streams.INSTRUMENT]
        assert outcome.requested_exposure == target.target_exposure
        assert outcome.as_of == acting.period_end
        assert outcome.as_of > stream[index].as_of
    assert any(isinstance(outcome, Fill) for outcome in book.fills)
    for index, handed in enumerate(allocation.handed[1:], start=1):
        acted = [target for i, target in sent if i < index]
        if acted:
            assert handed.exposures[streams.INSTRUMENT] == acted[-1].target_exposure


class _RefusingAt(NullRiskSizer):
    """The stand-in mandate, refusing every wish formed at one decision point."""

    def __init__(self, refused_at) -> None:
        super().__init__()
        object.__setattr__(self, "_refused_at", refused_at)

    def size(self, intent, account):
        if intent.as_of == self._refused_at:
            return SizingOutcome(
                target=None,
                rejections=(
                    Rejection(
                        agent_id=intent.agent_id,
                        instrument=intent.instrument,
                        as_of=intent.as_of,
                        reason="refused_for_the_probe",
                    ),
                ),
            )
        return super().size(intent, account)


def test_a_refused_wish_leaves_what_was_permitted_before_it() -> None:
    """A refusal holds, so the agent is handed in at what it was last permitted."""
    stream = streams.priced(count=DECISION_POINTS)
    _, _, _, _, allocation = _allocated_run(_RefusingAt(stream[REFUSED_AT].as_of))

    assert any(
        value != 0.0 for value in allocation.permitted[REFUSED_AT - 1].values()
    ), (
        "premise: every agent was permitted flat before the refusal, so holding "
        "and flattening are the same target and the case shows nothing"
    )
    assert allocation.permitted[REFUSED_AT] == allocation.permitted[REFUSED_AT - 1]
    assert any(
        allocation.permitted[REFUSED_AT - 1] != each for each in allocation.permitted
    ), "the permitted targets never move, so a refusal holding them shows nothing"


def test_an_agent_is_flat_before_anything_was_permitted() -> None:
    """At the first decision point a refused agent was never permitted anything."""
    stream = streams.priced(count=DECISION_POINTS)
    _, _, _, _, allocation = _allocated_run(_RefusingAt(stream[0].as_of))

    assert set(allocation.permitted[0].values()) == {0.0}


@pytest.mark.parametrize(
    "changes",
    [
        {"agent_id": "founder"},
        {"instrument": "SOMEWHERE-ELSE"},
        {"as_of": timedelta(minutes=-1)},
    ],
    ids=["another-account", "another-instrument", "another-instant"],
)
def test_a_book_target_that_is_not_the_books_now_is_refused(
    changes: dict[str, object],
) -> None:
    """The book is sent only its own target, in its instrument, formed now."""
    state = _population()
    accounts = _accounts(state)
    window = streams.priced(count=1)[0]

    class Misdirected:
        def allocate(self, window: BarWindow, living: tuple[str, ...]) -> None:
            del window, living

        def book(self, window, permitted, account):
            del permitted, account
            shift = changes.get("as_of", timedelta(0))
            assert isinstance(shift, timedelta)
            return PositionTarget(
                agent_id=str(changes.get("agent_id", ecology.BOOK_ACCOUNT_ID)),
                instrument=str(changes.get("instrument", streams.INSTRUMENT)),
                target_exposure=0.5,
                as_of=window.as_of + shift,
            )

    with pytest.raises(ValueError, match="book"):
        accounts.visit(window, allocation=Misdirected())


# The fixed-share allocation. The hook the loop allocates through: at the
# update instant it reads what the accounts recorded at this decision point —
# each living agent's own outcome and the instrument's reference return — and
# nothing else; at the book instant it hands the ledger the weights and what
# the living were permitted, and returns the target the ledger sent. Its
# weights are checked against an independent rebuild from the frozen record's
# text, fed from the accounts' published states and the stream's closes, never
# from what the hook itself recorded.

#: A test-local band; the run's value is the composition's to supply.
BAND = 0.0


def _frozen_aggregator():
    from liq.evolution.ecology import fixed_share
    from tests.support import weight_oracle

    record = weight_oracle.read_record()
    return fixed_share.FixedShare(
        contract_sha256=record.sha256,
        eta=record.contract["eta"],  # ty: ignore[invalid-argument-type]
        share_rate=record.contract["share_rate"],  # ty: ignore[invalid-argument-type]
    )


def _fixed_share_run():
    from liq.evolution.ecology.adapters import NullPortfolioRisk

    state = _population()
    accounts = _accounts(state)
    hook, chosen = _one_death_and_one_birth(state, accounts)
    allocation = ecology.population_accounts.FixedShareAllocation(
        accounts, aggregator=_frozen_aggregator(), risk=NullPortfolioRisk(), band=BAND
    )
    living_at: list[tuple[str, ...]] = []

    def turn_over(window: BarWindow) -> None:
        hook(window)
        living_at.append(state.agent_ids())

    stream = streams.priced(count=DECISION_POINTS)
    result = accounts.run(
        streams.Watched(stream, state),
        EcologyConfig(run_id="population-accounts"),
        turnover=turn_over,
        allocation=allocation,
    )
    return stream, result, allocation, living_at, chosen["dead"]


def _outcomes_at(result, living, as_of) -> dict[str, float]:
    """Each living agent's own outcome at ``as_of``, read from its published states."""
    outcomes: dict[str, float] = {}
    for agent_id in living:
        states = result.accounts[agent_id].states
        at = [index for index, each in enumerate(states) if each.as_of == as_of]
        if at and at[0] > 0:
            outcomes[agent_id] = states[at[0]].equity / states[at[0] - 1].equity - 1.0
    return outcomes


def _rebuilt(stream, result, living_at):
    """The weights at every decision point, rebuilt from the record's text alone."""
    from tests.support import weight_oracle as oracle

    contract = oracle.read_record().contract
    eta = float(contract["eta"])  # ty: ignore[invalid-argument-type]
    share = float(contract["share_rate"])  # ty: ignore[invalid-argument-type]
    warmup = int(contract["scale_warmup"])  # ty: ignore[invalid-argument-type]
    founders = living_at[0]
    before = {agent_id: 1.0 / len(founders) for agent_id in founders}
    scale = oracle.Scale()
    rebuilt = []
    for index, window in enumerate(stream):
        close = window.bars[streams.INSTRUMENT].close
        reference = (
            None
            if index == 0
            else close / stream[index - 1].bars[streams.INSTRUMENT].close - 1.0
        )
        weights, scale = oracle.update(
            before,
            living_at[index],
            _outcomes_at(result, living_at[index], window.as_of),
            scale,
            reference,
            eta=eta,
            share_rate=share,
            warmup=warmup,
        )
        rebuilt.append((weights, frozenset(living_at[index]) - set(before), share))
        before = weights
    return rebuilt


def test_fixed_share_weights_are_the_records_from_what_each_point_recorded() -> None:
    """Every vector matches the rebuild; a newborn enters at exactly s/P; the dead leave."""
    from tests.support import weight_oracle as oracle

    stream, result, allocation, living_at, dead = _fixed_share_run()
    held = allocation.record().weights
    assert len(held) == len(stream)
    assert int(oracle.read_record().contract["scale_warmup"]) < len(stream), (  # ty: ignore[invalid-argument-type]
        "the walk ends inside the scale's warm-up, so no loss step is checked"
    )
    found: list[str] = []
    for index, (state, (rebuilt, newborns, share)) in enumerate(
        zip(held, _rebuilt(stream, result, living_at), strict=True)
    ):
        implemented = dict(zip(state.agent_ids, state.weights.tolist(), strict=True))
        found += [
            f"@{index} {each}"
            for each in oracle.mismatches(
                rebuilt, implemented, newborns=newborns, share_rate=share
            )
        ]
    assert not found, "\n".join(found[:10])
    at_turnover = held[TURNOVER_AT]
    assert NEWBORN in at_turnover.agent_ids and dead not in at_turnover.agent_ids


def test_the_update_reads_what_this_decision_point_recorded_and_nothing_later() -> None:
    """What the accounts recorded at the update instant is this point's, and settled."""
    stream, result, allocation, living_at, _ = _fixed_share_run()
    seen = allocation.record().recorded
    assert [each.as_of for each in seen] == [window.as_of for window in stream]
    for index, (recorded, window) in enumerate(zip(seen, stream, strict=True)):
        expected = _outcomes_at(result, living_at[index], window.as_of)
        assert {
            agent_id: value
            for agent_id, value in recorded.outcomes.items()
            if agent_id in living_at[index]
        } == expected, index
        if index == 0:
            assert recorded.reference is None
        else:
            close = window.bars[streams.INSTRUMENT].close
            previous = stream[index - 1].bars[streams.INSTRUMENT].close
            assert recorded.reference == close / previous - 1.0


def test_the_ledger_is_handed_this_points_weights_and_the_book_acts_next_bar() -> None:
    """One ledger row per point; each row's sent target is what the book holds a bar on."""
    stream, result, allocation, _, _ = _fixed_share_run()
    record = allocation.record()
    assert [row.as_of for row in record.rows] == [w.as_of for w in stream]
    assert result.book is not None
    book_states = result.book.states
    assert book_states[0].exposures.get(streams.INSTRUMENT, 0.0) == 0.0
    for index, (row, weights) in enumerate(
        zip(record.rows, record.weights, strict=True)
    ):
        assert row.decision.agent_ids == weights.agent_ids, index
        assert row.decision.target is None or row.decision.target.as_of == row.as_of
        if index + 1 < len(book_states):
            assert book_states[index + 1].exposures.get(
                streams.INSTRUMENT, 0.0
            ) == pytest.approx(row.decision.sent, abs=1e-15), index


def test_the_allocation_is_a_plain_value_with_no_random_state() -> None:
    """Weights and ledger are frozen values a run can snapshot; nothing draws."""
    import dataclasses as dc

    import numpy as np

    _, _, allocation, _, _ = _fixed_share_run()
    for value in (allocation.weights, allocation.ledger):
        assert dc.is_dataclass(value)
        assert type(value).__dataclass_params__.frozen  # ty: ignore[unresolved-attribute]
    for value in (*vars(allocation).values(), *vars(allocation.ledger).values()):
        assert not isinstance(value, np.random.Generator | np.random.RandomState)


def test_a_permitted_set_that_is_not_the_weighted_living_is_refused() -> None:
    """The book is netted over exactly the agents the weights were updated over."""
    from liq.evolution.ecology.adapters import NullPortfolioRisk

    state = _population()
    accounts = _accounts(state)
    allocation = ecology.population_accounts.FixedShareAllocation(
        accounts, aggregator=_frozen_aggregator(), risk=NullPortfolioRisk(), band=BAND
    )
    window = streams.priced(count=1)[0]

    class Stranger:
        def allocate(self, window: BarWindow, living: tuple[str, ...]) -> None:
            allocation.allocate(window, living)

        def book(self, window, permitted, account):
            return allocation.book(window, {**permitted, "stranger": 0.0}, account)

    with pytest.raises(ValueError, match="weighted"):
        accounts.visit(window, allocation=Stranger())


def test_an_update_handed_another_points_record_is_refused() -> None:
    """The update reads only what the accounts recorded at the window it is handed."""
    from liq.evolution.ecology.adapters import NullPortfolioRisk

    state = _population()
    accounts = _accounts(state)
    allocation = ecology.population_accounts.FixedShareAllocation(
        accounts, aggregator=_frozen_aggregator(), risk=NullPortfolioRisk(), band=BAND
    )
    window = streams.priced(count=1)[0]
    with pytest.raises(ValueError, match="recorded"):
        allocation.allocate(window, state.agent_ids())
