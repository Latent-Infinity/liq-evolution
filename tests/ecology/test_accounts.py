"""How one agent's evaluation account behaves on streams this repository builds.

The claim that this account reproduces a hand-checked fill sequence over a real
tape is evidence and lives in the consumer repository, beside the tape it needs.
What is checked here is the account's own shape: that a target is acted on at
the bar after the one that formed it, that a bar is carried into its open,
charged there and carried on to its close, that what the account holds is what
execution reached rather than what was wanted, and that a decision recorded as
consuming its own bar cannot be written down at all.

**Nothing here is market data and nothing here may be read as market data.** The
price paths below are declared arithmetic, in the same sense as the ramp
:class:`~liq.evolution.ecology.adapters.NullBarSource` offers: they exist so the
shape of the accounting can be checked in a repository that cannot reach the
approved fixtures and must not learn how to. Every number computed from them is
a number about the declared path and says nothing about any instrument.

The charged rate below is likewise declared for these checks. It is not a cost
book figure, it is not anybody's cost, and it exists only so that the leg where
a charge lands on the equity as it stands at the open can be exercised at all —
a stand-in charging nothing would leave that leg unreachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from liq.evolution.ecology.accounts import (
    FITNESS,
    LEARNING,
    AccountEvent,
    DecisionPointNotPriced,
    evaluate,
)
from liq.evolution.ecology.adapters import (
    LiqSimExecutionAdapter,
    NullExecutionSimulator,
    NullRiskSizer,
)
from liq.evolution.ecology.agent import (
    ENTRY_THRESHOLD,
    MASK_PREFIX,
    WEIGHT_PREFIX,
    Agent,
)
from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.types import (
    AccountState,
    Bar,
    BarWindow,
    Fill,
    Genome,
    Intent,
    NotFilled,
    PositionTarget,
)

#: Identity every account below is run under.
RUN_ID = "accounts-shape"
AGENT_ID = "accounts-shape-agent"

#: The single instrument the declared paths price.
INSTRUMENT = "DECLARED-A"

#: An instrument no declared path prices.
ABSENT = "DECLARED-B"

#: Where the declared paths start, and how long one of their bars lasts.
ORIGIN = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
BAR_DURATION = timedelta(minutes=1)

#: The one feature name the declared views carry, and the vocabulary it belongs
#: to. One is enough: what an agent reads is the agent module's subject.
LEVEL = "level"
SCHEMA_VERSION = "accounts-shape-features-1"
GENE_SCHEMA_VERSION = "accounts-shape-genes-1"

#: A declared rate for the stand-in that charges. See the module docstring: not
#: a cost, and not from any book.
DECLARED_ROUND_TRIP_BPS = 8.0
BASIS_POINTS_PER_UNIT = 10_000.0
SIDES_PER_ROUND_TRIP = 2.0

#: A declared scenario for the one check that drives the real execution model,
#: which resolves what it charges by name against the catalogue it is given.
DECLARED_SCENARIO = "accounts-shape-scenario"
DECLARED_CATALOGUE = {DECLARED_SCENARIO: {"round_trip_bps": DECLARED_ROUND_TRIP_BPS}}

#: What an account opens with in these checks. One, so that an exposure and the
#: equity it is a fraction of are the same number and the arithmetic below reads
#: as the docstring it is checking.
OPENING_EQUITY = 1.0

#: A declared path whose opens differ from the closes before them, so that the
#: leg carrying a position across the gap and the leg carrying it through the
#: bar can be told apart.
GAPPED_PATH = ((100.0, 101.0), (102.0, 104.0), (103.0, 99.0))

#: Half the account: an exposure that is neither flat nor the whole of it, so
#: the two legs of a bar compound rather than cancelling into one step.
HALF = 0.5

#: An exposure small enough for the real execution model's own position bound.
WITHIN_THE_VENUE_BOUND = 0.2


@dataclass(frozen=True)
class _DeclaredPath:
    """Decision points over a declared price path, offered once and in order.

    Each bar's period is one duration long and follows the one before it, so the
    instant a decision point is stamped at is the instant the next bar's period
    begins — which is what makes "the bar after the one that formed the target"
    a thing a check can name.
    """

    prices: tuple[tuple[float, float], ...]
    instrument: str = INSTRUMENT

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point in chronological order, exactly once."""
        for index, (opening, close) in enumerate(self.prices):
            period_start = ORIGIN + index * BAR_DURATION
            bar = Bar(
                instrument=self.instrument,
                period_start=period_start,
                period_end=period_start + BAR_DURATION,
                open=opening,
                high=max(opening, close),
                low=min(opening, close),
                close=close,
                volume=1.0,
            )
            yield BarWindow(
                as_of=bar.period_end,
                segment_id="declared-segment",
                segment_role="train",
                bars=MappingProxyType({self.instrument: bar}),
                features=MappingProxyType(
                    {self.instrument: MappingProxyType({LEVEL: close})}
                ),
                feature_schema_version=SCHEMA_VERSION,
            )


@dataclass
class _WantsInTurn:
    """A declared stand-in wanting a scripted exposure at each decision point.

    A stand-in rather than the shipped agent, because what is under test here is
    the account and not the rule: a scripted wish makes the exposure at each bar
    a thing the check states rather than a thing it has to derive from a genome.
    """

    wanted: tuple[float, ...]
    agent_id: str = AGENT_ID
    asked: int = 0

    def intend(self, window: BarWindow, instrument: str) -> Intent:
        """Want whatever this decision point was scripted to want."""
        exposure = self.wanted[self.asked]
        self.asked += 1
        return Intent(
            agent_id=self.agent_id,
            instrument=instrument,
            target_exposure=exposure,
            as_of=window.as_of,
        )


@dataclass
class _RecordingModel:
    """The stand-in model, with the pairs it was asked about kept.

    Which bar a target was acted on against cannot be read off the outcome — a
    fill is stamped at the acting bar and carries no trace of the bar the target
    was formed on — so the pairing is captured where it happens.
    """

    model: NullExecutionSimulator = field(default_factory=NullExecutionSimulator)
    pairs: list[tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def cost_scenario_id(self) -> str:
        """The scenario the model it stands in front of draws its charges from."""
        return self.model.cost_scenario_id

    def execute(
        self, target: PositionTarget, bar: Bar, account: AccountState
    ) -> tuple[Fill | NotFilled, AccountState]:
        """Act as the model does, keeping which target met which bar."""
        self.pairs.append((target.as_of, bar.period_start))
        return self.model.execute(target, bar, account)


@dataclass(frozen=True)
class _OffersNothing:
    """A source that offers no decision point at all."""

    def windows(self) -> Iterator[BarWindow]:
        """Yield nothing, once."""
        return iter(())


def _config() -> EcologyConfig:
    """The configuration every account below is driven under."""
    return EcologyConfig(run_id=RUN_ID)


def _charged(moved: float, equity: float) -> float:
    """What the charging stand-in takes for moving ``moved`` of ``equity``."""
    return (
        moved
        * equity
        * DECLARED_ROUND_TRIP_BPS
        / SIDES_PER_ROUND_TRIP
        / BASIS_POINTS_PER_UNIT
    )


def _run(
    wanted: tuple[float, ...],
    *,
    prices: tuple[tuple[float, float], ...] = GAPPED_PATH,
    simulator: object | None = None,
    instrument: str = INSTRUMENT,
):
    """Run one scripted agent over one declared path."""
    return evaluate(
        _DeclaredPath(prices=prices),
        _config(),
        agent=_WantsInTurn(wanted=wanted),
        instrument=instrument,
        sizer=NullRiskSizer(),
        simulator=(
            NullExecutionSimulator(effective_round_trip_bps=DECLARED_ROUND_TRIP_BPS)
            if simulator is None
            else simulator
        ),
        opening_equity=OPENING_EQUITY,
    )


def test_one_account_state_is_published_for_each_decision_point() -> None:
    """A state per decision point, in the order history was walked.

    A run that published fewer would let a bar it skipped reconcile by having
    nothing to reconcile, and one that published them out of turn would make
    every path-dependent figure below unreadable.
    """
    run = _run((0.0, 0.0, 0.0))
    offered = tuple(window.as_of for window in _DeclaredPath(GAPPED_PATH).windows())

    assert tuple(state.as_of for state in run.states) == offered
    assert run.history.run_id == RUN_ID
    assert len(run.history.visits) == len(run.states)
    assert all(state.agent_id == AGENT_ID for state in run.states)


def test_a_target_is_acted_on_at_the_bar_after_the_one_that_formed_it() -> None:
    """The acting bar begins where the forming decision point ended.

    A target acted on inside the bar it was formed on would trade at prices that
    printed before the decision, which is the cheapest look-ahead there is.
    """
    recorder = _RecordingModel()
    run = _run((HALF, HALF, HALF), simulator=recorder)

    assert recorder.pairs, "no target was ever acted on, so the ordering is untested"
    for formed_at, acting_bar_began in recorder.pairs:
        assert formed_at == acting_bar_began
    # The first decision point has no target behind it, so one fewer bar acts
    # than there are decision points.
    assert len(recorder.pairs) == len(run.states) - 1


def test_the_real_execution_model_never_refuses_the_bar_it_is_handed() -> None:
    """The model that refuses a target acted on too early accepts every one here.

    The refusal is the measured constraint this loop exists to respect: the
    execution model raises rather than absorbing a target that has not aged its
    minimum delay, so a loop that handed over the forming bar would fail here
    rather than quietly filling at prices that printed first.
    """
    run = _run(
        (WITHIN_THE_VENUE_BOUND,) * len(GAPPED_PATH),
        simulator=LiqSimExecutionAdapter(
            cost_scenario_id=DECLARED_SCENARIO, scenarios=DECLARED_CATALOGUE
        ),
    )

    assert any(isinstance(outcome, Fill) for outcome in run.fills)
    assert run.cost_scenario_id == DECLARED_SCENARIO
    assert all(outcome.cost_scenario_id == DECLARED_SCENARIO for outcome in run.fills)


def test_a_bar_that_traded_is_carried_into_its_open_charged_there_and_on_to_its_close() -> (
    None
):
    """The three steps of a bar, in the order they happen.

    Worked out here from the declared path and the declared rate, independently
    of the account that produced it. A position established at this bar's open
    earns the move from that open, and not the move across the gap it was not
    held over; the charge lands on the equity as it stands at the open, before
    anything the bar goes on to do.
    """
    run = _run((HALF, HALF, 0.0))

    # Nothing is held over the first bar, so nothing moves and nothing is
    # charged: the account is worth what it opened with.
    assert run.states[0].equity == pytest.approx(OPENING_EQUITY)
    assert run.states[0].exposures[INSTRUMENT] == pytest.approx(0.0)

    # The second bar establishes half the account at its open. Flat across the
    # gap, so the gap earns nothing; charged on the equity at the open; then
    # held from that open to that close.
    charge = _charged(HALF, OPENING_EQUITY)
    after_charge = OPENING_EQUITY - charge
    second = after_charge * (1.0 + HALF * (104.0 / 102.0 - 1.0))
    assert run.states[1].equity == pytest.approx(second)
    assert run.states[1].exposures[INSTRUMENT] == pytest.approx(HALF)
    assert run.states[1].costs_charged == pytest.approx(charge)

    # The third bar trades nothing — the target is already held — so it is
    # marked through its own open on the exposure it carried in, and charged
    # nothing.
    third = second * (1.0 + HALF * (103.0 / 104.0 - 1.0))
    third *= 1.0 + HALF * (99.0 / 103.0 - 1.0)
    assert run.states[2].equity == pytest.approx(third)
    assert run.states[2].costs_charged == pytest.approx(charge)
    assert run.closing_equity == pytest.approx(third)
    assert run.fitness == pytest.approx(third / OPENING_EQUITY - 1.0)


def test_an_unreached_target_leaves_the_account_holding_what_it_held() -> None:
    """Scoring follows the account, and the account followed the fill.

    The stand-in is asked for more than it says it can reach, so the target is
    permitted and not reached. Nothing is charged for a trade nobody made, the
    exposure does not move, and the outcome names why rather than reporting the
    request back as though it had happened.
    """
    beyond_reach = 0.75
    run = _run(
        (beyond_reach, beyond_reach, beyond_reach),
        simulator=NullExecutionSimulator(reachable_exposure=HALF),
    )

    assert all(
        state.exposures[INSTRUMENT] == pytest.approx(0.0) for state in run.states
    )
    assert all(state.costs_charged == pytest.approx(0.0) for state in run.states)
    assert run.fitness == pytest.approx(0.0)
    unreached = [outcome for outcome in run.fills if isinstance(outcome, NotFilled)]
    assert unreached and all(outcome.reason for outcome in unreached)


def test_a_wish_the_mandate_refuses_leaves_the_account_holding_what_it_held() -> None:
    """A refusal is recorded and is not an instruction to go flat.

    The account is taken long, and then a short is wished for at every remaining
    decision point. The mandate refuses a short in kind, so no target is carried
    into the next bar at all — and what was already held stays held rather than
    being closed by a refusal that said nothing about closing anything.
    """
    run = _run((HALF, -HALF, -HALF))

    assert run.states[0].exposures[INSTRUMENT] == pytest.approx(0.0)
    assert run.states[1].exposures[INSTRUMENT] == pytest.approx(HALF)
    assert run.states[2].exposures[INSTRUMENT] == pytest.approx(HALF)
    assert [rejection.as_of for rejection in run.rejections] == [
        state.as_of for state in run.states[1:]
    ]
    assert all(rejection.reason for rejection in run.rejections)


def test_every_bar_records_what_its_two_updates_consumed() -> None:
    """Learning and fitness are recorded apart, each naming a prior bar.

    Two update paths, so a run can get one right while the other reaches
    forward; the first decision point of a run has no prior outcome to be shown
    or scored, and says so rather than naming one.
    """
    run = _run((HALF, HALF, HALF))
    instants = [state.as_of for state in run.states]

    assert [event.kind for event in run.events] == [LEARNING, FITNESS] * len(instants)
    for index, as_of in enumerate(instants):
        emitted = [event for event in run.events if event.as_of == as_of]
        assert {event.kind for event in emitted} == {LEARNING, FITNESS}
        expected = None if index == 0 else instants[index - 1]
        assert all(event.consumed_as_of == expected for event in emitted)
        assert all(event.agent_id == AGENT_ID for event in emitted)


def test_an_update_that_consumed_its_own_bar_cannot_be_written_down() -> None:
    """The defect this record exists for is refused rather than recorded.

    Both shapes: the same-bar update, which is the dependency cycle itself, and
    the later-bar update, which is the coarser look-ahead. An implementation
    with either would not crash — it would settle, and read as healthy.
    """
    bar = ORIGIN + BAR_DURATION
    for consumed in (bar, bar + BAR_DURATION):
        with pytest.raises(ValueError, match="existed before the bar"):
            AccountEvent(
                agent_id=AGENT_ID, as_of=bar, kind=FITNESS, consumed_as_of=consumed
            )


def test_an_event_that_names_no_zone_is_refused() -> None:
    """A naive instant is not a valid stamp on either of an event's two clocks."""
    naive = datetime(2024, 1, 2, 14, 31)  # noqa: DTZ001 - the case under test
    with pytest.raises(ValueError, match="must be timezone-aware"):
        AccountEvent(agent_id=AGENT_ID, as_of=naive, kind=LEARNING, consumed_as_of=None)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        AccountEvent(
            agent_id=AGENT_ID,
            as_of=ORIGIN + BAR_DURATION,
            kind=LEARNING,
            consumed_as_of=naive,
        )


def test_the_shipped_agent_runs_end_to_end_against_both_ports() -> None:
    """The real rule, the real mandate's shape and a model, joined once.

    The scripted stand-in above says what the account does with a wish; this
    says the account is reachable from the agent that actually forms one, so
    nothing in between is wired only to a stand-in.
    """
    genome = Genome(
        genes={
            f"{MASK_PREFIX}{LEVEL}": 1.0,
            f"{WEIGHT_PREFIX}{LEVEL}": 1.0,
            ENTRY_THRESHOLD: 0.0,
        },
        schema_version=GENE_SCHEMA_VERSION,
    )
    run = evaluate(
        _DeclaredPath(prices=GAPPED_PATH),
        _config(),
        agent=Agent(agent_id=AGENT_ID, genome=genome),
        instrument=INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(
            effective_round_trip_bps=DECLARED_ROUND_TRIP_BPS
        ),
        opening_equity=OPENING_EQUITY,
    )

    assert run.agent_id == AGENT_ID
    assert run.instrument == INSTRUMENT
    assert len(run.states) == len(GAPPED_PATH)
    assert any(isinstance(outcome, Fill) for outcome in run.fills)
    assert run.states[-1].costs_charged > 0.0


def test_a_pass_over_nothing_scores_nothing() -> None:
    """An account nothing was ever offered is worth what it opened with."""
    run = evaluate(
        _OffersNothing(),
        _config(),
        agent=_WantsInTurn(wanted=()),
        instrument=INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator(),
        opening_equity=OPENING_EQUITY,
    )

    assert run.states == ()
    assert run.closing_equity == pytest.approx(OPENING_EQUITY)
    assert run.fitness == pytest.approx(0.0)


def test_a_decision_point_that_prices_something_else_is_refused() -> None:
    """An account pointed at an instrument the stream does not carry stops.

    Marking the position at the last price it had would forward-fill a mark and
    holding the account still would report an unchanged equity as though the
    market had not moved, so the wiring defect is raised instead.
    """
    with pytest.raises(DecisionPointNotPriced, match=ABSENT):
        _run((HALF, HALF, HALF), instrument=ABSENT)


def test_an_account_that_opens_with_nothing_is_refused() -> None:
    """A score is what an account earned on what it opened with."""
    for opening in (0.0, -OPENING_EQUITY):
        with pytest.raises(ValueError, match="must open with something"):
            evaluate(
                _DeclaredPath(prices=GAPPED_PATH),
                _config(),
                agent=_WantsInTurn(wanted=()),
                instrument=INSTRUMENT,
                sizer=NullRiskSizer(),
                simulator=NullExecutionSimulator(),
                opening_equity=opening,
            )
