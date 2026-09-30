"""The one order a decision point is taken in: settle what was earned, then decide.

What is checked here is the step's own contract, on stand-ins that record what
they were asked and when, so the order is asserted rather than inferred from a
number an estimator produced: nothing is settled at the first decision point,
the settlement is stamped at the decision point that made it knowable and pairs
the previous close with this one, whatever the caller does while the bar is
accounted happens between the settlement and the decision, and what the step
remembers moves on only once the decision is taken.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from liq.evolution.ecology import accounts
from liq.evolution.ecology.step import DecisionStep, what_the_reading_earned
from liq.evolution.ecology.types import Bar, BarWindow, UtcTimestamp

INSTRUMENT = "SPY"
START = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def _bar(as_of: UtcTimestamp, *, open_: float, close: float) -> Bar:
    """One completed bar ending at ``as_of``."""
    return Bar(
        instrument=INSTRUMENT,
        period_start=as_of - MINUTE,
        period_end=as_of,
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=1.0,
    )


def _window(as_of: UtcTimestamp, bar: Bar) -> BarWindow:
    """A decision point pricing ``bar`` and offering no feature."""
    return BarWindow(
        as_of=as_of,
        segment_id="whole",
        segment_role="train",
        bars=MappingProxyType({INSTRUMENT: bar}),
        features=MappingProxyType({}),
        feature_schema_version="v-test",
    )


def _point(index: int, close: float) -> tuple[BarWindow, Bar]:
    """The ``index``-th decision point, closing at ``close``."""
    as_of = START + index * MINUTE
    bar = _bar(as_of, open_=close, close=close)
    return _window(as_of, bar), bar


def test_the_reading_earned_is_the_move_between_the_two_closes() -> None:
    """What a reading earned is the instrument's fractional close-to-close move."""
    assert what_the_reading_earned(100.0, 101.0) == pytest.approx(0.01)
    assert what_the_reading_earned(200.0, 150.0) == pytest.approx(-0.25)


def test_the_account_still_offers_the_one_definition() -> None:
    """The account's name for the settlement quantity is this one, not a copy."""
    assert accounts.what_the_reading_earned is what_the_reading_earned


def test_nothing_is_settled_at_the_first_decision_point() -> None:
    """With no reading behind it, the first decision point decides and settles nothing."""
    step = DecisionStep()
    settled: list[tuple[UtcTimestamp, float]] = []
    window, bar = _point(0, 100.0)

    decided = step.take(
        window,
        bar,
        settle=lambda as_of, earned: settled.append((as_of, earned)),
        decide=lambda: "wish-0",
    )

    assert decided == "wish-0"
    assert settled == []
    assert step.previous_close == 100.0
    assert step.previous_as_of == window.as_of


def test_each_later_point_settles_the_last_reading_stamped_when_it_became_known() -> (
    None
):
    """The previous close is paired with this close and stamped at this instant."""
    step = DecisionStep()
    settled: list[tuple[UtcTimestamp, float]] = []
    closes = (100.0, 110.0, 99.0)
    windows = [_point(index, close) for index, close in enumerate(closes)]

    for window, bar in windows:
        step.take(
            window,
            bar,
            settle=lambda as_of, earned: settled.append((as_of, earned)),
            decide=lambda: None,
        )

    assert settled == [
        (windows[1][0].as_of, what_the_reading_earned(100.0, 110.0)),
        (windows[2][0].as_of, what_the_reading_earned(110.0, 99.0)),
    ]


def test_the_callers_bar_is_accounted_between_settling_and_deciding() -> None:
    """Settle, then whatever the caller accounts, then decide — in that order."""
    step = DecisionStep()
    order: list[str] = []
    seen_in_meanwhile: list[tuple[float | None, UtcTimestamp | None]] = []
    first, first_bar = _point(0, 100.0)
    second, second_bar = _point(1, 105.0)
    step.take(first, first_bar, settle=lambda *_: None, decide=lambda: None)

    def meanwhile() -> None:
        order.append("meanwhile")
        seen_in_meanwhile.append((step.previous_close, step.previous_as_of))

    step.take(
        second,
        second_bar,
        settle=lambda *_: order.append("settle"),
        meanwhile=meanwhile,
        decide=lambda: order.append("decide"),
    )

    assert order == ["settle", "meanwhile", "decide"]
    # What the step remembers still describes the previous decision point while
    # the caller accounts this one: the gap is carried from the last close.
    assert seen_in_meanwhile == [(100.0, first.as_of)]
    assert (step.previous_close, step.previous_as_of) == (105.0, second.as_of)


def test_a_decision_that_raises_leaves_the_previous_reading_in_place() -> None:
    """The step moves on only once the decision was taken."""
    step = DecisionStep()
    first, first_bar = _point(0, 100.0)
    second, second_bar = _point(1, 105.0)
    step.take(first, first_bar, settle=lambda *_: None, decide=lambda: None)

    def refuse() -> None:
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match="refused"):
        step.take(second, second_bar, settle=lambda *_: None, decide=refuse)

    assert (step.previous_close, step.previous_as_of) == (100.0, first.as_of)
