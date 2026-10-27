"""A bar is refused unless it could have been a completed trade bar.

Tier 3. Every case is an invalid mutation of a bar from the declared ramp, so
each claim is about structure and none about any instrument's prices. The
checks run in a fixed order: a non-finite figure is refused before any
comparison, because every ordering comparison with NaN is false and a NaN bar
would otherwise satisfy every order relation.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from liq.evolution.ecology.types import Bar
from tests.support.ramp_bars import ramp_bars

PRICE_FIELDS = ("open", "high", "low", "close")


def _bar() -> Bar:
    return ramp_bars(1)[0]


def _mutate(**changes: float) -> Bar:
    return dataclasses.replace(_bar(), **changes)


def test_the_declared_ramp_bars_construct() -> None:
    """The refusals do not reject the stand-in tape."""
    assert len(ramp_bars()) > 0


@pytest.mark.parametrize("field", (*PRICE_FIELDS, "volume"))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_figure_that_is_not_finite_is_refused(field: str, value: float) -> None:
    """NaN and the infinities are not prices or volumes."""
    with pytest.raises(ValueError, match=field):
        _mutate(**{field: value})


def test_a_nan_bar_is_refused_for_being_not_finite_not_for_its_order() -> None:
    """Finiteness is checked first, so the refusal names it."""
    with pytest.raises(ValueError, match="finite"):
        _mutate(open=math.nan, high=math.nan, low=math.nan, close=math.nan)


@pytest.mark.parametrize("field", PRICE_FIELDS)
@pytest.mark.parametrize("sign", [0.0, -1.0])
def test_a_price_at_or_below_zero_is_refused(field: str, sign: float) -> None:
    """A price is strictly positive."""
    with pytest.raises(ValueError, match=f"{field}.*greater than zero"):
        _mutate(**{field: sign})


@pytest.mark.parametrize("volume", [0.0, -1.0])
def test_a_volume_at_or_below_zero_is_refused(volume: float) -> None:
    """A bar in which nothing traded is not a completed trade bar."""
    with pytest.raises(ValueError, match="volume.*greater than zero"):
        _mutate(volume=volume)


def test_the_high_below_the_low_is_refused() -> None:
    bar = _bar()
    with pytest.raises(ValueError, match="high"):
        _mutate(high=bar.low - 1.0)


def test_the_high_below_the_open_is_refused() -> None:
    bar = _bar()
    with pytest.raises(ValueError, match="high.*open"):
        _mutate(open=bar.high + 1.0, low=bar.low)


def test_the_high_below_the_close_is_refused() -> None:
    bar = _bar()
    with pytest.raises(ValueError, match="high.*close"):
        _mutate(close=bar.high + 1.0)


def test_the_low_above_the_open_is_refused() -> None:
    bar = _bar()
    with pytest.raises(ValueError, match="low.*open"):
        _mutate(open=bar.low / 2.0)


def test_the_low_above_the_close_is_refused() -> None:
    bar = _bar()
    with pytest.raises(ValueError, match="low.*close"):
        _mutate(close=bar.low / 2.0)
