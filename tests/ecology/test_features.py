"""What a feature declaration and a feature set refuse, and why each refusal exists."""

from __future__ import annotations

import pytest

from liq.evolution.ecology.features import (
    DEFAULT_MAXIMUM_LOOKBACK,
    FeatureDeclaration,
    FeatureSet,
)
from tests.support import ramp_bars


def _declaration(**overrides: object) -> FeatureDeclaration:
    """A well-formed declaration, with whatever is being probed replaced."""
    values: dict[str, object] = {
        "name": "probe",
        "lookback": 4,
        "availability_lag": 0,
        "compute": ramp_bars.mean_close,
    }
    values.update(overrides)
    return FeatureDeclaration(**values)  # type: ignore[arg-type]


def test_a_well_formed_set_states_what_it_computes() -> None:
    """The set reports its names in declaration order, and keeps its maximum."""
    features = ramp_bars.feature_set()
    assert features.names == (ramp_bars.MEAN_CLOSE, ramp_bars.MEAN_CLOSE_DELAYED)
    assert features.maximum_lookback == ramp_bars.RAMP_MAXIMUM_LOOKBACK
    assert FeatureSet(declarations=(_declaration(),)).maximum_lookback == (
        DEFAULT_MAXIMUM_LOOKBACK
    )


def test_an_unnamed_feature_is_refused() -> None:
    """A value nobody can ask for is not a feature."""
    with pytest.raises(ValueError, match="must name the feature"):
        _declaration(name="")


def test_a_feature_reading_no_bars_is_refused() -> None:
    """Every feature reads at least the bar that has just closed."""
    with pytest.raises(ValueError, match="lookback of 0"):
        _declaration(lookback=0)


def test_a_feature_available_before_its_bar_is_refused() -> None:
    """A negative lag would make a value readable before it was computed."""
    with pytest.raises(ValueError, match="availability lag of -1"):
        _declaration(availability_lag=-1)


def test_a_set_that_computes_nothing_is_refused() -> None:
    """An empty broadcast is a configuration mistake, not a quiet run."""
    with pytest.raises(ValueError, match="at least one feature"):
        FeatureSet(declarations=())


def test_a_maximum_that_holds_nothing_is_refused() -> None:
    """A window has to hold the bar that has just closed."""
    with pytest.raises(ValueError, match="maximum lookback is 0"):
        FeatureSet(declarations=(_declaration(),), maximum_lookback=0)


def test_one_name_cannot_carry_two_values() -> None:
    """A duplicate name would make the broadcast ambiguous."""
    with pytest.raises(ValueError, match="declared twice"):
        FeatureSet(declarations=(_declaration(), _declaration(lookback=2)))


def test_a_lookback_deeper_than_the_maximum_is_refused_at_declaration_time() -> None:
    """The window is sized by the maximum, so a deeper feature is unaffordable."""
    with pytest.raises(ValueError) as refusal:
        FeatureSet(declarations=(_declaration(lookback=11),), maximum_lookback=10)
    assert "'probe'" in str(refusal.value)
    assert "11" in str(refusal.value)
    assert "10" in str(refusal.value)
