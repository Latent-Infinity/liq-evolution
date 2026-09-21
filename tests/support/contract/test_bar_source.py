"""The contract every source of decision points is run against.

Tier 2. These checks read the port's surface and the domain values it carries —
that history is walked forward and once, that a decision point is stamped, is
placed in a walk-forward segment, and reaches nothing that was not available at
its as-of instant. Nothing here reads how a particular adapter computes any of
it, so the suite is unchanged when a real source replaces a stand-in.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import timedelta
from typing import ClassVar, get_args

import pytest

from liq.evolution.ecology import BarSource, BarWindow, SegmentRole
from liq.evolution.ecology.adapters import NullBarSource

SEGMENT_ROLES = get_args(SegmentRole.__value__)


class BarSourceContract:
    """What a source of decision points promises, whoever provides it.

    An adapter joins the contract by subclassing this and naming a factory that
    builds it. No suite is copied to add one::

        class TestMyBarSource(BarSourceContract):
            adapter_factory = staticmethod(MyBarSource)

    The factory takes no arguments and returns a fresh adapter, so one test
    cannot see what another did to it.
    """

    adapter_factory: ClassVar[Callable[[], BarSource]]

    @pytest.fixture
    def adapter(self) -> BarSource:
        """The adapter under test, built fresh for this test."""
        return type(self).adapter_factory()

    @pytest.fixture
    def decision_points(self, adapter: BarSource) -> tuple[BarWindow, ...]:
        """Every window the adapter yields, in the order it yielded them."""
        return tuple(adapter.windows())

    def test_the_adapter_satisfies_the_port(self, adapter: BarSource) -> None:
        """Conformance is of the object handed over, not of a declared class."""
        assert isinstance(adapter, BarSource)

    def test_history_is_walked_once_and_cannot_be_replayed(
        self, adapter: BarSource
    ) -> None:
        """The stream is consumed, not re-read: there is no rewind or seek."""
        stream = adapter.windows()
        assert iter(stream) is stream
        assert tuple(stream) != ()
        assert tuple(stream) == ()

    def test_decision_points_arrive_in_time_order(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """A walk is forward: no window precedes the one yielded before it."""
        stamps = [window.as_of for window in decision_points]
        assert stamps == sorted(stamps)

    def test_no_decision_point_is_yielded_twice(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """Each decision point in a segment is offered exactly once."""
        seen = {(window.as_of, window.segment_id) for window in decision_points}
        assert len(seen) == len(decision_points)

    def test_every_decision_point_is_stamped_in_utc(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """An as-of is an absolute instant, never a naive or local one."""
        for window in decision_points:
            assert window.as_of.tzinfo is not None
            assert window.as_of.utcoffset() == timedelta(0)

    def test_every_decision_point_declares_its_segment(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """A split boundary is observable as a change of segment identity."""
        role_of: dict[str, str] = {}
        for window in decision_points:
            assert window.segment_id
            assert window.segment_role in SEGMENT_ROLES
            held = role_of.setdefault(window.segment_id, window.segment_role)
            assert held == window.segment_role

    def test_a_segment_is_visited_once_and_never_returned_to(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """Segments are walked in order; a run does not go back to one."""
        visited = [window.segment_id for window in decision_points]
        entered = [
            segment_id
            for index, segment_id in enumerate(visited)
            if index == 0 or visited[index - 1] != segment_id
        ]
        assert len(entered) == len(set(entered))

    def test_nothing_in_a_window_reaches_past_its_as_of(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """A window exists only where its bars were complete at its as-of."""
        for window in decision_points:
            for instrument, bar in window.bars.items():
                assert bar.instrument == instrument
                assert bar.period_start < bar.period_end
                assert bar.period_end <= window.as_of

    def test_every_bar_is_internally_consistent(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """A bar that was never complete cannot be made to look like one."""
        for window in decision_points:
            for bar in window.bars.values():
                assert bar.low <= min(bar.open, bar.close)
                assert bar.high >= max(bar.open, bar.close)
                assert bar.low <= bar.high
                assert bar.volume >= 0.0

    def test_the_view_a_window_carries_is_read_only(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """Readers share one view, so one of them cannot change another's."""
        for window in decision_points:
            with pytest.raises(TypeError):
                window.bars["absent-instrument"] = None
            with pytest.raises(TypeError):
                window.features["absent-instrument"] = {}
            for view in window.features.values():
                with pytest.raises(TypeError):
                    view["absent-feature"] = 0.0

    def test_one_feature_vocabulary_governs_the_whole_walk(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """An agent is never silently evaluated under a vocabulary it changed to."""
        versions = {window.feature_schema_version for window in decision_points}
        assert len(versions) == 1
        assert versions != {""}

    def test_feature_values_are_named_finite_numbers(
        self, decision_points: tuple[BarWindow, ...]
    ) -> None:
        """A feature is a number under a name, never a sentinel or a gap."""
        for window in decision_points:
            for view in window.features.values():
                for name, value in view.items():
                    assert name
                    assert isinstance(value, float)
                    assert math.isfinite(value)


class TestNullBarSource(BarSourceContract):
    """The stand-in, held to the same contract as any provider-backed source."""

    adapter_factory = staticmethod(NullBarSource)
