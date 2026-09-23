"""How the walk behaves on streams this repository can construct.

The evidence that a real tape is covered exactly once lives in the consumer
repository, beside the tape it needs. What is checked here is the loop's own
contract on streams the declared placeholder ramp can be bent into: one that
advances, one that steps back, one that goes round again, and one that returns
to a walk-forward segment it had already left.

The segmented stand-in is what makes the boundary cases reachable at all — the
substrate hands back a single segment, so a crossing cannot happen inside one
pass over it, and a rule about crossings would go unexercised.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.driver import HistoryNotWalkedForward, walk
from liq.evolution.ecology.types import BarWindow

RUN_ID = "ramp-walk"

#: A ramp long enough to enter all three walk-forward segments, so two crossings
#: happen inside one pass.
RAMP_DECISION_POINTS = 9
RAMP_SEGMENT_LENGTH = 3


def _config() -> EcologyConfig:
    """The configuration a ramp walk is driven under."""
    return EcologyConfig(run_id=RUN_ID)


def _segmented() -> NullBarSource:
    """A stand-in whose decision points span three walk-forward segments."""
    return NullBarSource(
        window_count=RAMP_DECISION_POINTS,
        segment_length=RAMP_SEGMENT_LENGTH,
    )


def _windows() -> tuple[BarWindow, ...]:
    """Every decision point the segmented stand-in offers, held in order."""
    return tuple(_segmented().windows())


@dataclasses.dataclass(frozen=True)
class _Offering:
    """A source that hands over exactly the windows it was given, in order."""

    windows_offered: tuple[BarWindow, ...]

    def windows(self) -> Iterator[BarWindow]:
        """Yield what this source was built from, once."""
        return iter(self.windows_offered)


def test_every_decision_point_is_shown_once_and_in_order() -> None:
    """The walk covers the stream, hands each point over once, and advances."""
    shown: list[BarWindow] = []

    report = walk(_segmented(), _config(), visit=shown.append)

    offered = _windows()
    assert [window.as_of for window in shown] == [window.as_of for window in offered]
    assert tuple(visit.as_of for visit in report.visits) == tuple(
        window.as_of for window in offered
    )
    assert len({visit.as_of for visit in report.visits}) == len(report.visits)
    assert report.run_id == RUN_ID


def test_a_visit_names_where_and_what_it_covered() -> None:
    """A visit records its segment, that segment's role, and the instruments."""
    report = walk(_segmented(), _config())

    for visit, window in zip(report.visits, _windows(), strict=True):
        assert visit.segment_id == window.segment_id
        assert visit.segment_role == window.segment_role
        assert visit.instruments == tuple(window.bars)


def test_a_walk_driven_to_be_measured_shows_its_points_to_nobody() -> None:
    """The stream is walked whole even when nothing is waiting to be shown it."""
    report = walk(_segmented(), _config())

    assert len(report.visits) == RAMP_DECISION_POINTS


def test_a_split_boundary_is_recorded_and_walked_through() -> None:
    """A crossing is noted and the walk continues; nothing restarts at a fold."""
    report = walk(_segmented(), _config())

    entered = report.segments
    assert len(entered) == len(set(entered)) > 1
    assert len(report.crossings) == len(entered) - 1
    for crossing, (left, went_on_to) in zip(
        report.crossings, zip(entered, entered[1:], strict=False), strict=True
    ):
        assert (crossing.left, crossing.entered) == (left, went_on_to)

    # The walk is unbroken across every crossing: the decision point after a
    # boundary is the next one in the stream, not the first of a fresh pass.
    stamps = [visit.as_of for visit in report.visits]
    assert stamps == sorted(stamps)
    for crossing in report.crossings:
        assert stamps.index(crossing.as_of) > 0

    assert report.window_carries_across_segments is True


def test_a_stream_offering_the_same_history_again_is_refused() -> None:
    """Going round a second time is refused where it happens, not absorbed."""
    offered = _windows()
    shown: list[BarWindow] = []

    with pytest.raises(HistoryNotWalkedForward, match="walked forward and once"):
        walk(
            _Offering(windows_offered=offered + offered),
            _config(),
            visit=shown.append,
        )

    assert [window.as_of for window in shown] == [window.as_of for window in offered]


def test_a_decision_point_that_does_not_advance_is_refused() -> None:
    """An instant already passed is not offered again, in any order."""
    offered = _windows()

    with pytest.raises(HistoryNotWalkedForward, match="walked forward and once"):
        walk(_Offering(windows_offered=offered[::-1]), _config())


def test_returning_to_a_segment_already_left_is_refused() -> None:
    """A held-out period stays held out even when the instants keep advancing."""
    offered = _windows()
    first, last = offered[0], offered[-1]
    returned = tuple(
        dataclasses.replace(
            window,
            segment_id=first.segment_id,
            segment_role=first.segment_role,
        )
        if window.segment_id == last.segment_id
        else window
        for window in offered
    )
    visited = [window.segment_id for window in returned]
    assert visited[0] == visited[-1] != visited[len(visited) // 2], (
        "the stream must leave a segment and come back to it, or this is a different case"
    )
    stamps = [window.as_of for window in returned]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps), (
        "the instants must still advance, or the step-backwards rule catches this first"
    )

    with pytest.raises(
        HistoryNotWalkedForward, match="returned to walk-forward segment"
    ):
        walk(_Offering(windows_offered=returned), _config())


def test_a_stream_with_nothing_in_it_is_a_walk_that_covered_nothing() -> None:
    """An empty stream is not an error; it is a run with nothing to report."""
    report = walk(_Offering(windows_offered=()), _config())

    assert report.visits == ()
    assert report.crossings == ()
    assert report.segments == ()
