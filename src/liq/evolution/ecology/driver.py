"""Walking history forward, once, and leaving a record that it was walked so.

The whole of this module is one loop, and the reason it is a module rather than
three lines inside whatever needed it is that the loop is where a program that
learns either stays honest about its sample or stops being able to say what its
sample was. Everything else here exists to make a departure from the loop
impossible to take quietly.

**Forward.** A decision point at or before the one already passed is refused,
not skipped and not sorted. A source is allowed to be wrong; what is not
allowed is for the run to absorb it, because a walk that quietly reorders its
input produces a result whose sample nobody can state afterwards.

**Once.** There is no epoch, no replay and no seek, in the configuration or
here. A second offering of history that arrives all the same — from a source
that rewinds itself, or from two sources composed by accident — is refused for
the same reason and at the same place.

**In order, and never back.** A walk-forward segment is entered, walked and
left. Returning to one is how a validation period becomes a training period
without anybody deciding that it should, so a segment already left behind is
refused on re-entry even when the instants themselves still advance.

**Across the boundary, not restarted by it.** A split boundary is observed here
as a change of segment identity on the windows a source yields — the source
derives those segments from the platform's walk-forward machinery, so nothing
about how history is divided is decided or recomputed in this module. The
crossing is recorded and the walk continues through it: an agent alive on
either side is the same agent, which is what makes a boundary a test of
generalisation rather than a restart wearing its name.

What a visit *is* is the other half of the record. A window that was built and
then dropped is not a visit; a visit is a decision point the population was
shown. That is why the report is written beside the call that shows it, one
entry per decision point handed over, so the run's own account of what it saw
and what was actually handed over cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from liq.evolution.ecology.config import EcologyConfig
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.ports import BarSource
from liq.evolution.ecology.types import (
    BarWindow,
    InstrumentId,
    SegmentRole,
    UtcTimestamp,
    _UtcTimestampMixin,
)

__all__ = [
    "HistoryNotWalkedForward",
    "SegmentCrossing",
    "Visit",
    "WalkReport",
    "walk",
]


class HistoryNotWalkedForward(EcologyError):
    """History was offered out of order, twice, or back to a segment already left.

    It lives here rather than in :mod:`liq.evolution.ecology.errors` because a
    leaf type arrives with the code that raises it, and the loop below is the
    only place in the ecology that can tell a walk forward from a walk that
    doubles back: nothing downstream of it still knows what came before what.

    It is raised rather than recorded. An exclusion says a bar was not used; this
    says the run's sample is not the one it would report, and a run whose sample
    is unknown has nothing worth carrying on to compute.
    """


@dataclass(frozen=True)
class Visit(_UtcTimestampMixin):
    """One decision point, as the population was shown it.

    Attributes:
        as_of: The decision instant the population was shown (UTC).
        segment_id: Identity of the walk-forward segment it fell in.
        segment_role: What that segment is used for.
        instruments: The instruments that carried a complete bar there, in the
            order the decision point offered them. Named rather than counted so
            a run can say which instrument a visit covered, not merely how many.
    """

    as_of: UtcTimestamp
    segment_id: str
    segment_role: SegmentRole
    instruments: tuple[InstrumentId, ...]

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class SegmentCrossing(_UtcTimestampMixin):
    """One walk-forward split boundary, as the run passed through it.

    Attributes:
        as_of: The first decision instant on the far side of the boundary (UTC).
        left: Identity of the segment the run was in.
        entered: Identity of the segment it went on into.
    """

    as_of: UtcTimestamp
    left: str
    entered: str

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class WalkReport:
    """What one pass over history covered, and where it crossed a boundary.

    Attributes:
        run_id: Identity the pass was driven under.
        visits: Every decision point the population was shown, in order.
        crossings: Every split boundary the run passed through, in order. Empty
            where the history walked fell in one segment.
        window_carries_across_segments: The continuity the run was configured
            under, carried so provenance records it rather than a reader having
            to infer it.
    """

    run_id: str
    visits: tuple[Visit, ...]
    crossings: tuple[SegmentCrossing, ...]
    window_carries_across_segments: bool

    @property
    def segments(self) -> tuple[str, ...]:
        """The walk-forward segments the run entered, in the order it entered them."""
        entered: list[str] = []
        for visit in self.visits:
            if not entered or entered[-1] != visit.segment_id:
                entered.append(visit.segment_id)
        return tuple(entered)


def _shown_to_no_one(window: BarWindow) -> None:
    """Show a decision point to nobody, for a walk driven only to be measured."""


def walk(
    source: BarSource,
    config: EcologyConfig,
    *,
    visit: Callable[[BarWindow], None] = _shown_to_no_one,
    log: RunLog = SILENT,
) -> WalkReport:
    """Walk ``source`` forward once, showing each decision point to ``visit``.

    Args:
        source: Where decision points come from. Only the port is used, so the
            same walk runs over a substrate, a guarded reader of real history or
            a declared stand-in without a line here changing.
        config: How the pass is driven, and the record of what it was allowed to
            be.
        visit: What the decision point is handed to. Called exactly once per
            decision point, in order, and after the window has been accepted —
            so nothing is shown to the population that the walk would refuse.
        log: Where this boundary writes what it did: a line per decision point
            handed over, and a line where the walk crossed a split boundary.
            Written here rather than at whatever produced the windows, because
            this is the one place every source is walked through — a run over a
            guarded reader of real history leaves the same record as a run over
            a substrate. Both lines follow the event they report: the decision
            point is written down once it has been shown, so a visit that
            raised is not recorded as one that happened. Silent until a run
            wires a writer.

    Returns:
        WalkReport: What the pass covered and where it crossed a boundary.

    Raises:
        HistoryNotWalkedForward: If a decision point arrives at or before the
            one already passed, or in a segment the run has already left.
    """
    visits: list[Visit] = []
    crossings: list[SegmentCrossing] = []
    behind: set[str] = set()
    previous: BarWindow | None = None

    for window in source.windows():
        if previous is not None:
            _refuse_a_step_backwards(previous, window)
            if window.segment_id != previous.segment_id:
                _refuse_a_segment_returned_to(behind, window)
                behind.add(previous.segment_id)
                crossings.append(
                    SegmentCrossing(
                        as_of=window.as_of,
                        left=previous.segment_id,
                        entered=window.segment_id,
                    )
                )
                log.record("segment_crossed", bar_ts=window.as_of, stage="feature")
        visits.append(
            Visit(
                as_of=window.as_of,
                segment_id=window.segment_id,
                segment_role=window.segment_role,
                instruments=tuple(window.bars),
            )
        )
        visit(window)
        log.record("decision_point_shown", bar_ts=window.as_of, stage="feature")
        previous = window

    return WalkReport(
        run_id=config.run_id,
        visits=tuple(visits),
        crossings=tuple(crossings),
        window_carries_across_segments=config.window_carries_across_segments,
    )


def _refuse_a_step_backwards(previous: BarWindow, window: BarWindow) -> None:
    """Refuse a decision point that does not advance on the one already passed."""
    if window.as_of <= previous.as_of:
        raise HistoryNotWalkedForward(
            f"a decision point at {window.as_of.isoformat()} arrived after one at "
            f"{previous.as_of.isoformat()}; history is walked forward and once, so "
            "an instant already passed is not offered again"
        )


def _refuse_a_segment_returned_to(behind: set[str], window: BarWindow) -> None:
    """Refuse re-entry to a walk-forward segment the run has already left."""
    if window.segment_id in behind:
        raise HistoryNotWalkedForward(
            f"the run returned to walk-forward segment '{window.segment_id}' at "
            f"{window.as_of.isoformat()} after leaving it; a segment is entered, "
            "walked and left, because going back is how a period held out stops "
            "being held out"
        )
