"""How the substrate behaves on rows this repository can construct.

The evidence for what the substrate does at real bar boundaries is in the
consumer repository, beside the tape it needs. What is checked here is the
module's own contract on inputs the declared placeholder ramp can be bent into:
a row with a field missing, a sequence with a bar taken out of it, a value that
its declaration says is not readable yet, and the report a pass hands back.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from liq.evolution.ecology.substrate import (
    BAR_FIELDS,
    FEATURE_WARM_UP,
    INCOMPLETE_BAR,
    SINGLE_SEGMENT,
    SINGLE_SEGMENT_ROLE,
    BarRecord,
    BroadcastFeatureView,
    FeatureSubstrate,
    FutureReadError,
    SegmentBoundary,
    SegmentNotScheduled,
    SegmentSchedule,
)
from tests.support import ramp_bars
from tests.support.contract.test_bar_source import BarSourceContract


class TestFeatureSubstrate(BarSourceContract):
    """The substrate supplies decision points, so it answers to the C-01 contract.

    Registered here rather than beside the contract base: the base is loaded by
    the consumer repository under a private package name of its own, so a module
    it imports cannot reach anything under `tests`. It is held over the declared
    placeholder ramp, because what the contract reads is the shape of what is
    yielded — forward, once, stamped, segmented, reaching nothing past its as-of
    — and none of that depends on what the bars say.
    """

    adapter_factory = staticmethod(ramp_bars.substrate)


def _views(substrate: FeatureSubstrate) -> list[BroadcastFeatureView]:
    """One view per decision point, for a pass over a single instrument."""
    return [next(iter(window.features.values())) for window in substrate.windows()]


def test_a_feature_is_broadcast_once_its_window_is_full() -> None:
    """Warm-up is exactly the declared lookback minus one, then the value appears."""
    substrate = ramp_bars.substrate()
    views = _views(substrate)

    warm_up = ramp_bars.RAMP_LOOKBACK - 1
    assert all(ramp_bars.MEAN_CLOSE not in view for view in views[:warm_up])
    assert all(ramp_bars.MEAN_CLOSE in view for view in views[warm_up:])
    assert len(views[warm_up]) == 2

    excluded = [
        exclusion
        for exclusion in substrate.exclusions
        if exclusion.feature == ramp_bars.MEAN_CLOSE
    ]
    assert len(excluded) == warm_up
    assert {exclusion.reason for exclusion in excluded} == {FEATURE_WARM_UP}
    assert all(str(ramp_bars.RAMP_LOOKBACK) in e.detail for e in excluded)


def test_a_declared_lag_defers_a_value_rather_than_dropping_it() -> None:
    """The value exists at the decision point it was computed at, and is refused there."""
    substrate = ramp_bars.substrate(features=ramp_bars.feature_set(lag=1))
    views = _views(substrate)
    first_full = ramp_bars.RAMP_LOOKBACK - 1

    with pytest.raises(FutureReadError, match=ramp_bars.MEAN_CLOSE_DELAYED):
        views[first_full][ramp_bars.MEAN_CLOSE_DELAYED]
    assert ramp_bars.MEAN_CLOSE_DELAYED not in views[first_full]
    assert ramp_bars.MEAN_CLOSE in views[first_full]

    bars = ramp_bars.ramp_bars()
    deferred = views[first_full + 1][ramp_bars.MEAN_CLOSE_DELAYED]
    assert deferred == ramp_bars.mean_close(bars[: ramp_bars.RAMP_LOOKBACK])
    assert deferred != views[first_full + 1][ramp_bars.MEAN_CLOSE]


def test_an_incomplete_row_is_excluded_and_named() -> None:
    """No decision point, no feature, and a record saying which field was absent."""
    records = list(ramp_bars.as_records(ramp_bars.ramp_bars()))
    broken = ramp_bars.without(records[6], "close")
    records[6] = broken
    substrate = ramp_bars.substrate(records)

    stamps = [window.as_of for window in substrate.windows()]
    assert broken.period_end not in stamps
    assert len(stamps) == len(records) - 1

    incomplete = [e for e in substrate.exclusions if e.reason == INCOMPLETE_BAR]
    assert len(incomplete) == 1
    assert incomplete[0].as_of == broken.period_end
    assert incomplete[0].feature is None
    assert "close" in incomplete[0].detail
    assert broken.absent_fields == ("close",)
    assert broken.completed() is None


def test_a_break_in_the_sequence_begins_the_window_again() -> None:
    """A window covers consecutive bars, so a hole restarts it rather than closing over it."""
    bars = ramp_bars.ramp_bars(count=20)
    records = ramp_bars.as_records(bars[:8] + bars[9:])
    views = _views(ramp_bars.substrate(records))

    warm_up = ramp_bars.RAMP_LOOKBACK - 1
    assert ramp_bars.MEAN_CLOSE in views[warm_up]
    assert all(ramp_bars.MEAN_CLOSE not in view for view in views[8 : 8 + warm_up])
    assert ramp_bars.MEAN_CLOSE in views[8 + warm_up]


def test_two_instruments_share_one_decision_point() -> None:
    """Rows that ended at the same instant are one decision point, not two."""
    left = ramp_bars.as_records(ramp_bars.ramp_bars(instrument="LEFT"))
    right = ramp_bars.as_records(ramp_bars.ramp_bars(instrument="RIGHT"))
    interleaved = [record for pair in zip(left, right, strict=True) for record in pair]
    substrate = ramp_bars.substrate(interleaved)

    windows = list(substrate.windows())
    assert len(windows) == len(left)
    assert set(windows[-1].bars) == {"LEFT", "RIGHT"}
    assert set(windows[-1].features) == {"LEFT", "RIGHT"}
    assert substrate.report().instruments == ("LEFT", "RIGHT")


def test_the_view_is_a_read_only_mapping_of_names_to_numbers() -> None:
    """What an agent holds offers values, refuses writes, and says so when asked."""
    view = _views(ramp_bars.substrate())[-1]

    assert len(view) == 2
    assert sorted(view) == sorted([ramp_bars.MEAN_CLOSE, ramp_bars.MEAN_CLOSE_DELAYED])
    assert "absent" not in view
    with pytest.raises(KeyError):
        view["absent"]
    with pytest.raises(TypeError):
        view[ramp_bars.MEAN_CLOSE] = 0.0
    assert ramp_bars.MEAN_CLOSE in repr(view)


def test_the_report_states_what_the_pass_held() -> None:
    """Memory is stated in bars and bar values, from the configuration not a guess."""
    substrate = ramp_bars.substrate()
    list(substrate.windows())
    report = substrate.report()

    assert report.instruments == ("NULL",)
    assert report.maximum_lookback == ramp_bars.RAMP_MAXIMUM_LOOKBACK
    assert report.bar_fields == len(BAR_FIELDS)
    assert report.window_bar_capacity == ramp_bars.RAMP_MAXIMUM_LOOKBACK
    assert report.window_value_capacity == ramp_bars.RAMP_MAXIMUM_LOOKBACK * len(
        BAR_FIELDS
    )
    assert report.window_carries_across_segments is True
    assert report.exclusions == substrate.exclusions


def test_a_second_walk_does_not_accumulate_the_previous_report() -> None:
    """A pass reports what that pass excluded, never what a previous one did."""
    substrate = ramp_bars.substrate()
    list(substrate.windows())
    first = substrate.exclusions
    list(substrate.windows())

    assert first
    assert substrate.exclusions == first


def test_the_window_is_bounded_by_the_configured_maximum() -> None:
    """History deeper than the maximum is dropped rather than accumulated."""
    records = ramp_bars.as_records(ramp_bars.ramp_bars(count=40))
    substrate = FeatureSubstrate(
        records=records,
        features=ramp_bars.feature_set(maximum=ramp_bars.RAMP_LOOKBACK),
        feature_schema_version=ramp_bars.SCHEMA_VERSION,
    )
    views = _views(substrate)
    bars = ramp_bars.ramp_bars(count=40)

    assert views[-1][ramp_bars.MEAN_CLOSE] == ramp_bars.mean_close(
        bars[-ramp_bars.RAMP_LOOKBACK :]
    )
    assert substrate.report().window_bar_capacity == ramp_bars.RAMP_LOOKBACK


def test_a_row_is_held_in_utc_and_a_naive_one_is_refused() -> None:
    """A row arriving at an offset is carried as the instant it names, not the clock."""
    elsewhere = timezone(timedelta(hours=-5))
    at = datetime(2026, 1, 2, 14, 30, tzinfo=elsewhere)
    record = BarRecord(
        instrument="NULL",
        period_start=at,
        period_end=at + timedelta(minutes=1),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )

    assert record.period_start == at
    assert record.period_start.utcoffset() == timedelta(0)
    assert record.period_start.hour == 19

    with pytest.raises(ValueError, match="timezone-aware"):
        BarRecord(
            instrument="NULL",
            period_start=datetime(2026, 1, 2, 14, 30),
            period_end=datetime(2026, 1, 2, 14, 31),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
        )


def test_a_pass_over_no_rows_produces_nothing() -> None:
    """An empty source is an empty walk, not an error and not one empty window."""
    substrate = ramp_bars.substrate(())

    assert list(substrate.windows()) == []
    assert substrate.exclusions == ()
    assert substrate.report().instruments == ()
    assert substrate.report().window_bar_capacity == 0


#: Where the second segment begins, as a position in the ramp: far enough past
#: the declared lookback that a window which restarted there would be visibly
#: warming up again, and far enough from the end that the segment after it holds
#: several decision points.
SECOND_SEGMENT_FROM = 8

#: The identities and roles the two-segment schedules below are written in.
TRAIN_SEGMENT = "ramp-walk:train"
VALIDATE_SEGMENT = "ramp-walk:validate"


def _two_segment_schedule(records: Sequence[BarRecord]) -> SegmentSchedule:
    """A schedule dividing one continuous ramp into two walk-forward segments."""
    return SegmentSchedule(
        boundaries=(
            SegmentBoundary(
                starts_at=records[0].period_end,
                segment_id=TRAIN_SEGMENT,
                segment_role="train",
            ),
            SegmentBoundary(
                starts_at=records[SECOND_SEGMENT_FROM].period_end,
                segment_id=VALIDATE_SEGMENT,
                segment_role="validate",
            ),
        )
    )


def test_with_no_schedule_the_whole_pass_falls_in_one_segment() -> None:
    """The present behaviour is the default, so nothing already walked changes."""
    windows = list(ramp_bars.substrate().windows())

    assert {window.segment_id for window in windows} == {SINGLE_SEGMENT}
    assert {window.segment_role for window in windows} == {SINGLE_SEGMENT_ROLE}


def test_segment_identity_is_resolved_at_each_decision_point() -> None:
    """One substrate, one window, and the boundary observed where it falls.

    A substrate carrying its segment as a per-instance constant flattens every
    walk-forward segment into one, which is not a loud failure: the run reads as
    a clean single-segment pass and the boundary a held-out period depends on is
    simply not there. So the identity is resolved from the decision point's own
    instant against the schedule the run supplies.
    """
    records = ramp_bars.as_records(ramp_bars.ramp_bars())
    substrate = FeatureSubstrate(
        records=records,
        features=ramp_bars.feature_set(),
        feature_schema_version=ramp_bars.SCHEMA_VERSION,
        segments=_two_segment_schedule(records),
    )

    windows = list(substrate.windows())
    before = windows[:SECOND_SEGMENT_FROM]
    after = windows[SECOND_SEGMENT_FROM:]

    assert before and after
    assert {window.segment_id for window in before} == {TRAIN_SEGMENT}
    assert {window.segment_role for window in before} == {"train"}
    assert {window.segment_id for window in after} == {VALIDATE_SEGMENT}
    assert {window.segment_role for window in after} == {"validate"}


def test_the_window_carries_across_a_boundary_rather_than_restarting_at_it() -> None:
    """A boundary divides what history is used for; it is not a break in the tape.

    The reading the configuration enforces, asserted rather than assumed: the
    feature is readable at the first decision point of the second segment, and
    nothing warms up again there. A substrate per segment — the obvious way to
    get segment identity — would fail both halves.
    """
    records = ramp_bars.as_records(ramp_bars.ramp_bars())
    boundary_at = records[SECOND_SEGMENT_FROM].period_end
    substrate = FeatureSubstrate(
        records=records,
        features=ramp_bars.feature_set(),
        feature_schema_version=ramp_bars.SCHEMA_VERSION,
        segments=_two_segment_schedule(records),
    )

    windows = list(substrate.windows())
    entered = windows[SECOND_SEGMENT_FROM]

    assert entered.as_of == boundary_at
    assert ramp_bars.MEAN_CLOSE in next(iter(entered.features.values()))
    assert substrate.report().window_carries_across_segments is True
    assert [
        exclusion
        for exclusion in substrate.exclusions
        if exclusion.reason == FEATURE_WARM_UP and exclusion.as_of >= boundary_at
    ] == []


def test_a_substrate_per_segment_is_the_discontinuity_the_check_above_catches() -> None:
    """The alternative, shown failing, so the continuity check is not decoration.

    One substrate per segment is the obvious way to obtain segment identity, and
    it is what the assertion above exists to rule out. Here it is built: the
    second segment's own substrate warms every feature up again at the boundary,
    which is an agent's view of history restarting because a fold did. A check
    that has never been shown reporting the defect it names is a check that
    would pass over it.
    """
    records = ramp_bars.as_records(ramp_bars.ramp_bars())
    boundary_at = records[SECOND_SEGMENT_FROM].period_end
    per_segment = FeatureSubstrate(
        records=records[SECOND_SEGMENT_FROM:],
        features=ramp_bars.feature_set(),
        feature_schema_version=ramp_bars.SCHEMA_VERSION,
    )

    windows = list(per_segment.windows())

    assert windows[0].as_of == boundary_at
    assert ramp_bars.MEAN_CLOSE not in next(iter(windows[0].features.values()))
    assert [
        exclusion
        for exclusion in per_segment.exclusions
        if exclusion.reason == FEATURE_WARM_UP and exclusion.as_of == boundary_at
    ] != []


def test_a_decision_point_the_schedule_does_not_cover_is_refused() -> None:
    """A decision point in no declared segment is a wiring defect, not an outcome."""
    records = ramp_bars.as_records(ramp_bars.ramp_bars())
    late = SegmentSchedule(
        boundaries=(
            SegmentBoundary(
                starts_at=records[SECOND_SEGMENT_FROM].period_end,
                segment_id=VALIDATE_SEGMENT,
                segment_role="validate",
            ),
        )
    )
    substrate = FeatureSubstrate(
        records=records,
        features=ramp_bars.feature_set(),
        feature_schema_version=ramp_bars.SCHEMA_VERSION,
        segments=late,
    )

    with pytest.raises(SegmentNotScheduled, match="no declared walk-forward segment"):
        list(substrate.windows())


def test_a_schedule_that_could_not_divide_a_walk_is_refused() -> None:
    """Every way a schedule can fail to be one is refused where it is written down."""
    first = ramp_bars.as_records(ramp_bars.ramp_bars())[0].period_end
    later = ramp_bars.as_records(ramp_bars.ramp_bars())[SECOND_SEGMENT_FROM].period_end

    with pytest.raises(ValueError, match="at least one segment"):
        SegmentSchedule(boundaries=())
    with pytest.raises(ValueError, match="must name the segment"):
        SegmentSchedule(
            boundaries=(
                SegmentBoundary(starts_at=first, segment_id="", segment_role="train"),
            )
        )
    with pytest.raises(ValueError, match="in the order the walk enters them"):
        SegmentSchedule(
            boundaries=(
                SegmentBoundary(
                    starts_at=later,
                    segment_id=VALIDATE_SEGMENT,
                    segment_role="validate",
                ),
                SegmentBoundary(
                    starts_at=first, segment_id=TRAIN_SEGMENT, segment_role="train"
                ),
            )
        )
    with pytest.raises(ValueError, match="entered twice"):
        SegmentSchedule(
            boundaries=(
                SegmentBoundary(
                    starts_at=first, segment_id=TRAIN_SEGMENT, segment_role="train"
                ),
                SegmentBoundary(
                    starts_at=later, segment_id=TRAIN_SEGMENT, segment_role="validate"
                ),
            )
        )


def test_a_schedule_reports_the_segments_it_divides_a_walk_into() -> None:
    """A run states its segments without a reader replaying the walk to find them."""
    records = ramp_bars.as_records(ramp_bars.ramp_bars())
    schedule = _two_segment_schedule(records)

    assert schedule.segments == (TRAIN_SEGMENT, VALIDATE_SEGMENT)
    assert schedule.at(records[0].period_end).segment_id == TRAIN_SEGMENT
    assert schedule.at(records[-1].period_end).segment_id == VALIDATE_SEGMENT
