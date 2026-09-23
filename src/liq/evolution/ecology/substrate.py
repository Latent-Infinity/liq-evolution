"""The one place bars become numbers, and the only thing agents are shown.

Every agent in the run sees the same broadcast, computed once per decision
point. That is the whole design: feature work is a function of the bars, so
doubling the population doubles the reading and not the computing, and no two
agents can disagree about what a feature was worth at an instant.

Four promises are kept here, and each of them is the kind that cannot be
retrofitted once results exist.

**Nothing is readable before it exists.** Every computed value carries the
instant it becomes available — the close of the bar it was computed from, plus
the feature's declared availability lag. A view is built for one decision point
and refuses any value stamped later than it. A feature computed from a bar's
close and declared with a lag of one is therefore present at that decision point
and unreadable there, rather than quietly absent: an agent that asks for it is
told it is reaching forward, which is a different thing from being told it does
not exist.

**An incomplete bar is excluded, never half-computed.** Records arrive as a
source offers them, with whatever is missing still missing. A record with a
required field absent produces no bar, no decision point and no feature; it
produces an exclusion naming the instant and the field. Nothing is filled, and
the rolling window does not close over the hole — the next bar does not continue
the one before it, so the window begins again behind it.

**No feature is computed on a window shorter than it declared.** A feature is
excluded, with a warm-up reason, at every decision point where its full run of
consecutive bars is not yet behind it — at the start of a series and again after
anything that breaks the run. Computing it short would be worse than excluding
it, because the number would look like every other number.

**The window stays inside.** The rolling bars are held here and reach no
further. What an agent is handed is a mapping of names to numbers with an as-of
instant; there is no attribute, item or method on it through which a past bar or
a prior window can be obtained, so an agent cannot recompute its own history,
cannot reach behind the as-of guard, and cannot hold the whole tape alive by
keeping one view.

The window is materialised once per instrument and updated one bar at a time.
There is no per-bar rebuild from full history and no per-agent copy: every
reader at a decision point is handed the same view object.

**A walk-forward boundary is a label on a decision point, not a break in the
tape.** One pass covers the whole walk and holds one continuous window; which
segment a decision point belongs to is resolved from its own instant against
the schedule the run supplies. The obvious alternative — one substrate per
segment — divides the history and restarts the rolling window with it, which is
the discontinuous reading
:class:`~liq.evolution.ecology.config.EcologyConfig` refuses outright rather
than accepts and ignores. A boundary divides what history is *used for*; an
agent alive on either side of one is the same agent with the same learned
state, which is what makes the boundary a test of generalisation rather than a
restart in disguise. Carrying state forward across a chronological boundary is
not look-ahead; only carrying it backward would be.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.features import FeatureSet
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.types import (
    Bar,
    BarWindow,
    InstrumentId,
    SegmentRole,
    UtcTimestamp,
    _UtcTimestampMixin,
)

__all__ = [
    "BAR_FIELDS",
    "FEATURE_WARM_UP",
    "INCOMPLETE_BAR",
    "SINGLE_SEGMENT",
    "SINGLE_SEGMENT_ROLE",
    "BarRecord",
    "BroadcastFeatureView",
    "Exclusion",
    "FeatureSubstrate",
    "FutureReadError",
    "SegmentBoundary",
    "SegmentNotScheduled",
    "SegmentSchedule",
    "SubstrateReport",
]

#: Reason code: the bar itself was never complete, so nothing was computed from
#: it and no decision point exists at its instant.
INCOMPLETE_BAR = "incomplete_bar"

#: Reason code: the feature's full declared lookback was not yet behind this
#: decision point, so it was excluded rather than computed on a short window.
FEATURE_WARM_UP = "feature_warm_up"

#: The fields a bar must carry to be complete. A record missing any of them
#: describes a bar nobody saw.
BAR_FIELDS = ("open", "high", "low", "close", "volume")

#: What a pass whose run declared no schedule calls the one segment it walks.
#: A single undivided segment is a walk nobody split, which is a different thing
#: from a walk whose splits were lost — so it is named, and the name is the same
#: one every such pass has always used.
SINGLE_SEGMENT = "single"

#: What that one segment is used for.
SINGLE_SEGMENT_ROLE: SegmentRole = "train"


class FutureReadError(EcologyError):
    """A value stamped later than the decision point being processed was read.

    It lives here rather than in :mod:`liq.evolution.ecology.errors` because a
    leaf type arrives with the code that raises it: the broadcast boundary below
    is the only place in the ecology that can still tell a value that exists
    from a value that exists *yet*, so it is the only place this can come from.

    It is a refusal rather than an absence on purpose. A feature that is quietly
    missing reads as one that could not be computed; this says that the read
    reached forward, which is what this program's largest false positive was
    made of. Catching it to carry on is almost always wrong — the one legitimate
    handler is a harness deliberately probing the boundary.
    """


class SegmentNotScheduled(EcologyError):
    """A decision point fell in no segment the run's schedule declares.

    It lives here rather than in :mod:`liq.evolution.ecology.errors` for the
    reason :class:`FutureReadError` does: the pass below is the only place that
    holds both a decision point and the schedule it is supposed to fall in.

    It is raised rather than recorded. An exclusion says a value was not
    computed; this says the run cannot state which walk-forward segment a
    decision it took belongs to, and a decision point silently assigned to the
    nearest segment is how a held-out period acquires training decisions.
    """


@dataclass(frozen=True)
class SegmentBoundary(_UtcTimestampMixin):
    """Where one walk-forward segment begins, and what it is used for.

    Attributes:
        starts_at: The first decision instant inside the segment (UTC). A
            decision point at or after it, and before the next boundary, falls
            in this segment.
        segment_id: Identity the segment is recorded under.
        segment_role: What the segment is used for.
    """

    starts_at: UtcTimestamp
    segment_id: str
    segment_role: SegmentRole

    def __post_init__(self) -> None:
        """Refuse a boundary that names no segment."""
        self._normalize_timestamps("starts_at")
        if not self.segment_id:
            raise ValueError(
                "a walk-forward boundary must name the segment it opens: a "
                "decision point recorded under no identity cannot be attributed "
                "to a training or a held-out period afterwards"
            )


@dataclass(frozen=True)
class SegmentSchedule:
    """How one walk's decision points are divided into walk-forward segments.

    A schedule is a division of *what history is used for*, and nothing else.
    It says which segment a decision point belongs to; it says nothing about
    what an agent remembers, because a boundary is not an event in an agent's
    life. That separation is the whole reason segment identity is resolved per
    decision point from a schedule rather than fixed per substrate: one
    substrate per segment would divide the history and restart the rolling
    window with it, which
    :class:`~liq.evolution.ecology.config.EcologyConfig` refuses outright.

    Attributes:
        boundaries: Where each segment begins, in the order the walk enters
            them. Contiguous by construction — a segment runs until the next one
            opens — so a decision point cannot fall between two of them.

    Raises:
        ValueError: If the schedule divides nothing, is written out of order, or
            enters one segment twice.
    """

    boundaries: tuple[SegmentBoundary, ...]

    def __post_init__(self) -> None:
        """Refuse a schedule that could not divide a walk walked forward once."""
        if not self.boundaries:
            raise ValueError(
                "a schedule divides a walk into at least one segment; an empty "
                "one states a division nobody made"
            )
        entered: set[str] = set()
        previous: SegmentBoundary | None = None
        for boundary in self.boundaries:
            if previous is not None and boundary.starts_at <= previous.starts_at:
                raise ValueError(
                    f"segment '{boundary.segment_id}' opens at "
                    f"{boundary.starts_at.isoformat()}, at or before "
                    f"'{previous.segment_id}' at {previous.starts_at.isoformat()}; "
                    "boundaries are written in the order the walk enters them"
                )
            if boundary.segment_id in entered:
                raise ValueError(
                    f"segment '{boundary.segment_id}' is entered twice; a segment "
                    "is entered, walked and left, because going back is how a "
                    "period held out stops being held out"
                )
            entered.add(boundary.segment_id)
            previous = boundary

    @property
    def segments(self) -> tuple[str, ...]:
        """The segments this schedule divides a walk into, in order."""
        return tuple(boundary.segment_id for boundary in self.boundaries)

    def at(self, as_of: UtcTimestamp) -> SegmentBoundary:
        """The segment one decision instant falls in.

        Raises:
            SegmentNotScheduled: If the instant is before the first boundary,
                and so in no segment this schedule declares.
        """
        found: SegmentBoundary | None = None
        for boundary in self.boundaries:
            if boundary.starts_at > as_of:
                break
            found = boundary
        if found is None:
            raise SegmentNotScheduled(
                f"the decision point at {as_of.isoformat()} falls in no declared "
                f"walk-forward segment: the first opens at "
                f"{self.boundaries[0].starts_at.isoformat()}"
            )
        return found


@dataclass(frozen=True)
class BarRecord(_UtcTimestampMixin):
    """One row as a source of bars offers it, before completeness is judged.

    This is deliberately not a :class:`~liq.evolution.ecology.types.Bar`. A
    ``Bar`` exists only for a bar that was complete, so a source that had to
    produce one would have to decide what to do about a missing field before
    anything could record the decision. A record carries the absence through to
    where it is judged and reported.

    Attributes:
        instrument: The instrument the row describes.
        period_start: Inclusive start of the row's period (UTC).
        period_end: Exclusive end of the row's period (UTC); the instant the bar
            would have become complete and readable.
        open: First traded price in the period, or ``None`` where the source had
            none.
        high: Highest traded price, or ``None``.
        low: Lowest traded price, or ``None``.
        close: Last traded price, or ``None``.
        volume: Traded volume, or ``None``.
    """

    instrument: InstrumentId
    period_start: UtcTimestamp
    period_end: UtcTimestamp
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None

    def __post_init__(self) -> None:
        self._normalize_timestamps("period_start", "period_end")

    @property
    def absent_fields(self) -> tuple[str, ...]:
        """Which required fields the source had no value for."""
        return tuple(name for name in BAR_FIELDS if getattr(self, name) is None)

    def completed(self) -> Bar | None:
        """The bar this row describes, or ``None`` where it was never complete."""
        if (
            self.open is None
            or self.high is None
            or self.low is None
            or self.close is None
            or self.volume is None
        ):
            return None
        return Bar(
            instrument=self.instrument,
            period_start=self.period_start,
            period_end=self.period_end,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass(frozen=True)
class Exclusion(_UtcTimestampMixin):
    """Something that was not computed, and why it was not.

    An exclusion is an outcome and is reported as one. Nothing here is a
    failure to be recovered from: a bar that was never complete and a feature
    that is not yet warm are both ordinary, and the record exists so a run can
    state how often each happened instead of a reader inferring it from gaps.

    Attributes:
        as_of: The decision instant the exclusion belongs to (UTC).
        instrument: The instrument it concerns.
        feature: The feature excluded, or ``None`` where the whole bar was.
        reason: Stable machine-readable code, compared and counted rather than
            read.
        detail: What was missing, in words. Carries field and feature names
            only — never a value, a payload or anything not safe to log.
    """

    as_of: UtcTimestamp
    instrument: InstrumentId
    feature: str | None
    reason: str
    detail: str

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


class BroadcastFeatureView(Mapping[str, float]):
    """One instrument's features at one decision point, as an agent sees them.

    A read-only mapping of names to numbers, and nothing else. It refuses a
    write rather than handing back a private copy, because every agent at this
    decision point holds this same object: a copy-on-write would let one agent
    change what it sees while believing the others had changed with it.

    A value whose availability instant is later than the decision point is
    present and refused. That is not the same as being absent — absent means the
    feature could not be computed, refused means it exists and reaching for it
    is reaching forward — and the two are reported differently on purpose.
    """

    __slots__ = ("_as_of", "_deferred", "_values")

    def __init__(
        self,
        *,
        as_of: UtcTimestamp,
        values: Mapping[str, float],
        deferred: Mapping[str, UtcTimestamp],
    ) -> None:
        """Hold what is readable at ``as_of`` and what is not readable yet."""
        self._as_of = as_of
        self._values = dict(values)
        self._deferred = dict(deferred)

    def __getitem__(self, name: str) -> float:
        """The value of one feature, or a refusal if it is not readable yet."""
        deferred = self._deferred.get(name)
        if deferred is not None:
            raise FutureReadError(
                f"{name!r} becomes readable at {deferred.isoformat()}, which is "
                f"after the decision point at {self._as_of.isoformat()}"
            )
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        """Every feature that can be read here, never one that cannot."""
        return iter(self._values)

    def __len__(self) -> int:
        """How many features can be read here."""
        return len(self._values)

    def __contains__(self, name: object) -> bool:
        """Whether one feature can be read here."""
        return name in self._values

    def __repr__(self) -> str:
        """Name the decision point and what it offers, never the values."""
        return (
            f"BroadcastFeatureView(as_of={self._as_of.isoformat()}, "
            f"readable={sorted(self._values)}, deferred={sorted(self._deferred)})"
        )


@dataclass(frozen=True)
class SubstrateReport:
    """What one pass held in memory, and what it did not compute.

    Attributes:
        instruments: The instruments the pass saw a complete bar for.
        maximum_lookback: The configured depth of the rolling window.
        bar_fields: How many values each held bar carries.
        window_bar_capacity: Instruments times maximum lookback — the most bars
            held at once.
        window_value_capacity: The same in bar values, which is the figure a
            run's memory accounting is stated in.
        window_carries_across_segments: Whether the rolling window survives a
            walk-forward split boundary.
        exclusions: Everything the pass did not compute, with its reason.
    """

    instruments: tuple[InstrumentId, ...]
    maximum_lookback: int
    bar_fields: int
    window_bar_capacity: int
    window_value_capacity: int
    window_carries_across_segments: bool
    exclusions: tuple[Exclusion, ...]


@dataclass
class _Rolling:
    """One instrument's held bars and the values computed from them."""

    bars: deque[Bar]
    last_period_end: UtcTimestamp | None = None
    computed: dict[str, deque[tuple[float, UtcTimestamp]]] = field(default_factory=dict)


@dataclass
class FeatureSubstrate:
    """Compute the configured features once per decision point and broadcast them.

    Attributes:
        records: The rows to walk, in time order. They are consumed once: the
            stream is a walk forward over history, and there is no rewind.
        features: What is computed, how deep each computation reads, and how
            long after its bar each value becomes readable.
        feature_schema_version: Vocabulary the feature names belong to, stamped
            on every decision point so an agent is never silently evaluated
            under a vocabulary it was not born under.
        segments: How the walk's decision points are divided into walk-forward
            segments. ``None`` — the default — is a walk nobody split, recorded
            as the single segment :data:`SINGLE_SEGMENT` in the role
            :data:`SINGLE_SEGMENT_ROLE`, which is what a pass with no schedule
            has always been. A schedule is resolved *per decision point* rather
            than fixed per pass: one substrate per segment would be the obvious
            alternative and is not available, because it restarts the rolling
            window at a fold.
        window_carries_across_segments: Whether the rolling window survives a
            split boundary. Carrying is the continuous reading of the
            requirement — an agent's view of history does not reset because a
            fold did — and here it is true by construction rather than by
            setting: one pass holds one window, and :meth:`_advance` begins it
            again only where the bars themselves stop being consecutive, which
            a boundary does not make them. It is stated so a run's provenance
            records the reading it ran under.
        log: Where this boundary writes what it did. One line per bar that was
            never complete, because that is the outcome only this pass can
            report: a row that produced no bar produces no decision point
            either, so nothing downstream of here ever sees the instant it
            would have been at. A decision point that *was* built is written
            down by the walk that shows it, where a source other than this one
            is recorded the same way. Silent until a run wires a writer.
    """

    records: Iterable[BarRecord]
    features: FeatureSet
    feature_schema_version: str
    segments: SegmentSchedule | None = None
    window_carries_across_segments: bool = True
    log: RunLog = SILENT

    _exclusions: list[Exclusion] = field(default_factory=list, init=False, repr=False)
    _instruments: list[InstrumentId] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def exclusions(self) -> tuple[Exclusion, ...]:
        """Everything the pass so far did not compute, in the order it happened."""
        return tuple(self._exclusions)

    def report(self) -> SubstrateReport:
        """What the pass so far held, and what it excluded."""
        maximum = self.features.maximum_lookback
        instruments = tuple(self._instruments)
        capacity = len(instruments) * maximum
        return SubstrateReport(
            instruments=instruments,
            maximum_lookback=maximum,
            bar_fields=len(BAR_FIELDS),
            window_bar_capacity=capacity,
            window_value_capacity=capacity * len(BAR_FIELDS),
            window_carries_across_segments=self.window_carries_across_segments,
            exclusions=self.exclusions,
        )

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point in chronological order, exactly once."""
        self._exclusions.clear()
        self._instruments.clear()
        held: dict[InstrumentId, _Rolling] = {}
        for as_of, group in _by_decision_point(self.records):
            bars: dict[InstrumentId, Bar] = {}
            views: dict[InstrumentId, BroadcastFeatureView] = {}
            for record in group:
                bar = record.completed()
                if bar is None:
                    self._exclude_bar(record)
                    continue
                if record.instrument not in self._instruments:
                    self._instruments.append(record.instrument)
                rolling = self._advance(held, bar)
                bars[bar.instrument] = bar
                views[bar.instrument] = self._broadcast(rolling, bar, as_of)
            if not bars:
                continue
            segment_id, segment_role = self._segment_at(as_of)
            yield BarWindow(
                as_of=as_of,
                segment_id=segment_id,
                segment_role=segment_role,
                bars=MappingProxyType(bars),
                features=MappingProxyType(views),
                feature_schema_version=self.feature_schema_version,
            )

    def _segment_at(self, as_of: UtcTimestamp) -> tuple[str, SegmentRole]:
        """Which walk-forward segment one decision point falls in.

        Resolved here, at the decision point, rather than held as a property of
        the pass: one pass covers the whole walk, so a segment fixed per pass
        would flatten every boundary in it into one — and a run that reported a
        single segment over a walk that crossed three would read as a clean
        undivided pass rather than as a defect.
        """
        if self.segments is None:
            return SINGLE_SEGMENT, SINGLE_SEGMENT_ROLE
        boundary = self.segments.at(as_of)
        return boundary.segment_id, boundary.segment_role

    def _exclude_bar(self, record: BarRecord) -> None:
        """Record a bar that was never complete, naming what it was missing.

        Which field was missing is written to the exclusion report and not to
        the log. The report is the run's own account of its sample and is read
        with the run; a line is read on its own, and the fields a row did or
        did not carry are the row's content rather than the boundary's.
        """
        self.log.record("bar_excluded", bar_ts=record.period_end, stage="feature")
        absent = ", ".join(record.absent_fields)
        self._exclusions.append(
            Exclusion(
                as_of=record.period_end,
                instrument=record.instrument,
                feature=None,
                reason=INCOMPLETE_BAR,
                detail=(
                    f"required field(s) {absent} absent at the as-of instant, so "
                    "the bar was excluded rather than partially computed"
                ),
            )
        )

    def _advance(self, held: dict[InstrumentId, _Rolling], bar: Bar) -> _Rolling:
        """Add one bar to its instrument's window, beginning again after a break."""
        rolling = held.get(bar.instrument)
        if rolling is None or rolling.last_period_end != bar.period_start:
            rolling = _Rolling(bars=deque(maxlen=self.features.maximum_lookback))
            held[bar.instrument] = rolling
        rolling.bars.append(bar)
        rolling.last_period_end = bar.period_end
        return rolling

    def _broadcast(
        self, rolling: _Rolling, bar: Bar, as_of: UtcTimestamp
    ) -> BroadcastFeatureView:
        """Compute what this decision point can, and state what it cannot."""
        period = bar.period_end - bar.period_start
        readable: dict[str, float] = {}
        deferred: dict[str, UtcTimestamp] = {}
        for declaration in self.features.declarations:
            if len(rolling.bars) < declaration.lookback:
                self._exclusions.append(
                    Exclusion(
                        as_of=as_of,
                        instrument=bar.instrument,
                        feature=declaration.name,
                        reason=FEATURE_WARM_UP,
                        detail=(
                            f"{len(rolling.bars)} consecutive bar(s) behind this "
                            f"decision point against a declared lookback of "
                            f"{declaration.lookback}; excluded rather than "
                            "computed on a short window"
                        ),
                    )
                )
                continue
            computed = rolling.computed.setdefault(
                declaration.name, deque(maxlen=declaration.availability_lag + 1)
            )
            window = tuple(rolling.bars)[-declaration.lookback :]
            computed.append(
                (
                    declaration.compute(window),
                    bar.period_end + declaration.availability_lag * period,
                )
            )
            # Values are held in the order they were computed, so the last one
            # whose availability instant has arrived is the one this decision
            # point may read. A fresher value that has not arrived is only
            # worth refusing while there is no readable one under the same
            # name: once there is, reading the name is not reaching forward,
            # and the fresher value simply belongs to a later decision point.
            for value, available_at in computed:
                if available_at <= as_of:
                    readable[declaration.name] = value
                    continue
                if declaration.name not in readable:
                    deferred[declaration.name] = available_at
                break
        return BroadcastFeatureView(as_of=as_of, values=readable, deferred=deferred)


def _by_decision_point(
    records: Iterable[BarRecord],
) -> Iterator[tuple[UtcTimestamp, Sequence[BarRecord]]]:
    """Group consecutive rows by the instant their period ended."""
    batch: list[BarRecord] = []
    for record in records:
        if batch and record.period_end != batch[0].period_end:
            yield batch[0].period_end, tuple(batch)
            batch = []
        batch.append(record)
    if batch:
        yield batch[0].period_end, tuple(batch)
