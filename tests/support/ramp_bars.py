"""The declared placeholder ramp, offered as the rows a substrate consumes.

Nothing here is market data and nothing here may be read as market data. The
bars are the arithmetic ramp :class:`~liq.evolution.ecology.adapters.NullBarSource`
declares — a unit step at a constant half-range and a constant volume — taken
apart into the rows a source of bars offers before completeness is judged. They
exist so that the substrate's *shape* can be checked in this repository, which
cannot reach the approved fixtures and must not learn how to: the claims about
what happens at real bar boundaries are evidence and live where the tape does.

Every number a test here computes is a number about the ramp. None of them says
anything about any instrument.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.features import FeatureDeclaration, FeatureSet
from liq.evolution.ecology.substrate import BAR_FIELDS, BarRecord, FeatureSubstrate
from liq.evolution.ecology.types import Bar

#: How much history the ramp features read. Short, so a pass over a handful of
#: decision points still shows a warm-up and a broadcast either side of it.
RAMP_LOOKBACK = 5

#: The deepest window the ramp configuration will hold.
RAMP_MAXIMUM_LOOKBACK = 10

#: How many decision points a ramp pass covers by default.
RAMP_DECISION_POINTS = 12

#: Vocabulary the ramp feature names belong to.
SCHEMA_VERSION = "ramp-features-1"

#: The ramp features: one readable where it is computed, one declared a bar late.
MEAN_CLOSE = "ramp_close_mean"
MEAN_CLOSE_DELAYED = "ramp_close_mean_delayed"


def mean_close(bars: Sequence[Bar]) -> float:
    """Mean close over whatever window the declaration asked for."""
    return sum(bar.close for bar in bars) / len(bars)


def ramp_bars(
    count: int = RAMP_DECISION_POINTS, *, instrument: str = "NULL"
) -> tuple[Bar, ...]:
    """The ramp's bars for one instrument, in order."""
    source = NullBarSource(window_count=count, instrument=instrument)
    return tuple(bar for window in source.windows() for bar in window.bars.values())


def as_records(bars: Iterable[Bar]) -> tuple[BarRecord, ...]:
    """The same bars as the rows a source offers, with nothing missing."""
    return tuple(
        BarRecord(
            instrument=bar.instrument,
            period_start=bar.period_start,
            period_end=bar.period_end,
            **{name: getattr(bar, name) for name in BAR_FIELDS},
        )
        for bar in bars
    )


def without(record: BarRecord, field: str) -> BarRecord:
    """The same row with one required field removed, as an incomplete bar."""
    values = {name: getattr(record, name) for name in BAR_FIELDS}
    values[field] = None
    return BarRecord(
        instrument=record.instrument,
        period_start=record.period_start,
        period_end=record.period_end,
        **values,
    )


def feature_set(*, lag: int = 0, maximum: int = RAMP_MAXIMUM_LOOKBACK) -> FeatureSet:
    """A ramp feature set: one prompt feature and one declared ``lag`` bars late."""
    return FeatureSet(
        declarations=(
            FeatureDeclaration(
                name=MEAN_CLOSE,
                lookback=RAMP_LOOKBACK,
                availability_lag=0,
                compute=mean_close,
            ),
            FeatureDeclaration(
                name=MEAN_CLOSE_DELAYED,
                lookback=RAMP_LOOKBACK,
                availability_lag=lag,
                compute=mean_close,
            ),
        ),
        maximum_lookback=maximum,
    )


def substrate(
    records: Iterable[BarRecord] | None = None,
    *,
    features: FeatureSet | None = None,
) -> FeatureSubstrate:
    """A substrate over the ramp, ready to walk."""
    return FeatureSubstrate(
        records=as_records(ramp_bars()) if records is None else records,
        features=feature_set() if features is None else features,
        feature_schema_version=SCHEMA_VERSION,
    )
