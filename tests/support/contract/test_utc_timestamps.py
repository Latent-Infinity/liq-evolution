from datetime import UTC, datetime, timedelta, timezone

import pytest

from liq.evolution.ecology.types import Bar, Intent


def test_ecology_timestamps_are_normalized_to_utc() -> None:
    offset = timezone(timedelta(hours=-5))
    bar = Bar(
        instrument="AAPL",
        period_start=datetime(2026, 1, 2, 7, tzinfo=offset),
        period_end=datetime(2026, 1, 2, 8, tzinfo=offset),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )

    assert bar.period_start == datetime(2026, 1, 2, 12, tzinfo=UTC)
    assert bar.period_end.tzinfo == UTC


def test_ecology_timestamps_reject_naive_values() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Intent(
            agent_id="agent-1",
            instrument="AAPL",
            target_exposure=0.1,
            as_of=datetime(2026, 1, 2),
        )
