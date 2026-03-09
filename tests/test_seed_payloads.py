from __future__ import annotations

import pytest

from liq.evolution.seeds import built_in_seed_payloads


@pytest.mark.parametrize(
    ("seed_program_set", "expected_count"),
    [
        ("regime_v1", 3),
        ("regime_v2", 5),
        ("regime_v3", 6),
        ("regime_v4", 5),
    ],
)
def test_built_in_seed_payloads_known_sets(
    seed_program_set: str,
    expected_count: int,
) -> None:
    payloads = built_in_seed_payloads(seed_program_set)

    assert len(payloads) == expected_count
    assert all(payload["schema_version"] == "1.0.0" for payload in payloads)
    assert all(isinstance(payload["program"], dict) for payload in payloads)


def test_built_in_seed_payloads_rejects_unknown_set() -> None:
    with pytest.raises(ValueError, match="Unknown seed_program_set"):
        built_in_seed_payloads("unknown")
