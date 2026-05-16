"""Tests for serialized built-in seed payload catalogs."""

from __future__ import annotations

import pytest

from liq.evolution.seed_catalog import built_in_seed_payloads as catalog_payloads
from liq.evolution.seeds import built_in_seed_payloads


def _walk_nodes(node: dict[str, object]) -> list[dict[str, object]]:
    children = node.get("children")
    if not isinstance(children, list):
        return [node]
    descendants: list[dict[str, object]] = [node]
    for child in children:
        assert isinstance(child, dict)
        descendants.extend(_walk_nodes(child))
    return descendants


@pytest.mark.parametrize(
    ("seed_program_set", "expected_count"),
    [
        ("regime_v1", 3),
        ("regime_v2", 5),
        ("regime_v3", 6),
        ("regime_v4", 5),
    ],
)
def test_built_in_seed_payloads_return_schema_wrapped_programs(
    seed_program_set: str,
    expected_count: int,
) -> None:
    payloads = built_in_seed_payloads(seed_program_set)

    assert len(payloads) == expected_count
    assert payloads == catalog_payloads(seed_program_set)
    for payload in payloads:
        assert payload["schema_version"] == "1.0.0"
        program = payload["program"]
        assert isinstance(program, dict)
        nodes = _walk_nodes(program)
        assert {node["type"] for node in nodes} <= {"terminal", "constant", "function", "parameterized"}
        assert any(node.get("type") == "terminal" and node.get("name") == "close" for node in nodes)
        assert any(node.get("type") in {"function", "parameterized"} for node in nodes)


def test_built_in_seed_payloads_reject_unknown_set() -> None:
    with pytest.raises(ValueError, match="Unknown seed_program_set"):
        built_in_seed_payloads("missing_set")
