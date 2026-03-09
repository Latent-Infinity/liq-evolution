"""Stage-7 typed artifact and rejection-catalog tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from liq.evolution.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    KNOWN_REJECTION_REASON_CODES,
    DependencyFingerprint,
    EvolutionRunArtifact,
    RejectionEvent,
)


def test_dependency_fingerprint_captures_runtime_versions_and_seed_lineage() -> None:
    fingerprint = DependencyFingerprint.capture(
        seed_lineage=(11, 42),
        package_names=("liq-evolution", "liq-gp"),
        evaluator_fingerprint="abc123",
    )
    assert fingerprint.seed_lineage == (11, 42)
    assert fingerprint.package_version_map["liq-evolution"] != ""
    assert fingerprint.package_version_map["liq-gp"] != ""
    assert fingerprint.evaluator_fingerprint == "abc123"


def test_rejection_event_requires_catalog_code() -> None:
    with pytest.raises(ValueError, match="unknown rejection reason code"):
        RejectionEvent(code="unknown_code")


def test_rejection_event_validates_stage_and_penalty() -> None:
    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        RejectionEvent(code="degenerate_scores", stage="")
    with pytest.raises(ValueError, match="finite numeric"):
        RejectionEvent(code="degenerate_scores", penalty=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        RejectionEvent(code="degenerate_scores", penalty=-0.1)


def test_evolution_run_artifact_round_trip() -> None:
    artifact = EvolutionRunArtifact(
        run_id="run-7",
        dependency_fingerprint=DependencyFingerprint.capture(seed_lineage=(20260303,)),
        selected_candidate_ids=("cand-1", "cand-2"),
        per_split_metrics={"time_window:split_0:train": {"cagr": 0.3}},
        rejection_events=(RejectionEvent(code="degenerate_scores", stage="stage_a"),),
        metadata={"owner": "stage7"},
    )
    encoded = artifact.to_json_bytes()
    restored = EvolutionRunArtifact.from_json_bytes(encoded)
    assert restored.schema_version == ARTIFACT_SCHEMA_VERSION
    assert restored.run_id == "run-7"
    assert restored.selected_candidate_ids == ("cand-1", "cand-2")
    assert restored.rejection_events[0].code == "degenerate_scores"


def test_evolution_run_artifact_loads_legacy_schema_via_migration() -> None:
    legacy_payload = {
        "schema_version": "1.0",
        "run_id": "legacy-run",
        "seed": 99,
        "selected_programs": ["cand-a"],
        "rejection_reasons": ["degenerate_scores"],
    }
    artifact = EvolutionRunArtifact.from_payload(legacy_payload)
    assert artifact.schema_version == "2.0"
    assert artifact.dependency_fingerprint.seed_lineage == (99,)
    assert artifact.selected_candidate_ids == ("cand-a",)
    assert artifact.rejection_events[0].code == "degenerate_scores"


def test_rejection_catalog_covers_reason_code_literals_in_source() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "liq" / "evolution"
    pattern = re.compile(r'reason_code\s*=\s*"([^"]+)"')
    discovered: set[str] = set()
    for file_path in source_root.rglob("*.py"):
        discovered.update(pattern.findall(file_path.read_text(encoding="utf-8")))

    assert discovered
    unknown = sorted(
        code for code in discovered if code not in set(KNOWN_REJECTION_REASON_CODES)
    )
    assert unknown == []


def test_dependency_fingerprint_from_payload_handles_non_mapping_versions() -> None:
    fingerprint = DependencyFingerprint.from_payload(
        {
            "python_version": "3.12.0",
            "platform": "x",
            "package_versions": "invalid",
            "seed_lineage": 5,
            "captured_at_utc": "2026-03-03T00:00:00Z",
        }
    )
    assert fingerprint.seed_lineage == (5,)
    assert fingerprint.package_versions == ()


def test_evolution_run_artifact_requires_valid_run_id_and_schema() -> None:
    fingerprint = DependencyFingerprint.capture(seed_lineage=(1,))
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        EvolutionRunArtifact(
            run_id="",
            dependency_fingerprint=fingerprint,
            selected_candidate_ids=("cand",),
        )
    with pytest.raises(ValueError, match="schema_version must be"):
        EvolutionRunArtifact(
            run_id="r",
            dependency_fingerprint=fingerprint,
            selected_candidate_ids=("cand",),
            schema_version="1.0",
        )


def test_evolution_run_artifact_from_payload_uses_fallback_defaults() -> None:
    artifact = EvolutionRunArtifact.from_payload(
        {
            "schema_version": "2.0",
            "run_id": "fallback",
            "selected_programs": ["cand-x"],
            "rejection_events": ["invalid"],
            "per_split_metrics": "invalid",
            "metadata": "invalid",
        }
    )
    assert artifact.selected_candidate_ids == ("cand-x",)
    assert artifact.rejection_events == ()
    assert artifact.per_split_metrics == {}
    assert artifact.metadata == {}


def test_evolution_run_artifact_from_payload_requires_migrator_for_legacy(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args, **kwargs):
        if name == "liq.store.artifacts":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ValueError, match="requires liq-store migration helpers"):
        EvolutionRunArtifact.from_payload({"schema_version": "1.0", "run_id": "legacy"})


def test_evolution_run_artifact_from_json_bytes_requires_mapping() -> None:
    with pytest.raises(ValueError, match="must decode to a mapping"):
        EvolutionRunArtifact.from_json_bytes(b"[]")
