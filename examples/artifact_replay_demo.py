#!/usr/bin/env python3
"""Demonstrate artifact persistence and replay through liq-store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from liq.evolution import (
    ARTIFACT_SCHEMA_VERSION,
    DependencyFingerprint,
    EvolutionRunArtifact,
    LiqStoreEvolutionArtifactStore,
    RejectionEvent,
)


def make_artifact() -> EvolutionRunArtifact:
    fingerprint = DependencyFingerprint.capture(
        seed_lineage=(20260303, 20260304),
        evaluator_fingerprint="stage10-demo-v2",
    )
    return EvolutionRunArtifact(
        run_id="stage10-replay-demo",
        dependency_fingerprint=fingerprint,
        selected_candidate_ids=("cand-101", "cand-202"),
        per_split_metrics={"time_window:split_0:train": {"sharpe_ratio": 1.11, "max_drawdown": 0.03}},
        rejection_events=(
            RejectionEvent(code="degenerate_scores", stage="stage_a", penalty=0.0),
            RejectionEvent(
                code="regime_features_missing", stage="stage_b", detail="demo fallback", penalty=0.05
            ),
        ),
        metadata={"api_key": "demo-key", "note": {"access_token": "demo-token"}},
    )


def run_demo() -> None:
    artifact = make_artifact()
    with tempfile.TemporaryDirectory() as data_root:
        root = Path(data_root)
        bridge = LiqStoreEvolutionArtifactStore(data_root=root)

        run_id = artifact.run_id
        key = bridge.save_run_artifact(artifact)
        print(f"stored artifact key: {key}")

        loaded = bridge.load_run_artifact(run_id)
        assert loaded is not None
        print(f"loaded run_id: {loaded.run_id}")
        print(f"schema_version: {loaded.schema_version}")
        print(f"artifact selected candidates: {loaded.selected_candidate_ids}")
        print(f"replay rejection events: {[event.code for event in loaded.rejection_events]}")

        payload_path = root / key
        raw_payload = payload_path.read_text(encoding="utf-8")
        print(f"raw payload path: {payload_path}")
        print(f"contains redacted api_key: {'***REDACTED***' in raw_payload}")
        print(f"schema version constant: {ARTIFACT_SCHEMA_VERSION}")


if __name__ == "__main__":
    run_demo()
