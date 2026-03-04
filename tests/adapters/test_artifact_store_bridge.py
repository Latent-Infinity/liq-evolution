"""Tests for liq-store artifact persistence bridge."""

from __future__ import annotations

import builtins

import pytest

from liq.evolution.adapters.artifact_store import LiqStoreEvolutionArtifactStore
from liq.evolution.artifacts import DependencyFingerprint, EvolutionRunArtifact, RejectionEvent
from liq.evolution.errors import AdapterError


class _MemoryBackend:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data


def test_bridge_round_trip_with_memory_backend() -> None:
    bridge = LiqStoreEvolutionArtifactStore(backend=_MemoryBackend())
    artifact = EvolutionRunArtifact(
        run_id="run-bridge",
        dependency_fingerprint=DependencyFingerprint.capture(seed_lineage=(1, 2)),
        selected_candidate_ids=("cand-1",),
        rejection_events=(RejectionEvent(code="degenerate_scores", stage="stage_a"),),
    )
    key = bridge.save_run_artifact(artifact)
    assert key.endswith("/run.json")

    loaded = bridge.load_run_artifact("run-bridge")
    assert loaded is not None
    assert loaded.run_id == "run-bridge"
    assert loaded.rejection_events[0].code == "degenerate_scores"


def test_bridge_migrates_legacy_payload_from_liq_store_backend() -> None:
    from liq.store import key_builder
    from liq.store.artifacts import serialize_evolution_artifact_payload

    backend = _MemoryBackend()
    bridge = LiqStoreEvolutionArtifactStore(backend=backend)
    legacy_key = key_builder.evolution_run("legacy-bridge", "run.json")
    backend.put(
        legacy_key,
        serialize_evolution_artifact_payload(
            {
                "schema_version": "1.0",
                "run_id": "legacy-bridge",
                "seed": 7,
                "selected_programs": ["cand-legacy"],
                "rejection_reasons": ["degenerate_scores"],
            }
        ),
    )

    loaded = bridge.load_run_artifact("legacy-bridge")
    assert loaded is not None
    assert loaded.schema_version == "2.0"
    assert loaded.dependency_fingerprint.seed_lineage == (7,)
    assert loaded.selected_candidate_ids == ("cand-legacy",)


def test_bridge_data_root_mode_uses_liq_store_local_backend(tmp_path) -> None:  # type: ignore[annotation-unchecked]
    bridge = LiqStoreEvolutionArtifactStore(data_root=tmp_path)
    artifact = EvolutionRunArtifact(
        run_id="run-local",
        dependency_fingerprint=DependencyFingerprint.capture(seed_lineage=(101,)),
        selected_candidate_ids=("cand-101",),
    )
    bridge.save_run_artifact(artifact)
    loaded = bridge.load_run_artifact("run-local")
    assert loaded is not None
    assert loaded.run_id == "run-local"


def test_bridge_requires_backend_or_data_root() -> None:
    with pytest.raises(AdapterError, match="backend or data_root is required"):
        LiqStoreEvolutionArtifactStore()


def test_bridge_artifact_key_fallback_when_liq_store_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "liq.store":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert (
        LiqStoreEvolutionArtifactStore._artifact_key("run-fallback")
        == "evolution/runs/run-fallback/run.json"
    )


def test_bridge_save_and_load_fallback_without_schema_helpers(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "liq.store.artifacts":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    backend = _MemoryBackend()
    bridge = LiqStoreEvolutionArtifactStore(backend=backend)
    artifact = EvolutionRunArtifact(
        run_id="run-fallback-serde",
        dependency_fingerprint=DependencyFingerprint.capture(seed_lineage=(2,)),
        selected_candidate_ids=("cand",),
    )
    bridge.save_run_artifact(artifact)
    loaded = bridge.load_run_artifact("run-fallback-serde")
    assert loaded is not None
    assert loaded.run_id == "run-fallback-serde"


def test_bridge_load_run_artifact_missing_key_returns_none() -> None:
    bridge = LiqStoreEvolutionArtifactStore(backend=_MemoryBackend())
    assert bridge.load_run_artifact("missing") is None


def test_bridge_save_artifact_generic_payload_modes() -> None:
    backend = _MemoryBackend()
    bridge = LiqStoreEvolutionArtifactStore(backend=backend)

    class WithModelDump:
        def model_dump(self) -> dict[str, str]:
            return {"k": "v"}

    class WithPayload:
        payload = {"x": 1}

    bridge.save_artifact("model-dump", WithModelDump())
    bridge.save_artifact("payload-only", WithPayload())
    bridge.save_artifact("plain", object())

    assert bridge.load_artifact("model-dump") == {"k": "v"}
    assert bridge.load_artifact("payload-only") == {"payload": {"x": 1}}
    assert "artifact" in bridge.load_artifact("plain")


def test_bridge_save_run_artifact_redacts_sensitive_metadata() -> None:
    from liq.store import key_builder

    backend = _MemoryBackend()
    bridge = LiqStoreEvolutionArtifactStore(backend=backend)
    artifact = EvolutionRunArtifact(
        run_id="run-redacted",
        dependency_fingerprint=DependencyFingerprint.capture(seed_lineage=(3,)),
        selected_candidate_ids=("cand-redacted",),
        metadata={"api_key": "top-secret-key", "note": {"access_token": "token"}},
    )
    bridge.save_run_artifact(artifact)

    key = key_builder.evolution_run("run-redacted", "run.json")
    raw = backend.get(key)
    assert raw is not None
    decoded = raw.decode("utf-8")
    assert "top-secret-key" not in decoded
    assert '"access_token":"token"' not in decoded
    assert "***REDACTED***" in decoded

    loaded = bridge.load_run_artifact("run-redacted")
    assert loaded is not None
    assert loaded.metadata["api_key"] == "***REDACTED***"
    assert loaded.metadata["note"]["access_token"] == "***REDACTED***"
