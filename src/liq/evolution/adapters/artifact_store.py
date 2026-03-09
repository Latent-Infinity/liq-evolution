"""liq-store persistence bridge for evolution run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from liq.core.security import serialize_sensitive_payload
from liq.evolution.artifacts import EvolutionRunArtifact
from liq.evolution.errors import AdapterError
from liq.evolution.protocols import EVOLUTION_PROTOCOL_VERSION, StoreBackend


class LiqStoreEvolutionArtifactStore:
    """Persist and replay evolution artifacts through a StoreBackend."""

    protocol_version = EVOLUTION_PROTOCOL_VERSION

    def __init__(
        self,
        backend: StoreBackend | None = None,
        *,
        data_root: str | Path | None = None,
    ) -> None:
        if backend is None:
            if data_root is None:
                raise AdapterError("backend or data_root is required")
            try:
                from liq.store.artifacts import LocalArtifactStore
            except Exception as exc:  # pragma: no cover - defensive
                raise AdapterError(
                    "liq-store LocalArtifactStore is required for data_root mode"
                ) from exc
            backend = LocalArtifactStore(data_root)
        self._backend = backend

    @staticmethod
    def _artifact_key(run_id: str) -> str:
        try:
            from liq.store import key_builder

            return key_builder.evolution_run(run_id, "run.json")
        except Exception:
            return f"evolution/runs/{run_id}/run.json"

    def save_run_artifact(self, artifact: EvolutionRunArtifact) -> str:
        key = self._artifact_key(artifact.run_id)
        payload = artifact.to_payload()
        try:
            from liq.store.artifacts import serialize_evolution_artifact_payload

            encoded = serialize_evolution_artifact_payload(payload)
        except Exception:
            encoded = serialize_sensitive_payload(payload)
        self._backend.put(key, encoded)
        return key

    def load_run_artifact(self, run_id: str) -> EvolutionRunArtifact | None:
        key = self._artifact_key(run_id)
        raw = self._backend.get(key)
        if raw is None:
            return None
        try:
            from liq.store.artifacts import deserialize_evolution_artifact_payload

            payload = deserialize_evolution_artifact_payload(raw)
            return EvolutionRunArtifact.from_payload(payload)
        except Exception:
            return EvolutionRunArtifact.from_json_bytes(raw)

    def save_artifact(self, artifact_id: str, artifact: object) -> None:
        if isinstance(artifact, EvolutionRunArtifact):
            self.save_run_artifact(artifact)
            return

        key = self._artifact_key(artifact_id)
        if hasattr(artifact, "model_dump") and callable(artifact.model_dump):
            payload = artifact.model_dump()  # type: ignore[attr-defined]
        elif hasattr(artifact, "payload"):
            payload = {"payload": artifact.payload}
        else:
            payload = {"artifact": str(artifact)}
        encoded = serialize_sensitive_payload(
            {"__artifact_type__": "generic", "payload": payload},
        )
        self._backend.put(key, encoded)

    def load_artifact(self, artifact_id: str) -> Any | None:
        key = self._artifact_key(artifact_id)
        raw = self._backend.get(key)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict) and parsed.get("__artifact_type__") == "generic":
                return parsed.get("payload")
        except Exception:
            pass

        try:
            from liq.store.artifacts import deserialize_evolution_artifact_payload

            payload = deserialize_evolution_artifact_payload(raw)
            return EvolutionRunArtifact.from_payload(payload)
        except Exception:
            try:
                return EvolutionRunArtifact.from_json_bytes(raw)
            except Exception:
                return json.loads(raw.decode("utf-8"))
