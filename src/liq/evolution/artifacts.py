"""Typed evolution artifact schema and runtime fingerprint helpers."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from liq.evolution.protocols import EVOLUTION_PROTOCOL_VERSION

ARTIFACT_SCHEMA_VERSION = "2.0"

KNOWN_REJECTION_REASON_CODES: tuple[str, ...] = (
    "all_nan_scores",
    "constraint_saturation",
    "critical_constraint_saturation",
    "degenerate_scores",
    "empty_features",
    "empty_scores",
    "future_reference",
    "invalid_regime_inputs",
    "max_leverage",
    "metric_missing",
    "negative_cash",
    "no_folds",
    "non_finite_scores",
    "ok",
    "regime_features_missing",
    "robustness:drawdown_missing",
    "robustness:max_drawdown_above_cap",
    "robustness:regime_coverage_below_floor",
    "robustness:regime_coverage_missing",
    "robustness:turnover_above_cap",
    "robustness:turnover_missing",
    "score_length_mismatch",
    "threshold_gate",
)
_KNOWN_REJECTION_REASON_SET = frozenset(KNOWN_REJECTION_REASON_CODES)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _seed_lineage_tuple(seed_lineage: Any) -> tuple[int, ...]:
    if seed_lineage is None:
        return ()
    if isinstance(seed_lineage, (int, float)) and not isinstance(seed_lineage, bool):
        return (int(seed_lineage),)
    if not isinstance(seed_lineage, (list, tuple)):
        return ()
    values: list[int] = []
    for entry in seed_lineage:
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            values.append(int(entry))
    return tuple(values)


def is_known_rejection_reason(code: str) -> bool:
    return code in _KNOWN_REJECTION_REASON_SET


def require_known_rejection_reason(code: str) -> str:
    if not is_known_rejection_reason(code):
        raise ValueError(f"unknown rejection reason code: {code!r}")
    return code


@dataclass(frozen=True)
class RejectionEvent:
    """Structured rejection record for explainable candidate filtering."""

    code: str
    stage: str = "unknown"
    detail: str | None = None
    penalty: float = 0.0

    def __post_init__(self) -> None:
        require_known_rejection_reason(self.code)
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.penalty, (int, float)) or not math.isfinite(self.penalty):
            raise ValueError("penalty must be a finite numeric value")
        if float(self.penalty) < 0.0:
            raise ValueError("penalty must be non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "detail": self.detail,
            "penalty": float(self.penalty),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RejectionEvent:
        return cls(
            code=str(payload.get("code", "")),
            stage=str(payload.get("stage", "unknown")),
            detail=payload.get("detail") if isinstance(payload.get("detail"), str) else None,
            penalty=float(payload.get("penalty", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class DependencyFingerprint:
    """Deterministic runtime/dependency fingerprint including seed lineage."""

    python_version: str
    platform: str
    package_versions: tuple[tuple[str, str], ...]
    seed_lineage: tuple[int, ...]
    captured_at_utc: str
    evaluator_fingerprint: str | None = None

    @classmethod
    def capture(
        cls,
        *,
        seed_lineage: tuple[int, ...] | list[int] | int | None = None,
        package_names: tuple[str, ...] = (
            "liq-evolution",
            "liq-gp",
            "liq-features",
            "liq-sim",
            "liq-store",
            "numpy",
            "polars",
        ),
        evaluator_fingerprint: str | None = None,
    ) -> DependencyFingerprint:
        versions: dict[str, str] = {}
        for name in package_names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "missing"
        return cls(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            package_versions=tuple(sorted(versions.items())),
            seed_lineage=_seed_lineage_tuple(seed_lineage),
            captured_at_utc=_utc_now(),
            evaluator_fingerprint=evaluator_fingerprint,
        )

    @property
    def package_version_map(self) -> dict[str, str]:
        return dict(self.package_versions)

    def to_payload(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "package_versions": dict(self.package_versions),
            "seed_lineage": list(self.seed_lineage),
            "captured_at_utc": self.captured_at_utc,
            "evaluator_fingerprint": self.evaluator_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DependencyFingerprint:
        raw_versions = payload.get("package_versions", {})
        versions: dict[str, str] = {}
        if isinstance(raw_versions, Mapping):
            for name, version in raw_versions.items():
                versions[str(name)] = str(version)
        return cls(
            python_version=str(payload.get("python_version", "")),
            platform=str(payload.get("platform", "")),
            package_versions=tuple(sorted(versions.items())),
            seed_lineage=_seed_lineage_tuple(payload.get("seed_lineage")),
            captured_at_utc=str(payload.get("captured_at_utc", "")),
            evaluator_fingerprint=(
                str(payload["evaluator_fingerprint"])
                if payload.get("evaluator_fingerprint") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class EvolutionRunArtifact:
    """Canonical stage-7 run artifact persisted for replay and audit."""

    run_id: str
    dependency_fingerprint: DependencyFingerprint
    selected_candidate_ids: tuple[str, ...]
    per_split_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    rejection_events: tuple[RejectionEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    protocol_version: str = EVOLUTION_PROTOCOL_VERSION
    created_at_utc: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ARTIFACT_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        split_metrics: dict[str, dict[str, float]] = {}
        for split_key, metrics in self.per_split_metrics.items():
            if not isinstance(metrics, Mapping):
                continue
            split_metrics[str(split_key)] = {
                str(name): float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            }

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "protocol_version": self.protocol_version,
            "created_at_utc": self.created_at_utc,
            "dependency_fingerprint": self.dependency_fingerprint.to_payload(),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "per_split_metrics": split_metrics,
            "rejection_events": [event.to_payload() for event in self.rejection_events],
            "metadata": dict(self.metadata),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvolutionRunArtifact:
        if str(payload.get("schema_version", "1.0")) != ARTIFACT_SCHEMA_VERSION:
            try:
                from liq.store.artifacts import normalize_evolution_artifact_payload
            except Exception as exc:
                raise ValueError(
                    "non-canonical artifact schema requires liq-store migration helpers"
                ) from exc
            payload = normalize_evolution_artifact_payload(payload)

        raw_fingerprint = payload.get("dependency_fingerprint")
        if isinstance(raw_fingerprint, Mapping):
            fingerprint = DependencyFingerprint.from_payload(raw_fingerprint)
        else:
            fingerprint = DependencyFingerprint.capture(
                seed_lineage=_seed_lineage_tuple(payload.get("seed_lineage"))
            )

        raw_selected = payload.get("selected_candidate_ids")
        if not isinstance(raw_selected, list):
            raw_selected = payload.get("selected_programs", [])
        if not isinstance(raw_selected, list):
            raw_selected = []

        raw_events = payload.get("rejection_events", [])
        events: list[RejectionEvent] = []
        if isinstance(raw_events, list):
            for raw_event in raw_events:
                if isinstance(raw_event, Mapping):
                    events.append(RejectionEvent.from_payload(raw_event))

        split_metrics = payload.get("per_split_metrics")
        if not isinstance(split_metrics, Mapping):
            split_metrics = {}

        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}

        return cls(
            schema_version=str(payload.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")),
            protocol_version=str(payload.get("protocol_version", EVOLUTION_PROTOCOL_VERSION)),
            created_at_utc=str(payload.get("created_at_utc", _utc_now())),
            dependency_fingerprint=fingerprint,
            selected_candidate_ids=tuple(str(item) for item in raw_selected),
            per_split_metrics=dict(split_metrics),
            rejection_events=tuple(events),
            metadata=dict(metadata),
        )

    @classmethod
    def from_json_bytes(cls, raw_payload: bytes) -> EvolutionRunArtifact:
        parsed = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(parsed, Mapping):
            raise ValueError("artifact payload must decode to a mapping")
        return cls.from_payload(parsed)
