"""Config-driven seed loading and program construction utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import Program
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog import get_seed_template

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml: Any = None


def _normalize_seed_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()


class SeedSpec(BaseModel):
    """Single strategy seed configuration item."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("strategy")
    @classmethod
    def normalize_strategy(cls, value: str) -> str:
        return _normalize_seed_name(value)


class SeedManifest(BaseModel):
    """Container for one or more seed strategies."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    seeds: list[SeedSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def enabled_seeds(self) -> list[SeedSpec]:
        return [seed for seed in self.seeds if seed.enabled]


def _read_payload(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to read seed config from {path!s}: {exc}"
            ) from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Invalid JSON in seed config {path!s}: {exc}"
            ) from exc

    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ConfigurationError(
                f"PyYAML is required to load {path.suffix!s} seed configs; "
                "install pyyaml and retry"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to read seed config from {path!s}: {exc}"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML in seed config {path!s}: {exc}"
            ) from exc

    raise ConfigurationError(f"Unsupported seed config extension for {path!s}")


def _coerce_seed_entries(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw: list[dict[str, Any]] = []
        for index, value in enumerate(payload):
            if not isinstance(value, dict):
                raise ConfigurationError(
                    f"Invalid seed entry in {source!s} at index {index}; expected object"
                )
            raw.append(value)
        return raw

    if isinstance(payload, dict):
        if "seeds" in payload:
            seeds_payload = payload["seeds"]
            if not isinstance(seeds_payload, list):
                raise ConfigurationError(
                    f"Invalid seed config in {source!s}; 'seeds' must be a list"
                )
            return _coerce_seed_entries(seeds_payload, source)
        if "seed" in payload:
            seed_payload = payload["seed"]
            if not isinstance(seed_payload, dict):
                raise ConfigurationError(
                    f"Invalid seed config in {source!s}; 'seed' must be an object"
                )
            return [seed_payload]
        if "strategy" in payload:
            return [payload]

    raise ConfigurationError(
        f"Invalid seed config format in {source!s}; expected object with strategy/seed/seeds or array"
    )


def _load_manifest_from_file(path: Path) -> SeedManifest:
    payload = _read_payload(path)
    if payload is None:
        return SeedManifest(
            schema_version=1, seeds=[], metadata={"source": path.as_posix()}
        )
    if not isinstance(payload, dict):
        entries = _coerce_seed_entries(payload, path)
        seeds: list[SeedSpec] = []
        for index, raw in enumerate(entries):
            try:
                seeds.append(SeedSpec.model_validate(raw))
            except ValidationError as exc:
                raise ConfigurationError(
                    f"Invalid seed entry in {path!s} at index {index}: {exc}"
                ) from exc
        return SeedManifest(seeds=seeds, metadata={"source": path.as_posix()})

    if "seeds" in payload or "schema_version" in payload:
        try:
            return SeedManifest.model_validate(payload)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid seed manifest in {path!s}: {exc}"
            ) from exc

    entries = _coerce_seed_entries(payload, path)
    seeds: list[SeedSpec] = []
    for index, raw in enumerate(entries):
        try:
            seeds.append(SeedSpec.model_validate(raw))
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid seed entry in {path!s} at index {index}: {exc}"
            ) from exc
    return SeedManifest(seeds=seeds, metadata={"source": path.as_posix()})


def _iter_seed_files(path: Path) -> list[Path]:
    return sorted(
        [
            *path.glob("*.json"),
            *path.glob("*.yml"),
            *path.glob("*.yaml"),
        ]
    )


def load_seed_manifest(path: str | Path) -> SeedManifest:
    """Load a seed manifest from a JSON/YAML file or directory of files."""
    location = Path(path)
    if not location.exists():
        raise ConfigurationError(f"Seed path does not exist: {location}")

    if location.is_dir():
        manifests = [
            _load_manifest_from_file(file_path)
            for file_path in _iter_seed_files(location)
        ]
        seeds: list[SeedSpec] = [
            seed for manifest in manifests for seed in manifest.seeds
        ]
        return SeedManifest(
            schema_version=1,
            seeds=seeds,
            metadata={"directory": location.as_posix()},
        )

    if not location.is_file():
        raise ConfigurationError(f"Seed path is not a file: {location}")
    return _load_manifest_from_file(location)


def load_seed_specs(path: str | Path) -> list[SeedSpec]:
    """Load enabled seed specs from a config path."""
    return load_seed_manifest(path).enabled_seeds


def build_seed_from_spec(
    spec: SeedSpec,
    registry: PrimitiveRegistry,
) -> Program:
    """Build a single seeded strategy from a validated seed spec."""
    template = get_seed_template(spec.strategy)
    return template.builder(registry, **spec.params)


def build_seed_programs(
    specs: Iterable[SeedSpec],
    registry: PrimitiveRegistry,
) -> list[Program]:
    """Build seeds for all enabled specs."""
    programs: list[Program] = []
    for spec in specs:
        if not spec.enabled:
            continue
        programs.append(build_seed_from_spec(spec, registry))
    return programs


def build_seed_programs_from_path(
    path: str | Path, registry: PrimitiveRegistry
) -> list[Program]:
    """Load and build all enabled seed specs from a config path."""
    return build_seed_programs(load_seed_specs(path), registry)


__all__ = [
    "SeedManifest",
    "SeedSpec",
    "build_seed_from_spec",
    "build_seed_programs",
    "build_seed_programs_from_path",
    "load_seed_manifest",
    "load_seed_specs",
]
