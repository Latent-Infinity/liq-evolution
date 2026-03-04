"""Public seed APIs and deterministic seed-injection helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from liq.evolution.errors import ConfigurationError
from liq.evolution.seed import (
    SeedManifest,
    SeedSpec,
    build_seed_from_spec,
    build_seed_programs,
    build_seed_programs_from_path,
    load_seed_manifest,
    load_seed_specs,
)
from liq.evolution.seed_catalog import (
    SeedTemplateRole,
    StrategySeedTemplate,
    build_strategy_seed,
    build_strategy_seeds,
    get_seed_template,
    list_known_strategy_seeds,
    list_seed_templates_by_role,
)
from liq.gp.config import GPConfig, SeedInjectionConfig
from liq.gp.evolution.injection import inject_seeds
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.types import FitnessResult, GPType


@dataclass(frozen=True)
class SeedInjectionCadence:
    """Deterministic policy for periodic champion-pool seed injection."""

    interval: int = 1
    champion_pool_size: int = 8
    injection_count: int = 2
    method: Literal["direct", "variation", "ramped"] = "variation"

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("SeedInjectionCadence.interval must be >= 1")
        if self.champion_pool_size < 1:
            raise ValueError("SeedInjectionCadence.champion_pool_size must be >= 1")
        if self.injection_count < 1:
            raise ValueError("SeedInjectionCadence.injection_count must be >= 1")


def should_inject(generation: int, cadence: SeedInjectionCadence) -> bool:
    """Return whether seed injection should fire for the given generation."""
    if generation < 0:
        raise ValueError("generation must be >= 0")
    return generation > 0 and generation % cadence.interval == 0


def program_signature(program: Program) -> str:
    """Return a deterministic signature used for seed-diversity tracking."""
    payload = serialize(program)
    return json.dumps(
        payload["program"],
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_objective_direction(direction: str) -> Literal["maximize", "minimize"]:
    lowered = direction.strip().lower()
    if lowered not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'")
    return lowered  # type: ignore[return-value]


def select_seed_champion_pool(
    population: Sequence[Program],
    fitnesses: Sequence[FitnessResult],
    *,
    pool_size: int,
    objective_index: int = 0,
    objective_direction: Literal["maximize", "minimize"] = "maximize",
) -> list[Program]:
    """Select a deterministic, de-duplicated pool of champion seed programs."""
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    if objective_index < 0:
        raise ValueError("objective_index must be >= 0")
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have matching lengths")
    if len(population) == 0:
        return []

    direction = _normalize_objective_direction(objective_direction)
    reverse = direction == "maximize"

    for index, result in enumerate(fitnesses):
        if objective_index >= len(result.objectives):
            raise ValueError(
                f"fitness entry at index {index} has no objective {objective_index}"
            )

    ranked = sorted(
        range(len(population)),
        key=lambda idx: fitnesses[idx].objectives[objective_index],
        reverse=reverse,
    )
    selected: list[Program] = []
    seen_signatures: set[str] = set()
    for index in ranked:
        candidate = population[index]
        signature = program_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(candidate)
        if len(selected) >= pool_size:
            break
    return selected


def build_seed_champion_payload(
    programs: Sequence[Program],
    *,
    source: str = "champion_pool",
) -> dict[str, Any]:
    """Build a portable JSON payload for champion/external seed programs."""
    entries = [
        {"source": source, "program": serialize(program)}
        for program in programs
    ]
    return {"schema_version": "1.0", "seed_programs": entries}


def _load_payload_document(payload: Any) -> Any:
    if isinstance(payload, Path):
        try:
            text = payload.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to read external seed payload from {payload!s}: {exc}"
            ) from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Invalid JSON external seed payload in {payload!s}: {exc}"
            ) from exc

    if isinstance(payload, str):
        path = Path(payload)
        if path.exists():
            return _load_payload_document(path)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Invalid JSON external seed payload string: {exc}"
            ) from exc

    return payload


def _extract_payload_entries(document: Any) -> list[Any]:
    if isinstance(document, list):
        return list(document)

    if isinstance(document, Mapping):
        if "seed_programs" in document:
            seed_programs = document["seed_programs"]
            if not isinstance(seed_programs, list):
                raise ConfigurationError(
                    "external seed payload 'seed_programs' must be a list"
                )
            return list(seed_programs)
        if "best_program" in document:
            return [document["best_program"]]
        if "pareto_front" in document:
            pareto_front = document["pareto_front"]
            if not isinstance(pareto_front, list):
                raise ConfigurationError(
                    "external seed payload 'pareto_front' must be a list"
                )
            return list(pareto_front)
        if "entry_program" in document:
            return [document["entry_program"]]
        if "program" in document:
            return [document]

    raise ConfigurationError(
        "Unsupported external seed payload; expected list, seed_programs, "
        "best_program, pareto_front, entry_program, or program envelope"
    )


def validate_external_seed_payload(
    payload: Any,
    registry: PrimitiveRegistry,
    *,
    strict: bool = True,
) -> list[Program]:
    """Parse and validate external seed payload programs.

    When ``strict=False``, malformed entries are ignored and only valid
    programs are returned.
    """
    if payload is None:
        return []

    try:
        document = _load_payload_document(payload)
        entries = _extract_payload_entries(document)
    except ConfigurationError:
        if strict:
            raise
        return []

    programs: list[Program] = []
    for index, entry in enumerate(entries):
        try:
            if not isinstance(entry, Mapping):
                raise ConfigurationError(
                    f"external seed entry at index {index} must be an object"
                )
            if "schema_version" in entry and "program" in entry:
                raw_program = entry
            else:
                raw_program = entry["program"] if "program" in entry else entry
            if not isinstance(raw_program, Mapping):
                raise ConfigurationError(
                    f"external seed entry at index {index} has invalid 'program' payload"
                )
            programs.append(deserialize(dict(raw_program), registry))
        except Exception as exc:
            if strict:
                raise ConfigurationError(
                    f"invalid external seed entry at index {index}"
                ) from exc
            continue
    return programs


def _merge_seed_sources(
    champions: Sequence[Program],
    external_programs: Sequence[Program],
) -> list[Program]:
    merged: list[Program] = []
    seen_signatures: set[str] = set()
    for source in (champions, external_programs):
        for program in source:
            signature = program_signature(program)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged.append(program)
    return merged


def inject_seed_programs_from_champion_pool(
    population: Sequence[Program],
    fitnesses: Sequence[FitnessResult],
    *,
    generation: int,
    cadence: SeedInjectionCadence,
    gp_config: GPConfig,
    registry: PrimitiveRegistry,
    rng: Any,
    external_seed_payload: Any | None = None,
    objective_index: int = 0,
    objective_direction: Literal["maximize", "minimize"] = "maximize",
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
    elite_indices: set[int] | None = None,
    injection_event: int | None = None,
    output_type: GPType | None = None,
) -> tuple[list[Program], int]:
    """Inject deterministic champions + optional external seeds into population."""
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have matching lengths")
    if not should_inject(generation, cadence):
        return list(population), 0

    champions = select_seed_champion_pool(
        population,
        fitnesses,
        pool_size=cadence.champion_pool_size,
        objective_index=objective_index,
        objective_direction=objective_direction,
    )
    external_programs = validate_external_seed_payload(
        external_seed_payload,
        registry,
        strict=False,
    )
    seed_programs = _merge_seed_sources(champions, external_programs)
    if cadence.method != "ramped" and not seed_programs:
        return list(population), 0

    replaceable_count = len(population) - len(elite_indices or set())
    if replaceable_count < 1:
        return list(population), 0
    injected_count = min(cadence.injection_count, replaceable_count)

    seed_injection = SeedInjectionConfig(
        interval=1,
        count=injected_count,
        method=cadence.method,
    )
    local_config = gp_config.model_copy(update={"seed_injection": seed_injection})
    seeds_or_none: Sequence[Program] | None = seed_programs or None
    return inject_seeds(
        list(population),
        list(fitnesses),
        seeds_or_none,
        local_config,
        registry,
        rng,
        generation,
        ranks=ranks,
        crowding=crowding,
        elite_indices=elite_indices,
        injection_event=injection_event,
        output_type=output_type,
    )


__all__ = [
    "SeedTemplateRole",
    "StrategySeedTemplate",
    "build_strategy_seed",
    "build_strategy_seeds",
    "get_seed_template",
    "list_known_strategy_seeds",
    "list_seed_templates_by_role",
    "SeedManifest",
    "SeedSpec",
    "build_seed_from_spec",
    "build_seed_programs",
    "build_seed_programs_from_path",
    "load_seed_manifest",
    "load_seed_specs",
    "SeedInjectionCadence",
    "build_seed_champion_payload",
    "inject_seed_programs_from_champion_pool",
    "program_signature",
    "select_seed_champion_pool",
    "should_inject",
    "validate_external_seed_payload",
]
