"""Program serialization (re-exported from liq-gp) and Genome serialization."""

from __future__ import annotations

from typing import Any

from liq.evolution.errors import SerializationError
from liq.evolution.program.genome import Genome
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.serialize import (  # noqa: F401
    deserialize,
    deserialize_result,
    serialize,
    serialize_result,
)

GENOME_SCHEMA_VERSION = "1.0"


def serialize_genome(genome: Genome) -> dict[str, Any]:
    """Serialize a Genome to a dict."""
    result: dict[str, Any] = {
        "genome_schema_version": GENOME_SCHEMA_VERSION,
        "entry_program": serialize(genome.entry_program),
    }
    if genome.exit_program is not None:
        result["exit_program"] = serialize(genome.exit_program)
    if genome.risk_genes:
        result["risk_genes"] = dict(genome.risk_genes)
    return result


def deserialize_genome(data: dict[str, Any], registry: PrimitiveRegistry) -> Genome:
    """Deserialize a Genome from a dict.

    Handles legacy Program payloads (no genome_schema_version key)
    by treating them as entry-only Genomes.
    """
    version = data.get("genome_schema_version")
    if version is None:
        # Legacy: single Program payload from liq-gp serialize
        program = deserialize(data, registry)
        return Genome(entry_program=program)

    if version != GENOME_SCHEMA_VERSION:
        raise SerializationError(f"Unknown genome schema version: {version}")

    entry = deserialize(data["entry_program"], registry)
    exit_prog = None
    if "exit_program" in data:
        exit_prog = deserialize(data["exit_program"], registry)
    risk_genes = data.get("risk_genes", {})
    return Genome(entry_program=entry, exit_program=exit_prog, risk_genes=risk_genes)
