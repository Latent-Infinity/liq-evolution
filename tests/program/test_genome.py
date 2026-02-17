"""Tests for Genome v2 structure (Tasks 4.11-4.12)."""

from __future__ import annotations

import dataclasses

import pytest

from liq.evolution.errors import SerializationError
from liq.evolution.program.genome import Genome
from liq.evolution.program.serialize import (
    deserialize_genome,
    serialize_genome,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import TerminalNode
from liq.gp.program.serialize import serialize as gp_serialize
from liq.gp.types import Series


def _make_terminal(name: str = "close") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _make_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register("close", lambda: None, output_type=Series)
    registry.register("volume", lambda: None, output_type=Series)
    return registry


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGenomeConstruction:
    def test_entry_only_genome(self) -> None:
        terminal = _make_terminal()
        genome = Genome(entry_program=terminal)
        assert genome.entry_program is terminal
        assert genome.exit_program is None
        assert genome.risk_genes == {}

    def test_entry_and_exit_genome(self) -> None:
        entry = _make_terminal("close")
        exit_ = _make_terminal("volume")
        genome = Genome(entry_program=entry, exit_program=exit_)
        assert genome.entry_program is entry
        assert genome.exit_program is exit_

    def test_risk_genes_default_empty(self) -> None:
        genome = Genome(entry_program=_make_terminal())
        assert genome.risk_genes == {}

    def test_risk_genes_with_values(self) -> None:
        genes = {"stop_loss_atr_mult": 2.0, "take_profit_atr_mult": 3.0}
        genome = Genome(entry_program=_make_terminal(), risk_genes=genes)
        assert genome.risk_genes == genes

    def test_genome_is_frozen(self) -> None:
        genome = Genome(entry_program=_make_terminal())
        with pytest.raises(dataclasses.FrozenInstanceError):
            genome.entry_program = _make_terminal("volume")  # type: ignore[misc]

    def test_entry_program_accessible(self) -> None:
        terminal = _make_terminal()
        genome = Genome(entry_program=terminal)
        assert genome.entry_program is terminal


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestGenomeSerialization:
    def test_serialize_entry_only_round_trip(self) -> None:
        registry = _make_registry()
        genome = Genome(entry_program=_make_terminal())
        data = serialize_genome(genome)
        restored = deserialize_genome(data, registry)
        assert restored.entry_program == genome.entry_program
        assert restored.exit_program is None
        assert restored.risk_genes == {}

    def test_serialize_entry_and_exit_round_trip(self) -> None:
        registry = _make_registry()
        genome = Genome(
            entry_program=_make_terminal("close"),
            exit_program=_make_terminal("volume"),
        )
        data = serialize_genome(genome)
        restored = deserialize_genome(data, registry)
        assert restored.entry_program == genome.entry_program
        assert restored.exit_program == genome.exit_program

    def test_serialize_with_risk_genes_round_trip(self) -> None:
        registry = _make_registry()
        genes = {"stop_loss_atr_mult": 2.0, "take_profit_atr_mult": 3.0}
        genome = Genome(entry_program=_make_terminal(), risk_genes=genes)
        data = serialize_genome(genome)
        restored = deserialize_genome(data, registry)
        assert restored.risk_genes == genes

    def test_schema_version_included(self) -> None:
        genome = Genome(entry_program=_make_terminal())
        data = serialize_genome(genome)
        assert data["genome_schema_version"] == "1.0"

    def test_legacy_payload_loads_as_entry_only(self) -> None:
        """A liq-gp serialize() payload (no genome_schema_version) loads as entry-only."""
        registry = _make_registry()
        terminal = _make_terminal()
        legacy_data = gp_serialize(terminal)
        # Confirm this is a legacy payload (has "schema_version", no "genome_schema_version")
        assert "schema_version" in legacy_data
        assert "genome_schema_version" not in legacy_data

        restored = deserialize_genome(legacy_data, registry)
        assert restored.entry_program == terminal
        assert restored.exit_program is None
        assert restored.risk_genes == {}

    def test_unknown_version_raises(self) -> None:
        data = {"genome_schema_version": "99.0", "entry_program": {}}
        with pytest.raises(SerializationError, match="Unknown genome schema version"):
            deserialize_genome(data, _make_registry())
