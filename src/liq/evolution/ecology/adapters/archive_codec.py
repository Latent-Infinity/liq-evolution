"""Persisting the diversity record, and restoring it into the run that wrote it.

The platform archive's own serialisation keeps an occupant as its ``name`` or,
failing that, its ``repr``; an entry has no name, so its genome and descriptor
would persist as text nothing parses back, and the archive's own restore replays
that text through insertion. This module is the caller's side of that exchange,
done properly: each entry is encoded as text from which the same entry is
decoded, and the whole is wrapped in an envelope saying what it is a record of.

**The envelope** (:class:`~liq.evolution.ecology.types.RecordPayload`) names the
payload layout's version, the run that wrote it, the descriptor vocabulary, the
record's axes, its cells per axis and its cell capacity. A restore is refused,
before any entry is decoded, if any of those differs from the record it is
handed to, or if that record already holds an entry. A record is restored only
into the run that wrote it, for restart; it is never seeded into another run.

**Restoring replays.** The entries are handed back to the record's own insertion
in the order the record held them. With every entry of a cell mutually kept,
and no cell over its capacity, the replay keeps each of them in the same cell
and order; that it does is asserted by the round-trip evidence, never assumed.

Nothing here imports the platform archive: it reads the record through the few
members it needs, so the one module allowed to reach the evolution engine is
still the array-genome adapter that composes it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Protocol

from liq.evolution.ecology.types import (
    ArchiveEntry,
    Descriptor,
    Genome,
    RecordPayload,
)

__all__ = [
    "PAYLOAD_VERSION",
    "decode_entry",
    "encode_entry",
    "persisted",
    "refuse_a_foreign_payload",
    "restore",
]

#: The version of the payload layout this module writes and reads.
PAYLOAD_VERSION = "ecology-record-payload-1"


class _HeldRecord(Protocol):
    """What this module reads of the platform archive holding the record."""

    bins_per_dim: tuple[int, ...]
    bin_capacity: int

    @property
    def filled_bins(self) -> int: ...

    def elites(self) -> list[object]: ...

    def insert(
        self,
        individual: object,
        objectives: tuple[float, ...] | list[float],
        descriptors: tuple[float, ...] | list[float],
    ) -> bool: ...


def encode_entry(entry: ArchiveEntry) -> str:
    """One entry as text from which :func:`decode_entry` recovers it exactly.

    Every number is written as the shortest text that reads back as the same
    double, and the recording instant in ISO 8601 with its UTC offset.
    """
    return json.dumps(
        {
            "agent_id": entry.agent_id,
            "genome": {
                "genes": dict(entry.genome.genes),
                "schema_version": entry.genome.schema_version,
            },
            "descriptor": {
                "values": dict(entry.descriptor.values),
                "schema_version": entry.descriptor.schema_version,
            },
            "objectives": dict(entry.objectives),
            "recorded_at": entry.recorded_at.isoformat(),
        },
        allow_nan=False,
    )


def decode_entry(text: str) -> ArchiveEntry:
    """The entry :func:`encode_entry` wrote as ``text``."""
    raw = json.loads(text)
    return ArchiveEntry(
        agent_id=raw["agent_id"],
        genome=Genome(
            genes=MappingProxyType(dict(raw["genome"]["genes"])),
            schema_version=raw["genome"]["schema_version"],
        ),
        descriptor=Descriptor(
            values=MappingProxyType(dict(raw["descriptor"]["values"])),
            schema_version=raw["descriptor"]["schema_version"],
        ),
        objectives=MappingProxyType(dict(raw["objectives"])),
        recorded_at=datetime.fromisoformat(raw["recorded_at"]),
    )


def refuse_a_foreign_payload(
    payload: RecordPayload,
    *,
    run_id: str,
    descriptor_schema_version: str,
    axes: Sequence[str],
    bins: Sequence[int],
    bin_capacity: int,
    held: int,
) -> None:
    """Refuse a payload that is not a record of this run, in this record's shape.

    The refusals every record makes alike, stated once so no two records
    disagree about what they will load.

    Raises:
        ValueError: If the payload's version, run, vocabulary, axes, cells per
            axis or cell capacity differ from those given, or if the record it
            is handed to already holds an entry.
    """
    if payload.payload_version != PAYLOAD_VERSION:
        raise ValueError(
            f"a payload of version {payload.payload_version!r} is not one this "
            f"record reads; it reads version {PAYLOAD_VERSION!r}"
        )
    if payload.run_id != run_id:
        raise ValueError(
            f"a record written by run {payload.run_id!r} restores only into that "
            f"run, never into run {run_id!r}"
        )
    if payload.descriptor_schema_version != descriptor_schema_version:
        raise ValueError(
            f"a record described in vocabulary {payload.descriptor_schema_version!r} "
            f"cannot be restored into one kept in {descriptor_schema_version!r}"
        )
    if tuple(payload.axes) != tuple(axes):
        raise ValueError(
            f"a record on axes {tuple(payload.axes)} cannot be restored into one "
            f"on axes {tuple(axes)}"
        )
    if tuple(payload.bins) != tuple(bins):
        raise ValueError(
            f"a record of bins {tuple(payload.bins)} cannot be restored into one "
            f"of bins {tuple(bins)}"
        )
    if payload.bin_capacity != bin_capacity:
        raise ValueError(
            f"a record of cell capacity {payload.bin_capacity} cannot be restored "
            f"into one of capacity {bin_capacity}"
        )
    if held:
        raise ValueError(
            f"a record is restored only into an empty one, and this one holds "
            f"{held} entr{'y' if held == 1 else 'ies'}"
        )


def persisted(
    record: _HeldRecord,
    *,
    run_id: str,
    descriptor_schema_version: str,
    axes: Sequence[str],
) -> RecordPayload:
    """The platform archive's entries, encoded in its order, inside their envelope.

    Raises:
        TypeError: If the archive holds an occupant that is not an
            :class:`~liq.evolution.ecology.types.ArchiveEntry`. The record is
            persisted whole or not at all: a payload that left the occupant out
            would disagree with the record it names.
    """
    entries = []
    for occupant in record.elites():
        if not isinstance(occupant, ArchiveEntry):
            raise TypeError(
                f"the record holds an occupant of type {type(occupant).__name__}, "
                "not an ArchiveEntry; only entries the ecology offered are "
                "persisted, and none is skipped"
            )
        entries.append(encode_entry(occupant))
    return RecordPayload(
        payload_version=PAYLOAD_VERSION,
        run_id=run_id,
        descriptor_schema_version=descriptor_schema_version,
        axes=tuple(axes),
        bins=tuple(record.bins_per_dim),
        bin_capacity=record.bin_capacity,
        entries=tuple(entries),
    )


def restore(
    record: _HeldRecord,
    payload: RecordPayload,
    *,
    run_id: str,
    descriptor_schema_version: str,
    axes: Sequence[str],
) -> None:
    """Replay ``payload``'s entries into the empty platform archive, in order.

    Raises:
        ValueError: As :func:`refuse_a_foreign_payload` does, before any entry
            is decoded.
    """
    refuse_a_foreign_payload(
        payload,
        run_id=run_id,
        descriptor_schema_version=descriptor_schema_version,
        axes=axes,
        bins=record.bins_per_dim,
        bin_capacity=record.bin_capacity,
        held=record.filled_bins,
    )
    for text in payload.entries:
        entry = decode_entry(text)
        record.insert(
            entry,
            tuple(entry.objectives[name] for name in sorted(entry.objectives)),
            tuple(entry.descriptor.values[axis] for axis in axes),
        )
