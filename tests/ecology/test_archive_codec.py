"""What the record codec refuses to persist.

The codec reads the platform archive through the few members it needs. Every
occupant of a record the ecology fills is an entry it offered; an occupant that
is anything else is not part of the record, and persisting the rest of the
record without it would write a payload that silently disagrees with the record
it claims to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from liq.evolution.ecology.adapters import archive_codec
from liq.evolution.ecology.types import ArchiveEntry, Descriptor, Genome

AXES = ("activity",)


def _entry() -> ArchiveEntry:
    return ArchiveEntry(
        agent_id="codec-agent",
        genome=Genome(genes=MappingProxyType({"scale": 0.5}), schema_version="g"),
        descriptor=Descriptor(
            values=MappingProxyType({"activity": 0.5}), schema_version="d"
        ),
        objectives=MappingProxyType({"deflated": 1.0}),
        recorded_at=datetime(2000, 1, 1, tzinfo=UTC),
    )


@dataclass
class _Held:
    """A record holding the occupants it is handed, in order."""

    occupants: list[object] = field(default_factory=list)
    bins_per_dim: tuple[int, ...] = (11,)
    bin_capacity: int = 1

    @property
    def filled_bins(self) -> int:
        return len(self.occupants)

    def elites(self) -> list[object]:
        return list(self.occupants)

    def insert(self, individual, objectives, descriptors) -> bool:
        self.occupants.append(individual)
        return True


def _persisted(record: _Held):
    return archive_codec.persisted(
        record, run_id="codec-run", descriptor_schema_version="d", axes=AXES
    )


def test_a_record_of_entries_persists_every_one() -> None:
    entry = _entry()
    payload = _persisted(_Held([entry]))
    assert tuple(archive_codec.decode_entry(text) for text in payload.entries) == (
        entry,
    )


@pytest.mark.parametrize("foreign", ["an occupant", None, 1.0])
def test_an_occupant_that_is_not_an_entry_is_refused_rather_than_skipped(
    foreign: object,
) -> None:
    """The record is persisted whole or not at all."""
    with pytest.raises(TypeError, match="ArchiveEntry"):
        _persisted(_Held([_entry(), foreign]))
