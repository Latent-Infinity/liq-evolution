"""Cells counted from outside the loop: the record's, and the living's.

Both counts use the platform archive's own public binning, ``QDArchive.cell_of``,
on the record's own archive, so a check never re-implements the bin arithmetic
and never builds a second archive. The record's count reads the entries the
record holds; the living count reads each living agent's descriptor through the
population port and never the record. A descriptor the archive cannot bin, or an
agent the population cannot describe, occupies no cell.
"""

from __future__ import annotations

from liq.evolution.ecology import descriptors
from liq.evolution.ecology.adapters import ArrayGenomePopulation


def _coordinate(population: ArrayGenomePopulation, values) -> tuple[float, ...]:
    return tuple(values[axis] for axis in descriptors.AXES)


def record_cells(population: ArrayGenomePopulation) -> int:
    """How many distinct cells the record's held entries sit in."""
    return len(
        {
            population.archive.cell_of(_coordinate(population, entry.descriptor.values))
            for entry in population.recorded()
        }
    )


def living_cells(population: ArrayGenomePopulation) -> int:
    """How many distinct cells the living sit in, from their descriptors alone."""
    cells: set[tuple[int, ...]] = set()
    for agent_id in population.agent_ids():
        try:
            described = population.descriptor(agent_id)
        except descriptors.DescriptorUnavailable:
            continue
        try:
            cells.add(
                population.archive.cell_of(_coordinate(population, described.values))
            )
        except ValueError:
            continue
    return len(cells)
