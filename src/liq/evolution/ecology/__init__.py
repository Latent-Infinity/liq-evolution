"""Agent ecology: a population that lives, learns and is scored bar by bar.

The ecology walks real market history once, forward. At each decision point the
features are computed a single time and broadcast to every agent; each agent
forms an intent, is told what it may hold, has that acted on, and is scored on
what actually happened to its own evaluation account.

This package owns the domain's vocabulary and the capabilities it reaches the
rest of the platform through. :mod:`liq.evolution.ecology.types` holds the
immutable values that cross those boundaries; :mod:`liq.evolution.ecology.ports`
holds the five capabilities themselves — supplying decision points, deciding
what may be held, acting on it, keeping the population, and drawing from a
declared null. Domain and use-case code depends on these protocols only.
Adapters that translate a particular library into them live at the edges, and
which adapter is used is decided where the run is assembled, never here.
"""

from liq.evolution.ecology.ports import (
    AgentPopulation,
    BarSource,
    ExecutionSimulator,
    RiskSizer,
    SurrogateSource,
)
from liq.evolution.ecology.types import (
    AccountState,
    AgentId,
    ArchiveEntry,
    Bar,
    BarWindow,
    CostScenarioId,
    Descriptor,
    Fill,
    Genome,
    InstrumentId,
    Intent,
    NullDeclaration,
    PositionTarget,
    Rejection,
    SegmentRole,
    SizingOutcome,
    SurrogateSeries,
    UtcTimestamp,
)

__all__ = [
    "AccountState",
    "AgentId",
    "AgentPopulation",
    "ArchiveEntry",
    "Bar",
    "BarSource",
    "BarWindow",
    "CostScenarioId",
    "Descriptor",
    "ExecutionSimulator",
    "Fill",
    "Genome",
    "InstrumentId",
    "Intent",
    "NullDeclaration",
    "PositionTarget",
    "Rejection",
    "RiskSizer",
    "SegmentRole",
    "SizingOutcome",
    "SurrogateSeries",
    "SurrogateSource",
    "UtcTimestamp",
]
