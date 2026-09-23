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

:mod:`liq.evolution.ecology.accounts`, :mod:`liq.evolution.ecology.agent`,
:mod:`liq.evolution.ecology.config`, :mod:`liq.evolution.ecology.driver`,
:mod:`liq.evolution.ecology.substrate`, :mod:`liq.evolution.ecology.features`
and :mod:`liq.evolution.ecology.errors`
are reached as attributes of this package rather than re-exported name by name. The boundary ``__all__`` names is
the ports and the values that cross them, and it is asserted to be exactly that;
what a slice builds behind the boundary stays addressed by its own module, so a
reader can tell a capability from a contract by how it is written down.
"""

from liq.evolution.ecology import (  # noqa: F401
    accounts,
    agent,
    config,
    driver,
    errors,
    features,
    substrate,
)
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
    CostProvenance,
    CostScenarioId,
    Descriptor,
    Fill,
    Genome,
    InstrumentId,
    Intent,
    NotFilled,
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
    "CostProvenance",
    "CostScenarioId",
    "Descriptor",
    "ExecutionSimulator",
    "Fill",
    "Genome",
    "InstrumentId",
    "Intent",
    "NotFilled",
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
