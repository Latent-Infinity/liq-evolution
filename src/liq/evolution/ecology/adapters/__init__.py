"""Adapters that satisfy the ecology's ports.

An adapter translates one provider's vocabulary into the ports' domain values
at the boundary, so a provider can be replaced without a line of domain or
use-case code changing. Which adapter a run uses is decided where the run is
assembled, never inside the ecology.

:mod:`liq.evolution.ecology.adapters.null` holds one conforming stand-in per
port. Those are test doubles rather than providers: they read nothing, they
carry no market data, and they exist so a contract suite has something to run
against before a provider-backed adapter exists.

:mod:`liq.evolution.ecology.adapters.liq_sim` is the first that is not a
stand-in: it satisfies the execution port through the platform's execution
simulator, and it is the only module in this library that imports it.

:mod:`liq.evolution.ecology.adapters.array_genome` satisfies the population port
over the ecology's own array-genome store. It is the only module in this library
permitted to reach the evolution engine's selection, variation and archive
primitives, so a second importer of those would be a second engine.
"""

from liq.evolution.ecology.adapters.array_genome import ArrayGenomePopulation
from liq.evolution.ecology.adapters.liq_sim import LiqSimExecutionAdapter
from liq.evolution.ecology.adapters.null import (
    NullAgentPopulation,
    NullBarSource,
    NullExecutionSimulator,
    NullRiskSizer,
    NullSurrogateSource,
)

__all__ = [
    "ArrayGenomePopulation",
    "LiqSimExecutionAdapter",
    "NullAgentPopulation",
    "NullBarSource",
    "NullExecutionSimulator",
    "NullRiskSizer",
    "NullSurrogateSource",
]
