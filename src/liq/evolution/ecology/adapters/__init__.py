"""Adapters that satisfy the ecology's ports.

An adapter translates one provider's vocabulary into the ports' domain values
at the boundary, so a provider can be replaced without a line of domain or
use-case code changing. Which adapter a run uses is decided where the run is
assembled, never inside the ecology.

:mod:`liq.evolution.ecology.adapters.null` holds one conforming stand-in per
port. Those are test doubles rather than providers: they read nothing, they
carry no market data, and they exist so a contract suite has something to run
against before a provider-backed adapter exists.
"""

from liq.evolution.ecology.adapters.null import (
    NullAgentPopulation,
    NullBarSource,
    NullExecutionSimulator,
    NullRiskSizer,
    NullSurrogateSource,
)

__all__ = [
    "NullAgentPopulation",
    "NullBarSource",
    "NullExecutionSimulator",
    "NullRiskSizer",
    "NullSurrogateSource",
]
