"""The capabilities the ecology reaches the rest of the platform through.

Each port names a capability the ecology needs — supply me decision points,
tell me what I may hold, act on it, keep my population, draw me a declared null
— and says nothing about who provides it. Every signature is written in the
domain's own value types, so an adapter can be replaced without a line of
domain or use-case code changing, and a provider's vocabulary cannot leak
inward by being mentioned in a signature.

Ports are protocols, not base classes: an adapter satisfies one by having the
right shape, never by inheriting from anything here. Each is runtime-checkable
so that the contract suite every adapter is run against can assert conformance
of the object it was handed, not of the class it was told about.

Three seams the ecology also crosses deliberately have no port of their own:
the per-bar feature view and the walk-forward split boundary are delivered by
:class:`BarSource`, and the cost scenario is resolved once outside the ecology
and handed to :class:`ExecutionSimulator` by name. Each sits behind a port that
already exists rather than becoming a seam domain code would cross directly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

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
    Intent,
    NotFilled,
    NullDeclaration,
    PositionTarget,
    SizingOutcome,
    SurrogateSeries,
)

__all__ = [
    "AgentPopulation",
    "BarSource",
    "ExecutionSimulator",
    "RiskSizer",
    "SurrogateSource",
]


@runtime_checkable
class BarSource(Protocol):
    """Supply decision points over real market history, strictly forward.

    Contract:

    * Windows arrive in chronological order and each is yielded exactly once.
      There is no epoch, replay, rewind or seek: history is walked, not
      trained on repeatedly, and the absence of such a capability is part of
      what this port promises.
    * A window exists only where the underlying bar was complete at its as-of
      instant. An incomplete bar is excluded, never partially computed and
      never filled in.
    * Feature values are computed once per decision point and shared by every
      reader. The view a window carries is read-only; readers do not receive
      private copies.
    * Nothing a window exposes depends on information that became available
      after its as-of instant.
    * Each window carries the walk-forward segment it falls in, so a split
      boundary is observable to the caller as a change of segment identity and
      no separate boundary source is needed.
    """

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point in chronological order, exactly once."""
        ...


@runtime_checkable
class RiskSizer(Protocol):
    """Decide what an agent is permitted to hold, given what it wants.

    Contract:

    * Returns a single :class:`~liq.evolution.ecology.types.SizingOutcome`:
      what may be held, together with every bound that bound. The two are not
      alternatives, because bounds are not all of one kind and the two kinds
      do not have the same right answer.
    * A bound on **direction** is refused whole. A short wish under a long-only
      mandate breaches the mandate rather than overshooting it — no smaller
      short would have been permitted either — so the outcome carries no
      target and names the reason, and the wish is never re-signed into a long.
    * A bound on **magnitude** is applied and recorded. An oversized or
      over-levered wish differs from a permitted one only in size, so the
      mandate reduces it to what it allows: that reduced exposure is the one
      that would be traded, and therefore the one the agent is scored on, so
      discarding it would score an agent on a position nobody held. The outcome
      carries the reduced target and names every bound that reduced it, so a
      number smaller than the wish is never a silent one. Where a bound on size
      leaves nothing that can be traded, no target is carried and the reason is
      named all the same.
    * Rejection reasons are stable codes, so refusals can be counted and
      compared across runs rather than read.
    * The mandate itself is not expressed here. This port applies it; which
      bounds are in force, and which deliberately are not, is the adapter's
      configuration and is recorded there.
    """

    def size(self, intent: Intent, account: AccountState) -> SizingOutcome:
        """Decide what ``intent`` may hold against ``account``, and what bound it."""
        ...


@runtime_checkable
class ExecutionSimulator(Protocol):
    """Act on a permitted target against the bar that acts on it.

    Contract:

    * ``cost_scenario_id`` names the single cost scenario every charge is drawn
      from. It is resolved once, outside the ecology, and handed to the adapter
      by name. No cost is chosen, defaulted or written down here, and every
      outcome carries the name so a result can be reproduced and re-costed.
    * :meth:`cost_provenance` states what that scenario *effectively* charges
      once every leg of it has been applied. A scenario is applied as written,
      including a leg the book being modelled does not trade: a parameter
      dropped in silence would make the charge a number the harness chose, so
      the effective figure and the treatment of the hedge leg are both readable.
    * :meth:`execute` reports what happened, not what was asked for. Either
      something traded — a fill, priced inside the bar and charged under the
      named scenario — or nothing did, and the outcome says so and names a
      stable reason for it. A target that is not reached is never reported as a
      fill of the request; scoring reads the realised value either way. How much
      of a difference between the requested and the realised exposure is even
      expressible belongs to the model behind this port: under an adapter over a
      simulator that fills an order in full or not at all, the outcome is
      all-or-nothing and a partial quantity cannot arise.
    * The bar handed to :meth:`execute` is the bar the target is acted on, which
      is not the bar it was formed on. A target formed at a bar's close and
      acted on inside that same bar would trade at prices that printed before it
      was formed; an adapter is therefore entitled to require that a target have
      aged by the execution model's own minimum delay, and to refuse rather than
      absorb a target that has not.
    * Account state is handed in and handed back. The port keeps no hidden
      per-agent state between calls, so a run can be resumed from a state that
      was handed out rather than from a provider's private memory.
    """

    cost_scenario_id: CostScenarioId

    def cost_provenance(self) -> CostProvenance:
        """Return what the named scenario effectively charges, as applied here."""
        ...

    def execute(
        self,
        target: PositionTarget,
        bar: Bar,
        account: AccountState,
    ) -> tuple[Fill | NotFilled, AccountState]:
        """Act on ``target`` at ``bar`` and return the outcome and the new state."""
        ...


@runtime_checkable
class AgentPopulation(Protocol):
    """Hold the living agents and the record of what behaviours have been seen.

    Contract:

    * An agent's heritable part and its learned part are reachable separately.
      Anything that can only hand back one combined state cannot express the
      question of which part a birth carries across, which is the reason the
      separation is a contract rather than a convenience.
    * :meth:`spawn` is the whole of inheritance: parents in, one offspring
      genome out, variation already applied. Which variation — recombination
      between two parents, perturbation of one, or both — is the adapter's
      business; the ecology asks for a child, not for an operator.
    * Spawning is reproducible: the same parents and the same seed produce the
      same offspring.
    * The diversity record decides what it keeps. :meth:`record` offers an
      entry and reports whether it was kept; the ecology does not reach into
      the record to place or evict anything itself.
    """

    def agent_ids(self) -> tuple[AgentId, ...]:
        """Return the identities of the living agents."""
        ...

    def genome(self, agent_id: AgentId) -> Genome:
        """Return the heritable part of one agent."""
        ...

    def learned_state(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return what one agent has learned, separately from its genome."""
        ...

    def descriptor(self, agent_id: AgentId) -> Descriptor:
        """Return where one agent currently sits in behaviour space."""
        ...

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Genome:
        """Return one offspring genome of ``parents``, with variation applied."""
        ...

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the diversity record; report whether it was kept."""
        ...

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the diversity record currently holds."""
        ...


@runtime_checkable
class SurrogateSource(Protocol):
    """Draw from one null whose construction is declared.

    Contract:

    * :meth:`declaration` states what the construction preserves, what it
      destroys and the hypothesis a draw from it tests. A source that cannot
      say this is not usable as a null, because the same number means
      different things depending on which structure survived.
    * Draws are reproducible: the same series, replicate and seed give the same
      surrogate, and the surrogate carries all three so a null distribution can
      be rebuilt from its record alone.
    * A draw is derived from the real series it is handed. Nothing here
      invents, imputes or extends data.
    """

    def declaration(self) -> NullDeclaration:
        """Return what this null preserves, destroys and therefore tests."""
        ...

    def draw(
        self,
        observed: Sequence[float],
        *,
        source_id: str,
        replicate: int,
        seed: int,
    ) -> SurrogateSeries:
        """Return one reproducible surrogate of ``observed`` under this null."""
        ...
