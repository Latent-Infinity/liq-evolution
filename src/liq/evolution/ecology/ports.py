"""The capabilities the ecology reaches the rest of the platform through.

Each port names a capability the ecology needs — supply me decision points,
tell me what I may hold, act on it, keep my population, draw me a declared null,
deflate the fitness I compare on — and says nothing about who provides it. Every signature is written in the
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

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import (
    AccountState,
    AgentId,
    ArchiveEntry,
    Bar,
    BarWindow,
    CostProvenance,
    CostScenarioId,
    DeflatedFitness,
    Descriptor,
    Fill,
    FitnessSample,
    Genome,
    Intent,
    NotFilled,
    NullDeclaration,
    Offspring,
    PositionTarget,
    RecordCoverage,
    RecordPayload,
    SizingOutcome,
    SurrogateSeries,
)

__all__ = [
    "AgentPopulation",
    "BarSource",
    "ExecutionSimulator",
    "FitnessDeflator",
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
    * :meth:`spawn` is the whole of inheritance: parents in, one
      :class:`~liq.evolution.ecology.types.Offspring` out, variation already
      applied. Which variation — recombination between two parents,
      perturbation of one, or both — is the adapter's business; the ecology
      asks for a child, not for an operator. What it gets back says what the
      variation did: the mutation scale in force and how often the boundary
      rule acted on each gene, so a birth can be checked against the rule
      declared for it.
    * Spawning is reproducible: the same parents and the same seed produce the
      same offspring.
    * :meth:`select_parents` chooses who breeds, through the platform's own
      selection rather than one written here. It returns exactly ``count``
      parents, each of them alive, reproducibly for a seed. The fitness it is
      handed is keyed by the living population *as it is at that call* — every
      living agent and nobody else — and each figure is finite or ``None``,
      higher preferred. A fitness keyed otherwise is refused rather than
      answered, because a population that turns over drifts away from any size
      a configuration recorded, and a selection sized from that configuration
      returns quietly too few parents or reads past the figures it was given.
    * ``None`` means the agent has no figure. It has no place in the ordering
      and is never chosen, unless **no** agent has a figure, in which case
      parents are drawn uniformly from the living, reproducibly for the seed.
      A mapping in which some agents, but fewer than
      :attr:`minimum_candidates`, carry a figure is refused: the caller must
      decide whether the trigger is blind.
    * The diversity record decides what it keeps. :meth:`record` offers an
      entry and reports whether it was kept; the ecology does not reach into
      the record to place or evict anything itself.
    * An agent the population cannot yet describe — one the store holds too
      little of what its descriptor reads — is refused by :meth:`descriptor`
      with :class:`~liq.evolution.ecology.descriptors.DescriptorUnavailable`,
      and given no coordinate in its place.
    * The record reports what it holds. :meth:`coverage` is read from the
      record itself and agrees with :meth:`recorded`: the cells filled, the
      cells there are, and per axis how many held entries sit in each cell.
      An entry whose coordinate the record cannot bin is not kept, changes
      nothing else, and is handed back by :meth:`excluded`.
    * The record persists for restart of the run that wrote it, and only that
      run. :meth:`persisted` reads the record into a payload whose envelope
      names the payload version, the run, the descriptor vocabulary and the
      record's cells; :meth:`restore` loads such a payload into an empty
      record as the same entries in the same order, and refuses one whose
      envelope names anything else, or a record that already holds an entry.
    """

    @property
    def minimum_candidates(self) -> int:
        """The fewest agents with a figure this population's selection can choose among."""
        ...

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
        """Return where one agent currently sits in behaviour space.

        Raises:
            DescriptorUnavailable: If the agent cannot yet be described.
        """
        ...

    def spawn(self, *, parents: Sequence[AgentId], seed: int) -> Offspring:
        """Return one offspring of ``parents``, with the variation it applied."""
        ...

    def select_parents(
        self, *, fitness: Mapping[AgentId, float | None], count: int, seed: int
    ) -> tuple[AgentId, ...]:
        """Return ``count`` living parents chosen on ``fitness``, reproducibly."""
        ...

    def record(self, entry: ArchiveEntry) -> bool:
        """Offer ``entry`` to the diversity record; report whether it was kept."""
        ...

    def recorded(self) -> tuple[ArchiveEntry, ...]:
        """Return the entries the diversity record currently holds."""
        ...

    def coverage(self) -> RecordCoverage:
        """Return the cells the diversity record fills, read from the record."""
        ...

    def excluded(self) -> tuple[ArchiveEntry, ...]:
        """Return every entry refused for a coordinate the record cannot bin."""
        ...

    def persisted(self, run_id: str) -> RecordPayload:
        """Return the diversity record as a payload written by ``run_id``."""
        ...

    def restore(self, payload: RecordPayload, run_id: str) -> None:
        """Load ``payload`` into this empty record, for restart of ``run_id``.

        Raises:
            ValueError: If the payload's envelope names another payload
                version, run, vocabulary, axes, cells or cell capacity, or if
                the record already holds an entry.
        """
        ...


def refuse_an_unanswerable_selection(
    *,
    fitness: Mapping[AgentId, float | None],
    living: Sequence[AgentId],
    count: int,
    minimum: int,
) -> None:
    """Refuse a parent selection the population port's contract cannot answer.

    The refusals every population makes alike, stated once so no two adapters
    disagree about what they refuse: a request for no parents, a fitness not
    keyed by exactly the living population, a figure that has no place in an
    ordering, and figures too few to choose among that are not none at all.

    Args:
        fitness: Each living agent's figure, or ``None`` where it has none.
        living: The living population at this call.
        count: How many parents are asked for.
        minimum: The population's declared ``minimum_candidates``.

    Raises:
        ValueError: If ``count`` is below one, if ``fitness`` names an agent
            that is not alive or leaves out one that is, if any present figure
            is not finite, or if some but fewer than ``minimum`` agents carry a
            figure. Each is a caller holding a stale or impossible view of the
            population, or one that has not decided whether its trigger is
            blind — a defect to fix, not a condition to handle.
    """
    if count < 1:
        raise ValueError(
            "a selection must ask for at least one parent: a request for none "
            "chooses nobody and is not a selection"
        )
    if set(fitness) != set(living):
        raise ValueError(
            "fitness must be keyed by exactly the living population at this "
            f"call: {len(set(fitness) - set(living))} named agent(s) are not "
            f"alive and {len(set(living) - set(fitness))} living agent(s) carry "
            "no entry"
        )
    figures = [value for value in fitness.values() if value is not None]
    if not all(math.isfinite(value) for value in figures):
        raise ValueError(
            "every present fitness figure must be finite: a missing number has "
            "no place in an ordering, and a selection that accepted one would "
            "choose by the order agents happen to be held in; an agent with no "
            "figure is handed in as None"
        )
    if 0 < len(figures) < minimum:
        raise ValueError(
            f"{len(figures)} agent(s) carry a figure, fewer than the minimum of "
            f"{minimum} this population's selection can choose among: hand in "
            "every figure as None if the trigger is blind, never a partial "
            "ordering"
        )


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


@runtime_checkable
class FitnessDeflator(Protocol):
    """Deflate each agent's fitness for the comparison it is part of.

    Contract:

    * :meth:`deflate` returns one
      :class:`~liq.evolution.ecology.types.DeflatedFitness` per sample, in the
      order handed in, each deflated for the number of samples handed in.
    * The undeflated figure travels beside the deflated one, and the same
      samples give the same results.
    * A result the deflator cannot assess carries no deflated figure, and no
      undeflated figure either where it has none. Its standing is
      ``"indistinguishable"``. It is never scored, padded or given a
      placeholder.
    * :attr:`minimum_window` is the fewest outcomes a window must hold for this
      deflator to be able to assess an agent on its own evidence. It is at
      least two, the port's own refusal floor. An agent with a longer window
      may still be unassessed, for cross-sectional or edge reasons.
    * A window of fewer than two outcomes, a non-finite outcome, or an
      independent-outcome count below two or above the window is refused as an
      :class:`~liq.evolution.ecology.errors.EcologyError`, never scored, padded
      or skipped: an agent too young to have the evidence is a condition a
      turning-over population reaches, not a defect in the caller.
    * For any two comparison sizes at which an agent with an unchanged sample
      is assessed, the deflated figure at the larger size is no higher, and an
      ``"indistinguishable"`` standing at the smaller size stays
      ``"indistinguishable"`` at the larger. An agent may be unassessed at one
      size and assessed at another, because a cross-section can be too thin to
      assess anyone. That is the absence of a verdict, not a verdict made
      easier. When the same per-outcome figure rests on more independent
      outcomes, with the comparison unchanged, a ``"better"`` or ``"worse"``
      standing is kept, and the deflated figure moves, if at all, in that
      standing's direction: no lower for ``"better"``, no higher for
      ``"worse"``. An ``"indistinguishable"`` agent's figure may move either
      way, because more evidence makes a deflation surer of whichever side of
      its bar the agent is on.
    * Each result's standing is the deflator's verdict, at its own declared
      confidence, on whether and in which direction the agent is told apart
      from the population. A reader orders and culls on what it is told; it
      never infers a direction by comparing figures itself.
    * The formula is the platform's single statistics authority's. No deflation
      is computed in this library; an adapter reaches that authority from the
      consumer that composes the run.
    """

    @property
    def minimum_window(self) -> int:
        """The fewest outcomes a window needs for this deflator to assess it."""
        ...

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        """Return each sample's deflated figure and standing, in order.

        A result whose deflated figure is ``None`` was not assessed; see
        :attr:`DeflatedFitness.assessed
        <liq.evolution.ecology.types.DeflatedFitness.assessed>`.
        """
        ...


class UndeflatableSample(EcologyError):
    """A sample carries too little, or unusable, evidence to be deflated."""


def refuse_an_undeflatable_sample(samples: Sequence[FitnessSample]) -> None:
    """Refuse a deflation the fitness-deflation port's contract cannot answer.

    The refusals every deflator makes alike, stated once so no two adapters
    disagree about what they refuse.

    Raises:
        UndeflatableSample: If any sample's window holds fewer than two
            outcomes, holds an outcome that is not finite, or states an
            independent-outcome count below two or above the window.
    """
    for each in samples:
        if len(each.outcomes) < 2:
            raise UndeflatableSample(
                f"{each.agent_id}: a window of fewer than two outcomes carries "
                "no spread and cannot be deflated"
            )
        if not all(math.isfinite(outcome) for outcome in each.outcomes):
            raise UndeflatableSample(
                f"{each.agent_id}: a non-finite outcome is never scored, "
                "imputed or skipped"
            )
        if not 2 <= each.independent_outcomes <= len(each.outcomes):
            raise UndeflatableSample(
                f"{each.agent_id}: independent outcomes must number at least two "
                "and at most the window"
            )
