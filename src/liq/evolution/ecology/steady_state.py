"""The steady-state loop: births and deaths between bars, never at generation boundaries.

A population walks history once, forward, through
:class:`~liq.evolution.ecology.population_accounts.PopulationAccounts`, and at
every decision point that falls on a trigger it is turned over: some eligible
agents die, and as many are born. Nothing here computes a statistic. Every verdict
comes from the fitness-deflation port, every parent from the population port's
selection, and every genome from its ``spawn``. What is decided here is only the
order those are asked in and what is done with the answers.

**When.** A trigger falls at every positive multiple of the configured period: a
function of the bar index and configuration, never of the data, and never fired
because some agent became worse. Turnover happens after the bar has been settled
and every account has published it, and before anyone decides at it, so a
turnover at a decision point reads outcomes up to and including the one that
became known there — the outcome of the decision before it — and nothing later.

**Who dies** (``plans/oracle-v3-indistinguishable-cull-2026-09-30.md`` §2.2). An
agent may die only once it has lived the configured minimum age. The eligible
agents are handed to the deflator together, once; exactly ``min(k, eligible)`` of
them die. Those the deflation tells apart as worse are culled first, lowest
deflated figure first. The places left are filled by retiring the oldest agents
the evidence is silent on — indistinguishable or not assessed at all — and only
when none is left the oldest it told apart as better, each recorded as a
protected retirement. Ties go to the earlier birth bar, then the earlier
admission. The order consults no fitness figure and draws no random number.

**Who is born** (``plans/oracle-v3-unassessed-fitness-2026-10-01.md`` §2.2). One
birth per death. Parents are chosen by the population port from the figures of
the same deflate call that decided the deaths. An agent that call did not assess
— too young to be in it, or given no figure — is handed to selection as having
no figure, and is never chosen. When fewer surviving agents carry a figure than
the port's own floor, the whole trigger is blind: every agent is handed without a
figure and the port draws parents uniformly, at the same birth count. A trigger
is wholly blind or wholly on figures. Each offspring is exactly what ``spawn``
returned, and enters through the population's one admission path; one the path
refuses is drawn again from fresh parents, up to the configured number of times.

**One exit.** Every death goes through :meth:`SteadyState._die`: one departure
from the population, which records the kind on the lineage, and one account
close, which knows nothing of the kind. Why an agent died is a label on its
record and never a branch in its accounting.

*The diversity record.* With the record's offers configured on
(``diversity_record = "record_only"``), at every trigger, after its one deflate
call and before its first death, the loop offers the population's diversity
record one entry for each eligible agent that call assessed and whose descriptor
is defined, carrying exactly that call's deflated figure under the name
``"deflated"`` and the agent's genome and descriptor at the trigger; an agent
without a figure is never offered, and the undeflated figure never is. Nothing
in the loop reads the record: no birth, death, parent choice or admission
consults an entry, so discarding every offer changes nothing the loop decides,
and a cell's occupant — the highest figure offered to it, each relative to its
own trigger — is a record, not a ranking. An eviction removes an entry, never an
agent; a death removes an agent, never an entry. The record carries across split
boundaries and never from one run to another. Its coverage is cumulative and is
reported beside the living occupancy, which is read from the living and never
from the record. Configuring the offers on is not niching: no niching rule acts
on the living. With the offers off (the default) the loop offers nothing
(``liq-docs/plans/oracle-v4-archive-interaction-2026-10-02.md``;
``liq-docs/plans/oracle-v4-offers-under-no-rule-2026-10-13.md``).

**Allocation.** With an allocation configured, the loop allocates through
:class:`~liq.evolution.ecology.population_accounts.FixedShareAllocation`, the one
driver's hook: once per decision point, after settlement and the trigger's
births and deaths and before the population decides, the one aggregator's
weights are updated over exactly the living, from the outcomes and reference
return recorded at that point; after the decision the aggregate book is netted
from those weights and what each living agent was permitted, bounded by the
portfolio-risk port the loop was handed, and banded, and its target is acted on
at the next bar. Every allocation event is written under the ``"allocate"``
stage. Nothing turnover decides reads the weights or the book, and the
allocation reads no record entry. Weights and ledger are plain values the loop
holds, with no random state. With no allocation configured, the default, there
are no weights and no book, and the loop is the one above.

**What is refused before anything runs.** A configuration whose eligible pool
could fall short of the replacement count, one that could replace more than its
fraction over any run of its window, or one that would make a trigger blind for
reasons of age alone; a minimum age or fitness window too short for the
deflator to assess an eligible agent on; and a population outside its band.
With the record's offers on, a population not described in the declared
vocabulary, a declaration handed to the loop that is not the population's own,
and a record that already holds an entry are refused too: the report's
descriptors are read under the declaration the population itself states. An
allocation without a portfolio-risk port, and a port without an allocation,
are refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from liq.evolution.ecology import descriptors as declared_axes
from liq.evolution.ecology import inheritance
from liq.evolution.ecology.config import (
    EcologyConfig,
    Inheritance,
    TurnoverConfig,
    most_unripe_at_a_trigger,
)
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.logctx import SILENT, RunLog
from liq.evolution.ecology.ports import (
    AgentPopulation,
    BarSource,
    ExecutionSimulator,
    FitnessDeflator,
    PortfolioRisk,
    RiskSizer,
)
from liq.evolution.ecology.types import (
    AgentId,
    ArchiveEntry,
    BarWindow,
    DeflatedFitness,
    FitnessSample,
    Genome,
    InstrumentId,
    LivingOccupancy,
    RecordCoverage,
    Standing,
    UtcTimestamp,
)

from .agent import PopulationState
from .agent_contracts import (
    AgentBirth,
    DeathKind,
    LearnedStateDisposition,
    Lineage,
    StartingStateOutsideItsBound,
)
from .population_accounts import (
    AllocationRecord,
    FixedShareAllocation,
    PopulationAccounts,
    PopulationEvaluation,
)

__all__ = [
    "BarTurnover",
    "Birth",
    "Candidate",
    "Death",
    "DeathKind",
    "Inheritance",
    "InitialCoverage",
    "PopulationBandBreached",
    "RecordOnlyTrigger",
    "RecordOnlyTurnoverReport",
    "SteadyState",
    "SteadyStateRun",
    "Trigger",
    "TurnoverConfig",
    "TurnoverReport",
    "choose_deaths",
]

#: What each seed drawn from the run's seed root is for, so a selection and a
#: spawn at the same trigger never share one.
_SELECTION = 0
_SPAWN = 1

#: Why a run reports no initial archive coverage: no trigger offered an entry
#: (``liq-docs/plans/oracle-v4-archive-interaction-2026-10-02.md`` §2.6).
NO_OFFERING_TRIGGER = "no agent was both assessed and described"


class PopulationBandBreached(EcologyError):
    """Births lost past their redraws left the population below its band.

    Raised rather than recorded: a run whose living count left its band is not
    a run of the configuration it reports.
    """


@dataclass(frozen=True)
class Candidate:
    """One living agent at a trigger, as the death rule reads it.

    Attributes:
        agent_id: Who.
        age: How many decision points it has lived.
        born_index: The decision point it was born at; founders at zero.
        admitted: Its place in the order agents entered the population.
        fitness: What the trigger's deflate call returned for it, or ``None``
            for an agent too young to have been handed to it.
    """

    agent_id: AgentId
    age: int
    born_index: int
    admitted: int
    fitness: DeflatedFitness | None


def choose_deaths(
    living: Sequence[Candidate], *, replacement_count: int, minimum_age: int
) -> tuple[tuple[Candidate, DeathKind], ...]:
    """Who dies at one trigger, and what decided it.

    Args:
        living: Every living agent at the trigger.
        replacement_count: How many eligible agents die, if that many exist.
        minimum_age: The fewest decision points an agent must have lived.

    Returns:
        The agents that die, in the order they die, each with its kind:
        culls first, then retirements, then protected retirements.
    """
    eligible = [each for each in living if each.age >= minimum_age]
    places = min(replacement_count, len(eligible))
    worse = sorted(
        (
            each
            for each in eligible
            if _figure(each) is not None and _standing(each) == "worse"
        ),
        key=lambda each: (_figure(each), each.born_index, each.admitted),
    )[:places]
    culled = {each.agent_id for each in worse}
    oldest_first = sorted(
        (each for each in eligible if each.agent_id not in culled),
        key=lambda each: (each.born_index, each.admitted),
    )
    silent = [each for each in oldest_first if _standing(each) != "better"]
    better = [each for each in oldest_first if _standing(each) == "better"]
    left = places - len(worse)
    retired = silent[:left]
    protected = better[: left - len(retired)]
    return (
        tuple((each, DeathKind.CULL) for each in worse)
        + tuple((each, DeathKind.RETIREMENT) for each in retired)
        + tuple((each, DeathKind.PROTECTED_RETIREMENT) for each in protected)
    )


def _figure(candidate: Candidate) -> float | None:
    return None if candidate.fitness is None else candidate.fitness.deflated


def _standing(candidate: Candidate) -> Standing:
    """An agent with no figure is told apart from nobody."""
    if candidate.fitness is None or candidate.fitness.deflated is None:
        return "indistinguishable"
    return candidate.fitness.standing


@dataclass(frozen=True)
class Death:
    """One agent that died at a trigger, and what decided it.

    Attributes:
        agent_id: Who died.
        kind: What decided it.
        standing: The deflator's verdict on it at the trigger.
        deflated: Its deflated figure, or ``None`` where it was not assessed.
        undeflated: Its undeflated figure, or ``None`` where the deflator had
            none for its window.
        age: How many decision points it had lived.
        died_at: The decision instant of the trigger (UTC).
    """

    agent_id: AgentId
    kind: DeathKind
    standing: Standing
    deflated: float | None
    undeflated: float | None
    age: int
    died_at: UtcTimestamp


@dataclass(frozen=True)
class Birth:
    """One agent born at a trigger, and how.

    Attributes:
        agent_id: Who was born.
        parents: Whom the population port's selection returned for it.
        born_at: The decision instant of the trigger (UTC).
        index: The decision point of the trigger.
        genome: Exactly what the port's ``spawn`` returned.
        mutation_scale: The scale in force for that spawn, as it reported it.
        boundary_firings: How often the boundary rule acted on each gene.
        learned_state_disposition: What crossed the birth.
        spawn_seed: The seed the genome was drawn under.
        blind: Whether its parents were drawn blind rather than on figures.
        attempts: How many draws it took to be admitted.
    """

    agent_id: AgentId
    parents: tuple[AgentId, ...]
    born_at: UtcTimestamp
    index: int
    genome: Genome
    mutation_scale: float | None
    boundary_firings: Mapping[str, int]
    learned_state_disposition: LearnedStateDisposition
    spawn_seed: int
    blind: bool
    attempts: int


@dataclass(frozen=True)
class Trigger:
    """What one trigger saw and did. Culls and retirements are never summed here.

    Attributes:
        as_of: The decision instant (UTC).
        index: The decision point.
        replacement_count: How many it was configured to replace.
        eligible: Living agents old enough to die.
        below_minimum_age: Living agents too young to die.
        worse: Eligible agents the deflation told apart as worse.
        indistinguishable: Eligible agents it assessed and told apart from nobody.
        better: Eligible agents it told apart as better.
        unassessed: Eligible agents it returned no figure for.
        deaths: How many died.
        culls: How many of those were culls.
        retirements: How many were retirements.
        protected_retirements: How many were protected retirements.
        candidates: Assessed agents still alive after the deaths.
        blind: Whether the trigger's parents were drawn blind.
        births: How many were born.
        admission_refusals: How many births the admission path refused here.
        births_lost: How many births were lost past their redraws.
    """

    as_of: UtcTimestamp
    index: int
    replacement_count: int
    eligible: int
    below_minimum_age: int
    worse: int
    indistinguishable: int
    better: int
    unassessed: int
    deaths: int
    culls: int
    retirements: int
    protected_retirements: int
    candidates: int
    blind: bool
    births: int
    admission_refusals: int
    births_lost: int


@dataclass(frozen=True)
class RecordOnlyTrigger(Trigger):
    """A trigger with the record's offers on: what it did, and what it offered and read.

    Every field of :class:`Trigger` means what it means there and holds what it
    would hold with the offers off; the rest is report only, read at one of two
    instants. The offer counts and ``living_occupancy`` are read at the offer
    instant, after the trigger's deflate call and before its first death, over
    the living then, and ``cumulative_coverage`` once the offers are made. The
    descriptor fields (``described``, ``undescribed``, ``descriptor_exclusions``,
    ``drift``, ``population_quantiles`` and ``raw_beta``) are one reading taken
    once the trigger's deaths and births are done, over the living then. So an
    agent that dies at the trigger is counted in its occupancy and absent from
    its reading, a newborn is absent from its occupancy and present in its
    reading, ``described`` and ``living_occupancy.described`` count two
    populations, and within-life drift never includes the interval in which an
    agent dies.

    Attributes:
        segment_id: The walk-forward segment the trigger falls in.
        offers: Entries offered to the record.
        offers_kept: Of those, how many the record kept when offered; an entry
            it evicts later, at this trigger or after, still counts.
        offers_excluded: Of those, how many it refused for a coordinate it
            cannot bin.
        not_offered_descriptor_undefined: Agents the deflation assessed and
            the population could not describe, so never offered.
        cumulative_coverage: The record's coverage after the offers, read from
            the record itself. Cumulative over the run: it counts every cell
            ever reached and is never evidence of the living's spread.
        living_occupancy: The cells the living sit in at the offer instant,
            read from the living and never from the record.
        described: Living agents with a descriptor at the trigger's reading.
        undescribed: Living agents short of the descriptor window.
        descriptor_exclusions: Per axis, per reason, how many described agents
            carry a value on it no binning accepts.
        drift: Within-life drift per behavioural axis since the previous
            trigger's reading, or ``None`` at the run's first reading.
        population_quantiles: Per behavioural axis, the described population's
            10th, 50th and 90th percentiles, or ``None`` for an axis nobody
            has a binnable value on.
        raw_beta: Each described agent's raw beta estimate, never the clipped one.
    """

    segment_id: str
    offers: int
    offers_kept: int
    offers_excluded: int
    not_offered_descriptor_undefined: int
    cumulative_coverage: RecordCoverage
    living_occupancy: LivingOccupancy
    described: int
    undescribed: int
    descriptor_exclusions: Mapping[str, Mapping[str, int]]
    drift: Mapping[str, declared_axes.Drift] | None
    population_quantiles: Mapping[str, tuple[float, float, float] | None]
    raw_beta: Mapping[AgentId, float]


@dataclass(frozen=True)
class InitialCoverage:
    """The record's coverage just after the first trigger that offered it an entry.

    Attributes:
        trigger: That trigger's row, carrying its index, instant, offers and
            the living occupancy beside the coverage; ``None`` if no trigger
            offered an entry.
        absent: Why there is no initial coverage, or ``None`` if there is one.
            An empty record is never reported as coverage of nothing.

    Raises:
        ValueError: Unless exactly one of the two is given.
    """

    trigger: RecordOnlyTrigger | None
    absent: str | None

    def __post_init__(self) -> None:
        if (self.trigger is None) == (self.absent is None):
            raise ValueError(
                "initial coverage is either a trigger's or absent with its reason"
            )

    @property
    def coverage(self) -> RecordCoverage | None:
        """The coverage itself, or ``None`` where it is absent."""
        return None if self.trigger is None else self.trigger.cumulative_coverage


@dataclass(frozen=True)
class _Offers:
    """What one trigger's offers came to, before its first death."""

    made: int
    kept: int
    excluded: int
    undefined: int
    occupancy: LivingOccupancy
    coverage: RecordCoverage


@dataclass(frozen=True)
class BarTurnover:
    """How many were born and died at one decision point."""

    as_of: UtcTimestamp
    births: int
    deaths: int


@dataclass(frozen=True)
class TurnoverReport:
    """What a run's turnover did, bar by bar and trigger by trigger.

    Attributes:
        triggers: Every trigger, in order.
        deaths: Every death, in order, each with its kind.
        births: Every birth, in order.
        per_bar: Births and deaths at every decision point walked.
        admission_refusals: How many births the admission path refused over
            the run.
        boundary_firings: Per gene, how often the boundary rule acted over
            every birth of the run.
        parents_per_birth: How many parents each birth was drawn from.
        inheritance: What crossed each birth.
    """

    triggers: tuple[Trigger, ...]
    deaths: tuple[Death, ...]
    births: tuple[Birth, ...]
    per_bar: tuple[BarTurnover, ...]
    admission_refusals: int
    boundary_firings: Mapping[str, int]
    parents_per_birth: int
    inheritance: Inheritance


@dataclass(frozen=True)
class RecordOnlyTurnoverReport(TurnoverReport):
    """A run's turnover with the record's offers on: the report, and its initial coverage.

    Every trigger is a :class:`RecordOnlyTrigger`.

    Attributes:
        initial_coverage: The record's coverage after its first offering
            trigger, or absent with the reason.
    """

    initial_coverage: InitialCoverage


@dataclass(frozen=True)
class SteadyStateRun:
    """One steady-state pass: every account, what turnover did, and what was allocated.

    Attributes:
        evaluation: Every account, the book's among them where one was kept.
        report: What turnover did.
        allocation: The weights and the book's ledger at every decision
            point, or ``None`` where no allocation was configured.
    """

    evaluation: PopulationEvaluation
    report: TurnoverReport
    allocation: AllocationRecord | None = None


class SteadyState:
    """Walk a population once, turning it over at every trigger.

    Args:
        population: The population port: selection, spawn, learned state.
        state: The population itself, admitted into and departed from.
        deflator: The fitness-deflation port, the loop's only verdict source.
        config: The pass, including its turnover configuration.
        instrument: The single instrument every account follows.
        sizer: The mandate.
        simulator: The execution model, under its named cost scenario.
        opening_equity: What every account opens with.
        log: Where accounts write what they did.
        descriptor_declaration: Accepted so a caller that hands the loop its
            population's declaration stays valid, and never read in its place.
            With the record's offers on, the drift and raw beta the record's
            report carries are read under the population's own
            ``descriptor_declaration``; one handed here is only checked against
            it, and refused unless equal, because a second copy cannot
            disagree with the one the population describes under.
        portfolio_risk: The port the aggregate book is bounded by. Required
            where the configuration names an allocation, and refused where it
            names none.

    Raises:
        ValueError: If the configuration or the population is one the records
            refuse; see the module docstring.
    """

    def __init__(
        self,
        *,
        population: AgentPopulation,
        state: PopulationState,
        deflator: FitnessDeflator,
        config: EcologyConfig,
        instrument: InstrumentId,
        sizer: RiskSizer,
        simulator: ExecutionSimulator,
        opening_equity: float,
        log: RunLog = SILENT,
        descriptor_declaration: declared_axes.DescriptorDeclaration | None = None,
        portfolio_risk: PortfolioRisk | None = None,
    ) -> None:
        if config.turnover is None:
            raise ValueError(
                "a steady-state pass needs a turnover configuration; one without "
                "it births and kills nobody and is a plain evaluation"
            )
        self.turnover: TurnoverConfig = config.turnover
        self.config = config
        self.population = population
        self.state = state
        self.deflator = deflator
        self._refuse_what_the_records_refuse()
        self._offers_on = config.diversity_record == "record_only"
        self._declaration = (
            self._refuse_an_unrecordable_population(descriptor_declaration)
            if self._offers_on
            else None
        )
        self._last_reading: declared_axes.Reading | None = None
        self._first_offering: RecordOnlyTrigger | None = None
        self.accounts = PopulationAccounts(
            state,
            instrument=instrument,
            sizer=sizer,
            simulator=simulator,
            opening_equity=opening_equity,
            log=log,
        )
        self.allocation = self._allocation(portfolio_risk)
        living = state.agent_ids()
        self._born_index: dict[AgentId, int] = dict.fromkeys(living, 0)
        self._admission_order: dict[AgentId, int] = {
            agent_id: order for order, agent_id in enumerate(living)
        }
        self._index = -1
        self._refusals_at_start = state.admission_refusals()
        self._refused_here = 0
        self._triggers: list[Trigger] = []
        self._deaths: list[Death] = []
        self._births: list[Birth] = []
        self._per_bar: list[BarTurnover] = []
        self._firings: dict[str, int] = dict.fromkeys(state.gene_names, 0)

    def _refuse_what_the_records_refuse(self) -> None:
        turnover, state = self.turnover, self.state
        if state.learning is None:
            raise ValueError(
                "a population that does not learn has no outcomes to deflate and "
                "cannot be turned over on evidence"
            )
        if set(self.population.agent_ids()) != set(state.agent_ids()):
            raise ValueError(
                "the population port and the population store must hold the same "
                "living agents; two views of one population cannot disagree"
            )
        living = len(state.agent_ids())
        if not turnover.band_lower <= living <= turnover.band_upper:
            raise ValueError(
                f"{living} founders lie outside the band "
                f"[{turnover.band_lower}, {turnover.band_upper}]"
            )
        if turnover.fitness_window > state.history_capacity:
            raise ValueError(
                f"a fitness window of {turnover.fitness_window} outcomes is longer "
                f"than the {state.history_capacity} each agent's history keeps"
            )
        shortest = min(turnover.minimum_age, turnover.fitness_window)
        if shortest < self.deflator.minimum_window:
            raise ValueError(
                f"an eligible agent can hand the deflator {shortest} outcomes, "
                f"below its minimum window of {self.deflator.minimum_window}"
            )
        unripe = most_unripe_at_a_trigger(
            turnover.replacement_count, turnover.trigger_period, turnover.minimum_age
        )
        floor = self.population.minimum_candidates
        if turnover.band_lower - unripe < floor:
            raise ValueError(
                f"after a trigger's deaths as few as {turnover.band_lower - unripe} "
                f"eligible agents can survive, below the {floor} the population's "
                "selection can rank, so triggers would be blind by age alone"
            )

    def _allocation(
        self, portfolio_risk: PortfolioRisk | None
    ) -> FixedShareAllocation | None:
        """The allocation hook the configuration names, or ``None`` where it names none.

        Raises:
            ValueError: If an allocation is configured with no portfolio-risk
                port, or a port is handed with no allocation configured.
        """
        configured = self.config.allocation
        if configured is None:
            if portfolio_risk is not None:
                raise ValueError(
                    "a portfolio-risk port was handed to a loop with no allocation "
                    "configured; there is no book for it to bound"
                )
            return None
        if portfolio_risk is None:
            raise ValueError(
                "an allocation is configured but no portfolio-risk port was "
                "handed; the aggregate book's bounds are the port's"
            )
        return FixedShareAllocation(
            self.accounts,
            aggregator=configured.aggregator(),
            risk=portfolio_risk,
            band=configured.band,
        )

    def _refuse_an_unrecordable_population(
        self, handed: declared_axes.DescriptorDeclaration | None
    ) -> declared_axes.DescriptorDeclaration:
        """The population's own declaration, once its record can be offered to.

        Raises:
            ValueError: If the population is not described in the declared
                vocabulary, if a declaration handed to the loop is not equal
                to the population's own, or if the record already holds an
                entry before the run's first trigger.
        """
        population = self.population
        declaration = population.descriptor_declaration
        if declaration is None:
            raise ValueError(
                'diversity_record = "record_only" reports drift and raw beta at '
                "every trigger, read under the population's own descriptor "
                "declaration; this population states none, so it is not "
                f"described in the declared vocabulary {declared_axes.VOCABULARY!r}"
            )
        if handed is not None and handed != declaration:
            raise ValueError(
                "the descriptor declaration handed to the loop is not the "
                "population's own; the report is read under the population's, "
                "and a second copy cannot disagree with it"
            )
        held = len(population.recorded())
        if held:
            raise ValueError(
                f"the population's record holds {held} entr"
                f"{'y' if held == 1 else 'ies'} before the run's first trigger; a "
                "run's record starts empty and never carries from another run"
            )
        return declaration

    def run(self, source: BarSource) -> SteadyStateRun:
        """Walk ``source`` once, turning the population over at every trigger."""
        evaluation = self.accounts.run(
            source, self.config, turnover=self._turn_over, allocation=self.allocation
        )
        allocated = None if self.allocation is None else self.allocation.record()
        report = TurnoverReport(
            triggers=tuple(self._triggers),
            deaths=tuple(self._deaths),
            births=tuple(self._births),
            per_bar=tuple(self._per_bar),
            admission_refusals=self.state.admission_refusals()
            - self._refusals_at_start,
            boundary_firings=MappingProxyType(dict(self._firings)),
            parents_per_birth=self.turnover.parents_per_birth,
            inheritance=self.turnover.inheritance,
        )
        if not self._offers_on:
            return SteadyStateRun(
                evaluation=evaluation, report=report, allocation=allocated
            )
        first = self._first_offering
        return SteadyStateRun(
            evaluation=evaluation,
            allocation=allocated,
            report=RecordOnlyTurnoverReport(
                **vars(report),
                initial_coverage=InitialCoverage(
                    trigger=first,
                    absent=NO_OFFERING_TRIGGER if first is None else None,
                ),
            ),
        )

    def _fires(self, index: int) -> bool:
        """Whether a trigger falls at ``index``: bar index and configuration only."""
        return self.turnover.fires(index)

    def _turn_over(self, window: BarWindow) -> None:
        """Turn the population over if a trigger falls at this decision point."""
        self._index += 1
        index = self._index
        if not self._fires(index):
            self._per_bar.append(BarTurnover(as_of=window.as_of, births=0, deaths=0))
            return
        self._trigger(window, index)

    def _trigger(self, window: BarWindow, index: int) -> None:
        turnover, state = self.turnover, self.state
        living = state.agent_ids()
        ages = {agent_id: index - self._born_index[agent_id] for agent_id in living}
        eligible = [
            agent_id for agent_id in living if ages[agent_id] >= turnover.minimum_age
        ]
        verdicts = (
            self.deflator.deflate([self._sample(agent_id) for agent_id in eligible])
            if eligible
            else ()
        )
        by_id = {verdict.agent_id: verdict for verdict in verdicts}
        offered = self._offer(window, verdicts) if self._offers_on else None
        chosen = choose_deaths(
            [
                Candidate(
                    agent_id=agent_id,
                    age=ages[agent_id],
                    born_index=self._born_index[agent_id],
                    admitted=self._admission_order[agent_id],
                    fitness=by_id.get(agent_id),
                )
                for agent_id in living
            ],
            replacement_count=turnover.replacement_count,
            minimum_age=turnover.minimum_age,
        )
        for candidate, kind in chosen:
            self._die(candidate, kind, window)
        survivors = state.agent_ids()
        chunks, blind = self._parents(index, survivors, by_id, len(chosen))
        self._refused_here = 0
        born = sum(
            self._born(window, ordinal, chunks, blind, index) is not None
            for ordinal in range(len(chunks))
        )
        self._hold_the_band(window)
        kinds = [kind for _, kind in chosen]
        assessed = [verdict for verdict in verdicts if verdict.deflated is not None]
        trigger = Trigger(
            as_of=window.as_of,
            index=index,
            replacement_count=turnover.replacement_count,
            eligible=len(eligible),
            below_minimum_age=len(living) - len(eligible),
            worse=sum(verdict.standing == "worse" for verdict in assessed),
            indistinguishable=sum(
                verdict.standing == "indistinguishable" for verdict in assessed
            ),
            better=sum(verdict.standing == "better" for verdict in assessed),
            unassessed=len(verdicts) - len(assessed),
            deaths=len(chosen),
            culls=kinds.count(DeathKind.CULL),
            retirements=kinds.count(DeathKind.RETIREMENT),
            protected_retirements=kinds.count(DeathKind.PROTECTED_RETIREMENT),
            candidates=sum(
                by_id[agent_id].deflated is not None
                for agent_id in survivors
                if agent_id in by_id
            ),
            blind=blind,
            births=born,
            admission_refusals=self._refused_here,
            births_lost=len(chunks) - born,
        )
        self._triggers.append(
            trigger if offered is None else self._recorded(trigger, window, offered)
        )
        self._per_bar.append(
            BarTurnover(as_of=window.as_of, births=born, deaths=len(chosen))
        )

    def _offer(self, window: BarWindow, verdicts: Sequence[DeflatedFitness]) -> _Offers:
        """Offer the record each agent ``verdicts`` assessed and the population describes.

        Called after the trigger's one deflate call and before its first
        death. Living occupancy is read here, over the living at that instant,
        and the record's coverage once the offers are made. What the record
        keeps or excludes is counted and never acted on.
        """
        population = self.population
        occupancy = population.occupancy()
        excluded_before = len(population.excluded())
        made = kept = undefined = 0
        for verdict in verdicts:
            if verdict.deflated is None:
                continue
            try:
                descriptor = population.descriptor(verdict.agent_id)
            except declared_axes.DescriptorUnavailable:
                undefined += 1
                continue
            made += 1
            kept += population.record(
                ArchiveEntry(
                    agent_id=verdict.agent_id,
                    genome=population.genome(verdict.agent_id),
                    descriptor=descriptor,
                    objectives=MappingProxyType({"deflated": verdict.deflated}),
                    recorded_at=window.as_of,
                )
            )
        return _Offers(
            made=made,
            kept=kept,
            excluded=len(population.excluded()) - excluded_before,
            undefined=undefined,
            occupancy=occupancy,
            coverage=population.coverage(),
        )

    def _recorded(
        self, trigger: Trigger, window: BarWindow, offered: _Offers
    ) -> RecordOnlyTrigger:
        """The trigger's row, with what it offered and the descriptors read at it."""
        assert self._declaration is not None
        reading = declared_axes.read(self.state, self._declaration)
        earlier, self._last_reading = self._last_reading, reading
        exclusions: dict[str, dict[str, int]] = {}
        for described in reading.described.values():
            for axis, reason in described.excluded.items():
                counts = exclusions.setdefault(axis, {})
                counts[reason] = counts.get(reason, 0) + 1
        recorded = RecordOnlyTrigger(
            **vars(trigger),
            segment_id=window.segment_id,
            offers=offered.made,
            offers_kept=offered.kept,
            offers_excluded=offered.excluded,
            not_offered_descriptor_undefined=offered.undefined,
            cumulative_coverage=offered.coverage,
            living_occupancy=offered.occupancy,
            described=len(reading.described),
            undescribed=len(reading.undescribed),
            descriptor_exclusions=MappingProxyType(
                {axis: MappingProxyType(counts) for axis, counts in exclusions.items()}
            ),
            drift=None if earlier is None else declared_axes.drift(earlier, reading),
            population_quantiles=declared_axes.marginal_quantiles(reading),
            raw_beta=MappingProxyType(
                {
                    agent_id: described.raw[declared_axes.BETA]
                    for agent_id, described in reading.described.items()
                }
            ),
        )
        if offered.made and self._first_offering is None:
            self._first_offering = recorded
        return recorded

    def _sample(self, agent_id: AgentId) -> FitnessSample:
        """An agent's own most recent outcomes, as many as the window holds."""
        outcomes = self.state.history(agent_id)[-self.turnover.fitness_window :]
        return FitnessSample(
            agent_id=agent_id, outcomes=outcomes, independent_outcomes=len(outcomes)
        )

    def _die(self, candidate: Candidate, kind: DeathKind, window: BarWindow) -> None:
        """The one exit: depart the population, close the account, record the label."""
        self.state.depart(candidate.agent_id, died_at=window.as_of, kind=kind)
        self.accounts.close(candidate.agent_id, as_of=window.as_of)
        fitness = candidate.fitness
        self._deaths.append(
            Death(
                agent_id=candidate.agent_id,
                kind=kind,
                standing=_standing(candidate),
                deflated=None if fitness is None else fitness.deflated,
                undeflated=None if fitness is None else fitness.undeflated,
                age=candidate.age,
                died_at=window.as_of,
            )
        )

    def _parents(
        self,
        index: int,
        survivors: Sequence[AgentId],
        by_id: Mapping[AgentId, DeflatedFitness],
        births: int,
    ) -> tuple[list[list[tuple[AgentId, ...]]], bool]:
        """Every birth's parents, for every draw it may need, from one selection.

        Returns the parents per birth and per draw, and whether the trigger was
        blind. One call to the port's selection covers every draw a trigger can
        make, so the figures handed to it are keyed by exactly the population
        that is alive before anyone is born.
        """
        figures: dict[AgentId, float | None] = {
            agent_id: (by_id[agent_id].deflated if agent_id in by_id else None)
            for agent_id in survivors
        }
        candidates = sum(figure is not None for figure in figures.values())
        blind = candidates < self.population.minimum_candidates
        if blind:
            figures = dict.fromkeys(survivors, None)
        if births == 0:
            return [], blind
        width = self.turnover.parents_per_birth
        draws = self.turnover.admission_redraws + 1
        chosen = self.population.select_parents(
            fitness=MappingProxyType(figures),
            count=births * draws * width,
            seed=self._seed(index, _SELECTION),
        )
        return [
            [
                tuple(chosen[(ordinal * draws + draw) * width :][:width])
                for draw in range(draws)
            ]
            for ordinal in range(births)
        ], blind

    def _born(
        self,
        window: BarWindow,
        ordinal: int,
        chunks: Sequence[Sequence[tuple[AgentId, ...]]],
        blind: bool,
        index: int,
    ) -> Lineage | None:
        """One birth, drawn again from fresh parents until admitted or out of draws."""
        del index
        for attempt, parents in enumerate(chunks[ordinal]):
            lineage = self._admit_attempt(window, ordinal, parents, blind, attempt)
            if lineage is not None:
                return lineage
        return None

    def _admit_attempt(
        self,
        window: BarWindow,
        ordinal: int,
        parents: tuple[AgentId, ...],
        blind: bool,
        attempt: int,
    ) -> Lineage | None:
        """Spawn from ``parents`` and offer the offspring to the admission path."""
        index = self._index
        seed = self._seed(index, _SPAWN, ordinal, attempt)
        offspring = self.population.spawn(parents=parents, seed=seed)
        birth = inheritance.birth_of(
            agent_id=f"born-{index:06d}-{ordinal:03d}",
            offspring=offspring,
            parents=parents,
            population=self.population,
            versions=self.state.versions(parents[0]),
            inheritance=self.turnover.inheritance,
            born_at=window.as_of,
            spawn_seed=seed,
        )
        try:
            lineage = self._admitted(window, birth, offspring, blind, attempt)
        except StartingStateOutsideItsBound:
            self._refused_here += 1
            return None
        for gene, count in offspring.boundary_firings.items():
            self._firings[gene] = self._firings.get(gene, 0) + count
        return lineage

    def _admitted(
        self,
        window: BarWindow,
        birth: AgentBirth,
        offspring: object,
        blind: bool,
        attempt: int,
    ) -> Lineage:
        """Admit ``birth`` through the population's one path, and open its account."""
        lineage = self.state.admit(birth)
        self.accounts.open(birth.agent_id, as_of=window.as_of)
        self._born_index[birth.agent_id] = self._index
        self._admission_order[birth.agent_id] = len(self._admission_order)
        firings = getattr(offspring, "boundary_firings", MappingProxyType({}))
        self._births.append(
            Birth(
                agent_id=birth.agent_id,
                parents=birth.parents,
                born_at=window.as_of,
                index=self._index,
                genome=birth.genome,
                mutation_scale=birth.mutation_scale,
                boundary_firings=MappingProxyType(dict(firings)),
                learned_state_disposition=lineage.learned_state_disposition,
                spawn_seed=birth.spawn_seed if birth.spawn_seed is not None else 0,
                blind=blind,
                attempts=attempt + 1,
            )
        )
        return lineage

    def _hold_the_band(self, window: BarWindow) -> None:
        """Stop a run whose lost births left it below its band."""
        living = len(self.state.agent_ids())
        if living < self.turnover.band_lower:
            raise PopulationBandBreached(
                f"at {window.as_of.isoformat()} {living} agents are alive, below "
                f"the band's lower bound {self.turnover.band_lower}: births were "
                "refused past every redraw"
            )

    def _seed(self, *path: int) -> int:
        """A seed drawn from the run's seed root along ``path``, reproducibly."""
        return int(
            np.random.SeedSequence([self.turnover.seed, *path]).generate_state(1)[0]
        )
