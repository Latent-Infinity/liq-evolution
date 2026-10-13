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

*The diversity record.* The loop offers nothing to the population's diversity
record and reads nothing from it: no birth, death, cull or retirement inserts,
evicts or consults an entry, so in this loop the interaction between the record
and a birth or death is none
(``liq-docs/plans/oracle-v3-archive-offers-2026-10-02.md``). Who offers an entry,
when and with which figure is decided where the record first acquires a role,
with the niching descriptors; no figure this loop holds is offered in the
meantime, because the deflated one is relative to its own trigger's comparison
and may be absent, and the undeflated one is for reporting.

**What is refused before anything runs.** A configuration whose eligible pool
could fall short of the replacement count, one that could replace more than its
fraction over any run of its window, or one that would make a trigger blind for
reasons of age alone; a minimum age or fitness window too short for the
deflator to assess an eligible agent on; and a population outside its band.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

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
    RiskSizer,
)
from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    DeflatedFitness,
    FitnessSample,
    Genome,
    InstrumentId,
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
from .population_accounts import PopulationAccounts, PopulationEvaluation

__all__ = [
    "BarTurnover",
    "Birth",
    "Candidate",
    "Death",
    "DeathKind",
    "Inheritance",
    "PopulationBandBreached",
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
class SteadyStateRun:
    """One steady-state pass: every account, and what turnover did."""

    evaluation: PopulationEvaluation
    report: TurnoverReport


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
        self.accounts = PopulationAccounts(
            state,
            instrument=instrument,
            sizer=sizer,
            simulator=simulator,
            opening_equity=opening_equity,
            log=log,
        )
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

    def run(self, source: BarSource) -> SteadyStateRun:
        """Walk ``source`` once, turning the population over at every trigger."""
        evaluation = self.accounts.run(source, self.config, turnover=self._turn_over)
        return SteadyStateRun(
            evaluation=evaluation,
            report=TurnoverReport(
                triggers=tuple(self._triggers),
                deaths=tuple(self._deaths),
                births=tuple(self._births),
                per_bar=tuple(self._per_bar),
                admission_refusals=self.state.admission_refusals()
                - self._refusals_at_start,
                boundary_firings=MappingProxyType(dict(self._firings)),
                parents_per_birth=self.turnover.parents_per_birth,
                inheritance=self.turnover.inheritance,
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
        self._triggers.append(
            Trigger(
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
        )
        self._per_bar.append(
            BarTurnover(as_of=window.as_of, births=born, deaths=len(chosen))
        )

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
