"""Declared arithmetic streams a turning-over population walks, and how it is watched.

**Nothing here is market data and nothing here may be read as market data.** Three
streams are declared, the three the plan's operator ablation names:

* **S1** is the level-shift stream of ``tests/support/level_shift_stream.py`` —
  one exciting feature cycling through a short declared period, and a relationship
  that steps once — with each bar's close moving by a declared fraction of that
  stream's outcome, so an evaluation account over it earns what the relationship
  says. The bars are rebuilt from that arithmetic and nothing else.
* **S3** is S1 with the move's sign flipped.
* **S4** is the :class:`~liq.evolution.ecology.adapters.NullBarSource` ramp, with
  its own two-feature view.

Every number a check computes over these is a number about the arithmetic.

**How a population is watched.** A turning-over population is observed from
outside the loop that turns it over, never from the loop's own report: the
:class:`Watched` source reads the population's living identities once at
founding and once after each decision point has been handed over — which is
after that bar's deaths and births are complete — so births and deaths are set
differences between consecutive observations. A loop that under-reports cannot
make the observation under-count.

**A refusal, seeded.** Parents inside their declared ranges never produce an
offspring the admission path refuses, so a check that needs a refusal seeds one:
:class:`Recording` can hand the real admission path, at declared spawns, the real
offspring with its ``weight.`` genes pushed past the weight-norm bound. The
refusal is the admission path's own and is counted by its own counter, and the
offspring built to be refused is never admitted.

**The deflator doubles.** C-06 test doubles whose verdicts are a declared
function of the windows they are handed. They stand in for the port; they are
not a model of any deflation, and their arithmetic is the test's declaration of
what the double says, not a statistic about anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from liq.evolution.ecology import Genome, agent, genome_schema
from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.ports import refuse_an_undeflatable_sample
from liq.evolution.ecology.types import (
    AgentId,
    Bar,
    BarWindow,
    DeflatedFitness,
    FitnessSample,
    Standing,
)
from tests.support import level_shift_stream as s1

#: The instrument every stream here prices.
INSTRUMENT = s1.INSTRUMENT

#: The feature S1 and S3 broadcast.
FEATURE = s1.FEATURE

#: How much of the level-shift stream's outcome one bar's close moves by. A
#: fraction of the close, so the declared outcome of a few units becomes a move
#: of a few tenths of a percent; it is a declared scale, not a price.
MOVE_PER_UNIT_OF_OUTCOME = 1.0e-3

#: The close S1 and S3 start from. A declared anchor, not a price.
FIRST_CLOSE = 100.0

#: The volume every rebuilt bar carries. A declared placeholder.
VOLUME = 1.0

#: S4's vocabulary: the ramp's own feature names and schema.
RAMP_FEATURES = ("level", "step")
RAMP_SCHEMA_VERSION = "null-features-1"

#: The forgetting factors founders cycle through: the corners and middle of the
#: declared range the weight-gene ADR measured on.
FORGETTING = (0.9, 0.99, 0.9999)

#: The priors founders cycle through. Small and of both signs, so founders
#: differ in what they wish from the first decision point on.
PRIORS = (-0.4, -0.2, 0.0, 0.2, 0.4)


def priced(sign: int = 1, count: int = s1.DECISION_POINTS) -> tuple[BarWindow, ...]:
    """S1 (``sign`` +1) or S3 (``sign`` -1): the level-shift stream with priced bars.

    The bar a decision point at index ``i`` completes has its close moved from
    the previous one by ``sign`` times the declared fraction of the outcome the
    level-shift stream realises over that bar — the outcome of decision point
    ``i - 1``. Each bar opens at the previous close, so a target acted on at an
    open is carried over the whole of the bar's move and earns what the
    relationship says, and nothing is earned over a gap.
    """
    base = s1.windows(count)
    windows: list[BarWindow] = []
    close = FIRST_CLOSE
    for index, window in enumerate(base):
        opened = close
        if index > 0:
            close = opened * (
                1.0 + sign * MOVE_PER_UNIT_OF_OUTCOME * s1.outcome_at(index - 1)
            )
        ramp = window.bars[INSTRUMENT]
        bar = Bar(
            instrument=INSTRUMENT,
            period_start=ramp.period_start,
            period_end=ramp.period_end,
            open=opened,
            high=max(opened, close),
            low=min(opened, close),
            close=close,
            volume=VOLUME,
        )
        windows.append(replace(window, bars=MappingProxyType({INSTRUMENT: bar})))
    return tuple(windows)


def ramp(count: int = s1.DECISION_POINTS) -> tuple[BarWindow, ...]:
    """S4: the null ramp with its own two-feature view, segment boundary as S1's."""
    return tuple(
        NullBarSource(
            instrument=INSTRUMENT,
            window_count=count,
            segment_length=s1.BREAK_AT,
            feature_schema_version=RAMP_SCHEMA_VERSION,
        ).windows()
    )


def founder(
    index: int,
    *,
    features: Sequence[str] = (FEATURE,),
    priors: Sequence[float] | None = None,
    feature_schema_version: str = s1.FEATURE_SCHEMA_VERSION,
) -> agent.AgentBirth:
    """Founder ``index``: genes drawn from the declared grids by its index alone."""
    weights = (
        tuple(priors)
        if priors is not None
        else tuple(
            PRIORS[(index + offset) % len(PRIORS)] for offset in range(len(features))
        )
    )
    genes: dict[str, float] = {}
    for name, weight in zip(features, weights, strict=True):
        genes[f"{agent.MASK_PREFIX}{name}"] = 1.0
        genes[f"{agent.WEIGHT_PREFIX}{name}"] = weight
    genes[agent.ENTRY_THRESHOLD] = s1.ENTRY_AT
    genes[agent.FORGETTING_FACTOR] = FORGETTING[index % len(FORGETTING)]
    return agent.AgentBirth(
        agent_id=f"founder-{index:03d}",
        genome=Genome(
            genes=MappingProxyType(genes),
            schema_version=genome_schema.SCHEMA_VERSION,
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=feature_schema_version,
        model_version=s1.MODEL_VERSION,
    )


def founders(count: int) -> tuple[agent.AgentBirth, ...]:
    """``count`` founders for S1 and S3."""
    return tuple(founder(index) for index in range(count))


def ramp_founders(count: int) -> tuple[agent.AgentBirth, ...]:
    """``count`` founders for S4, weighting the ramp's own features."""
    return tuple(
        founder(
            index, features=RAMP_FEATURES, feature_schema_version=RAMP_SCHEMA_VERSION
        )
        for index in range(count)
    )


@dataclass
class Watched:
    """A bar source that watches the population between decision points.

    Attributes:
        stream: The decision points handed over, in order.
        state: The population watched. Read only through its own accessors.
        look: Called with the index of the decision point just handed over —
            or ``-1`` at founding — after the living have been recorded, for a
            check that wants to read more of the population at that moment.
        living: The living identities at founding, then after each decision
            point's deaths and births are complete. Filled by walking.
    """

    stream: Sequence[BarWindow]
    state: agent.PopulationState
    look: Callable[[int], None] | None = None
    living: list[frozenset[AgentId]] = field(default_factory=list)

    def windows(self) -> Iterator[BarWindow]:
        """Yield each decision point, recording who is alive between them."""
        self._watch(-1)
        for index, window in enumerate(self.stream):
            yield window
            self._watch(index)

    def _watch(self, index: int) -> None:
        self.living.append(frozenset(self.state.agent_ids()))
        if self.look is not None:
            self.look(index)

    def births(self, index: int) -> frozenset[AgentId]:
        """Who was born at decision point ``index``, as observed."""
        return self.living[index + 1] - self.living[index]

    def deaths(self, index: int) -> frozenset[AgentId]:
        """Who died at decision point ``index``, as observed."""
        return self.living[index] - self.living[index + 1]

    def born_at(self) -> dict[AgentId, int]:
        """The decision point each agent was first seen alive after; founders at 0."""
        born: dict[AgentId, int] = dict.fromkeys(self.living[0], 0)
        for index in range(len(self.living) - 1):
            for agent_id in self.births(index):
                born[agent_id] = index
        return born

    def died_at(self) -> dict[AgentId, int]:
        """The decision point each departed agent died at, as observed."""
        return {
            agent_id: index
            for index in range(len(self.living) - 1)
            for agent_id in self.deaths(index)
        }


#: A double's declaration: given every sample handed to one call, the figure and
#: standing for each, in order, or ``None`` for an agent it declines to assess.
Verdicts = Callable[[Sequence[FitnessSample]], Sequence[tuple[float, Standing] | None]]


@dataclass
class ScriptedDeflator:
    """A C-06 double whose verdicts are a declared rule of the windows it is handed.

    Attributes:
        verdicts: The declaration. ``None`` for an agent means "not assessed".
        minimum_window: The fewest outcomes this double assesses on.
        calls: Every call's results, in order, so a check can count calls and
            see which agents each call was handed.
    """

    verdicts: Verdicts
    minimum_window: int = 2
    calls: list[tuple[DeflatedFitness, ...]] = field(default_factory=list)

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        """Apply the declaration to the samples, unassessed where it declines."""
        refuse_an_undeflatable_sample(samples)
        called = tuple(
            _result(sample, verdict, len(samples))
            for sample, verdict in zip(samples, self.verdicts(samples), strict=True)
        )
        self.calls.append(called)
        return called


def _result(
    sample: FitnessSample, verdict: tuple[float, Standing] | None, compared: int
) -> DeflatedFitness:
    if verdict is None:
        return DeflatedFitness(
            agent_id=sample.agent_id,
            undeflated=None,  # ty: ignore[invalid-argument-type]
            deflated=None,  # ty: ignore[invalid-argument-type]
            compared=compared,
            independent_outcomes=sample.independent_outcomes,
            standing="indistinguishable",
        )
    figure, standing = verdict
    return DeflatedFitness(
        agent_id=sample.agent_id,
        undeflated=math.fsum(sample.outcomes),
        deflated=figure,
        compared=compared,
        independent_outcomes=sample.independent_outcomes,
        standing=standing,
    )


def per_sample(
    rule: Callable[[FitnessSample], tuple[float, Standing] | None],
) -> Verdicts:
    """A declaration that judges each sample on its own window alone."""
    return lambda samples: [rule(sample) for sample in samples]


def negative_is_worse(sample: FitnessSample) -> tuple[float, Standing]:
    """Worse where the window lost on balance; the figure is the window's sum."""
    total = math.fsum(sample.outcomes)
    return total, ("worse" if total < 0.0 else "indistinguishable")


def lowest_is_worse(samples: Sequence[FitnessSample]) -> list[tuple[float, Standing]]:
    """The single lowest window sum is worse; everyone else indistinguishable."""
    sums = [math.fsum(each.outcomes) for each in samples]
    lowest = min(range(len(samples)), key=lambda at: (sums[at], samples[at].agent_id))
    return [
        (total, "worse" if at == lowest else "indistinguishable")
        for at, total in enumerate(sums)
    ]


#: The descriptor vocabulary every population built here is stamped with.
DESCRIPTOR_SCHEMA_VERSION = "turnover-descriptors-1"

#: How many outcomes each agent's history keeps in the loops built here: more
#: than any fitness window a check asks for.
HISTORY_CAPACITY = 64


@dataclass
class Recording:
    """A population port that records what the loop asked of it, then answers as the real one.

    Every call is passed straight to the wrapped adapter; what is recorded is
    what the port returned, so a check can trace a birth to the selection and
    the spawn that produced it without reading the loop's own report.

    **A seeded refusal.** At each spawn whose place in the run's order of
    spawns is in ``spawns``, the real offspring is handed back with every
    ``weight.`` gene set to twice ``weight_norm_bound``, so its starting
    weights' norm is past that bound whatever the number of features, and the
    real admission path refuses it. Nothing else of the offspring changes, and
    it is never admitted. With no spawn named, every answer is the real one.

    Attributes:
        inner: The real adapter.
        spawns: The places, counted from zero, of the spawns to push.
        weight_norm_bound: The bound the pushed weights are set past.
        asked: How many spawns the loop has asked for.
        pushed: The places of the spawns actually pushed, in order.
    """

    inner: object
    spawns: frozenset[int] = frozenset()
    weight_norm_bound: float = 0.0
    asked: int = 0
    pushed: list[int] = field(default_factory=list)
    selected: list[tuple[AgentId, ...]] = field(default_factory=list)
    handed: list[dict[AgentId, float | None]] = field(default_factory=list)
    spawned: list[tuple[tuple[AgentId, ...], int, object]] = field(default_factory=list)

    @property
    def minimum_candidates(self) -> int:
        return self.inner.minimum_candidates  # ty: ignore[unresolved-attribute]

    def agent_ids(self):
        return self.inner.agent_ids()  # ty: ignore[unresolved-attribute]

    def genome(self, agent_id):
        return self.inner.genome(agent_id)  # ty: ignore[unresolved-attribute]

    def learned_state(self, agent_id):
        return self.inner.learned_state(agent_id)  # ty: ignore[unresolved-attribute]

    def descriptor(self, agent_id):
        return self.inner.descriptor(agent_id)  # ty: ignore[unresolved-attribute]

    def record(self, entry):
        return self.inner.record(entry)  # ty: ignore[unresolved-attribute]

    def recorded(self):
        return self.inner.recorded()  # ty: ignore[unresolved-attribute]

    def spawn(self, *, parents, seed):
        offspring = self.inner.spawn(parents=parents, seed=seed)  # ty: ignore[unresolved-attribute]
        place, self.asked = self.asked, self.asked + 1
        if place in self.spawns:
            genes = {
                name: (
                    2.0 * self.weight_norm_bound
                    if name.startswith(agent.WEIGHT_PREFIX)
                    else value
                )
                for name, value in offspring.genome.genes.items()
            }
            offspring = replace(
                offspring,
                genome=replace(offspring.genome, genes=MappingProxyType(genes)),
            )
            self.pushed.append(place)
        self.spawned.append((tuple(parents), seed, offspring))
        return offspring

    def select_parents(self, *, fitness, count, seed):
        self.handed.append(dict(fitness))
        chosen = self.inner.select_parents(fitness=fitness, count=count, seed=seed)  # ty: ignore[unresolved-attribute]
        self.selected.append(tuple(chosen))
        return chosen


#: The spawns a refusal is seeded at when a check needs one: the first a run
#: asks for, counted from zero in the order the loop asks.
FIRST_SPAWN = frozenset({0})


@dataclass
class Wired:
    """A steady-state loop wired over a founded population, not yet walked."""

    state: agent.PopulationState
    port: Recording
    loop: object


@dataclass
class Steady:
    """One steady-state run over a declared stream, and what watched it."""

    state: agent.PopulationState
    port: Recording
    watched: Watched
    loop: object
    run: object


def wire(
    *,
    founding: Sequence[agent.AgentBirth],
    turnover: object,
    deflator: object,
    learning: LearningConfig | None = None,
    simulator: object | None = None,
    run_id: str = "steady-state",
    refused_spawns: frozenset[int] = frozenset(),
) -> Wired:
    """Found ``founding`` and wire the real population adapter behind a recording port.

    ``refused_spawns`` names the spawns, counted from zero, whose offspring the
    admission path is handed pushed past the weight-norm bound; none by default.
    """
    from liq.evolution import ecology
    from liq.evolution.ecology.adapters import (
        ArrayGenomePopulation,
        NullExecutionSimulator,
        NullRiskSizer,
    )
    from liq.evolution.ecology.config import EcologyConfig, PopulationConfig

    learns = LearningConfig() if learning is None else learning
    held = PopulationConfig(history_capacity=HISTORY_CAPACITY)
    state = agent.PopulationState.founded(founding, population=held, learning=learns)
    port = Recording(
        ArrayGenomePopulation(
            state=state, descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION
        ),
        spawns=refused_spawns,
        weight_norm_bound=learns.weight_norm_bound,
    )
    loop = ecology.steady_state.SteadyState(  # ty: ignore[unresolved-attribute]
        population=port,
        state=state,
        deflator=deflator,
        config=EcologyConfig(
            run_id=run_id,
            population=held,
            learning=learns,
            turnover=turnover,  # ty: ignore[unknown-argument]
        ),
        instrument=INSTRUMENT,
        sizer=NullRiskSizer(),
        simulator=NullExecutionSimulator() if simulator is None else simulator,
        opening_equity=1.0,
    )
    return Wired(state=state, port=port, loop=loop)


def steady(
    *,
    windows: Sequence[BarWindow],
    founding: Sequence[agent.AgentBirth],
    turnover: object,
    deflator: object,
    learning: LearningConfig | None = None,
    look: Callable[[int], None] | None = None,
    simulator: object | None = None,
    run_id: str = "steady-state",
    refused_spawns: frozenset[int] = frozenset(),
) -> Steady:
    """Found ``founding``, wire the real population adapter, and walk ``windows`` once."""
    wired = wire(
        founding=founding,
        turnover=turnover,
        deflator=deflator,
        learning=learning,
        simulator=simulator,
        run_id=run_id,
        refused_spawns=refused_spawns,
    )
    watched = Watched(windows, wired.state, look=look)
    return Steady(
        state=wired.state,
        port=wired.port,
        watched=watched,
        loop=wired.loop,
        run=wired.loop.run(watched),  # ty: ignore[unresolved-attribute]
    )


def spans(watched: Watched, window: int) -> list[tuple[int, int, int]]:
    """Every run of ``window`` consecutive decision points: start, deaths in it, living at its start.

    Whatever bar it starts at, truncated where the walk ends. Deaths are the
    observed set differences, whatever decided them.
    """
    bars = len(watched.living) - 1
    return [
        (
            start,
            sum(
                len(watched.deaths(index))
                for index in range(start, min(start + window, bars))
            ),
            len(watched.living[start]),
        )
        for start in range(bars)
    ]
