"""A declared arithmetic stream whose relationship steps once, and the walk over it.

**Nothing here is market data and nothing here may be read as market data.** The
decision points are the arithmetic ramp
:class:`~liq.evolution.ecology.adapters.NullBarSource` declares — a unit step at
a constant half-range and a constant volume — carrying a feature view this
module computes from the decision point's index and nothing else. Every number
a test computes from it is a number about the ramp.

**What was asked for, what is here, and why they differ.** The plan's V2.5 asks
for the level shift to be *located* in the approved fixture DATA-E01 rather than
injected, on the ground that a manufactured shift would be fabricated data. That
is the right rule and it cannot be followed here: V2.5's ``Repo:`` is
``liq-evolution``, and this repository can neither hold nor reach the approved
fixtures — the plan's own header settles that, and the same correction was
applied to V1.11.3, V1.11.4 and V2.3. So the stream is declared arithmetic
throughout. The difference is not cosmetic and the memo must carry it: over real
tape the claim would be about an instrument, and here it is about the estimator's
response to a step it was handed. The estimator's response is the whole of what
EV-E09 claims, so the evidence is not empty — but it is structural evidence, of
the kind this repository is for, and it is not evidence about any market.

**What the real slice would have offered, recorded so the difference is legible.**
DATA-E01.pcar carries a break identified *structurally*, from the exchange
calendar, before any statistic was computed: the second session's open, 191
scorable bars past the first decision point, with 389 scorable bars after it.
Two properties of that slice bear on how this stream is shaped. The break is
**volatility-like**: the absolute per-bar return separates across it while the
signed return does not, so a learner shown that tape would have to be scored on
a volatility-like outcome. And the slice's own permutation p-values are
**conditional on a selection** — 1,521 qualifying session pairs were searched and
the largest step taken — so they certify that file and say nothing about markets,
and no panel or memo may read them up.

**How the break here is placed.** The step falls on the declared ramp's own
walk-forward segment boundary, which the bar source derives before this module
sees it — the closest this stream can come to a break that is located rather
than chosen. The *size* of the step is declared here, and that is an injection;
it is stated rather than disguised.

**The shape of the learning problem.** One exciting feature, cycling through a
short declared period, and an outcome that is a fixed multiple of it. The
multiple changes once, at the break. An estimator that forgets faster should
reach the new multiple sooner. That is the whole of what is being shown.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from liq.evolution.ecology import Genome, agent
from liq.evolution.ecology.adapters import NullBarSource
from liq.evolution.ecology.types import AgentId, BarWindow, InstrumentId

#: How many decision points the stream covers. Chosen so that the span after
#: the break is comparable with the 389 scorable bars DATA-E01.pcar offers after
#: its own, which is what a slow forgetting factor needs to be distinguishable
#: from a fast one rather than merely slower at the end.
DECISION_POINTS = 800

#: Where the relationship steps. It is the declared ramp's walk-forward segment
#: boundary, so the position is structural rather than chosen; only the size of
#: the step below is declared here.
BREAK_AT = DECISION_POINTS // 2

#: The declared feature's period. Short and odd, so the feature is persistently
#: exciting over any window longer than a few bars — which is the condition the
#: constant-column mutation in ``state_norm_probe`` removes.
CYCLE = 7

#: How much of the feature the outcome carries before the break, and after it.
#: The sign changes, so an estimator that has adapted can be told from one that
#: has not by the sign of what it learned, without reference to either target.
SENSITIVITY_BEFORE = 1.0
SENSITIVITY_AFTER = -2.0

#: The one feature the stream broadcasts.
FEATURE = "cycle"

#: The entry gene every agent here carries. Zero, so a wish turns on the sign
#: of the reading and a check can reconstruct one without quoting a number the
#: genome does not carry.
ENTRY_AT = 0.0

#: Vocabularies everything here is stamped with. Named so nothing built from
#: this stream can be mistaken for something built from the ramp's own feature
#: view, which carries different names.
FEATURE_SCHEMA_VERSION = "level-shift-features-1"
GENE_SCHEMA_VERSION = "level-shift-genes-1"
MODEL_VERSION = "level-shift-model-1"

#: The single instrument every decision point carries a bar for.
INSTRUMENT: InstrumentId = "NULL"

#: Who is alive in the populations built here: two agents differing only in how
#: fast they forget.
QUICK_TO_FORGET = "quick-to-forget"
SLOW_TO_FORGET = "slow-to-forget"


def feature_at(index: int) -> float:
    """The declared feature's value at decision point ``index``."""
    return float(index % CYCLE) - float(CYCLE // 2)


def sensitivity_at(index: int) -> float:
    """How much of the feature the outcome carries at decision point ``index``."""
    return SENSITIVITY_BEFORE if index < BREAK_AT else SENSITIVITY_AFTER


def outcome_at(index: int) -> float:
    """The outcome realised over the bar that follows decision point ``index``."""
    return sensitivity_at(index) * feature_at(index)


def windows(count: int = DECISION_POINTS) -> tuple[BarWindow, ...]:
    """The stream's decision points, in order, with the declared feature view.

    The bars, instants and walk-forward segments are the ramp's own; only the
    feature view is replaced, so the segment boundary the break sits on is the
    bar source's rather than this module's.
    """
    source = NullBarSource(
        instrument=INSTRUMENT,
        window_count=count,
        segment_length=BREAK_AT,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    return tuple(
        replace(
            window,
            features=MappingProxyType(
                {INSTRUMENT: MappingProxyType({FEATURE: feature_at(index)})}
            ),
        )
        for index, window in enumerate(source.windows())
    )


def birth(agent_id: AgentId, *, forgetting: float) -> agent.AgentBirth:
    """One agent that differs from the other only in how fast it forgets.

    Every other gene is identical, which is what makes the comparison a
    comparison of the forgetting gene rather than of two agents.
    """
    return agent.AgentBirth(
        agent_id=agent_id,
        genome=Genome(
            genes=MappingProxyType(
                {
                    f"{agent.MASK_PREFIX}{FEATURE}": 1.0,
                    f"{agent.WEIGHT_PREFIX}{FEATURE}": 1.0,
                    agent.ENTRY_THRESHOLD: ENTRY_AT,
                    agent.FORGETTING_FACTOR: forgetting,
                }
            ),
            schema_version=GENE_SCHEMA_VERSION,
        ),
        learned_state=MappingProxyType({}),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=MODEL_VERSION,
    )


def two_agents(*, quick: float, slow: float) -> tuple[agent.AgentBirth, ...]:
    """The pair the convergence comparison is made between."""
    return (
        birth(QUICK_TO_FORGET, forgetting=quick),
        birth(SLOW_TO_FORGET, forgetting=slow),
    )


@dataclass(frozen=True)
class Trajectory:
    """What one agent learned at each decision point, and what it was shown.

    Attributes:
        agent_id: Whose trajectory this is.
        learned: The weight the agent carried for the declared feature after
            each outcome it was shown, in order.
        in_force: The weight the agent carried for the declared feature *when
            it formed the wish* at each decision point — that is, before the
            outcome of that decision point was shown to it. Recorded apart
            from ``learned`` because they are a bar out of step with each
            other, and a check that reconstructed a wish from the wrong one
            would be reconstructing a decision nobody took.
        wanted: The exposure the agent wished for at each decision point.
        standardised: What the wish and the update both consumed at each
            decision point — the feature after the agent's own online
            standardisation.
        state_norms: The size of the estimator's internal state at each
            decision point.
        weight_norms: The size of the learned weight vector at each decision
            point.
    """

    agent_id: AgentId
    learned: tuple[float, ...]
    in_force: tuple[float, ...]
    wanted: tuple[float, ...]
    standardised: tuple[float, ...]
    state_norms: tuple[float, ...]
    weight_norms: tuple[float, ...]


def walk(
    state: agent.PopulationState,
    stream: Sequence[BarWindow],
    *,
    outcomes: Sequence[float] | None = None,
) -> Mapping[AgentId, Trajectory]:
    """Step ``state`` over ``stream``, showing it each outcome a bar after the fact.

    The pairing is the point. A decision point is answered from the feature
    view at that instant; the outcome of holding what was wanted there is
    realised over the bar that follows, and is shown to the population stamped
    at that later instant. Nothing is shown to the update at the instant it
    could not yet have been known.
    """
    realised = (
        tuple(outcome_at(index) for index in range(len(stream)))
        if outcomes is None
        else tuple(outcomes)
    )
    ids = state.agent_ids()
    learned: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}
    in_force: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}
    wanted: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}
    standardised: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}
    state_norms: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}
    weight_norms: dict[AgentId, list[float]] = {agent_id: [] for agent_id in ids}

    for index, window in enumerate(stream):
        step = state.step(window, INSTRUMENT)
        for agent_id, exposure in zip(step.agent_ids, step.wanted, strict=True):
            wanted[agent_id].append(exposure)
        for agent_id in ids:
            standardised[agent_id].append(state.standardised(agent_id)[0])
            in_force[agent_id].append(state.learned_weights(agent_id)[FEATURE])
        if index + 1 >= len(stream):
            break
        state.observe(
            stream[index + 1].as_of,
            dict.fromkeys(ids, realised[index]),
        )
        for agent_id in ids:
            learned[agent_id].append(state.learned_weights(agent_id)[FEATURE])
            estimator = state.estimator_state(agent_id)
            state_norms[agent_id].append(_norm(estimator.values()))
            weight_norms[agent_id].append(
                _norm(state.learned_weights(agent_id).values())
            )

    return MappingProxyType(
        {
            agent_id: Trajectory(
                agent_id=agent_id,
                learned=tuple(learned[agent_id]),
                in_force=tuple(in_force[agent_id]),
                wanted=tuple(wanted[agent_id]),
                standardised=tuple(standardised[agent_id]),
                state_norms=tuple(state_norms[agent_id]),
                weight_norms=tuple(weight_norms[agent_id]),
            )
            for agent_id in ids
        }
    )


def bars_until_the_sign_flips(trajectory: Trajectory) -> int | None:
    """How long after the break the learned weight took to change sign.

    Target-free on purpose: the two agents standardise with their own forgetting
    factors, so their converged weights are not the same number, and a statistic
    measured against either one would be measuring the standardiser as much as
    the learner. What the break does is reverse the sign of the relationship, so
    the first bar after it at which the learned weight has crossed zero is a
    statistic about adaptation and nothing else.
    """
    before = trajectory.learned[: BREAK_AT - 1]
    if not before or before[-1] <= 0.0:
        return None
    for offset, weight in enumerate(trajectory.learned[BREAK_AT - 1 :]):
        if weight < 0.0:
            return offset
    return None


#: How much of the move to its own new level an agent has to have made before
#: it counts as having settled, and how many bars at the end are averaged to
#: say where that level is. Both are properties of the statistic rather than of
#: any estimator, which is why they live here and not in a check.
SETTLED_WITHIN = 0.10
PLATEAU_BARS = 50


def bars_until_it_settles(trajectory: Trajectory) -> int | None:
    """How long after the break the learned weight took to reach its own plateau.

    Target-free, and it has to be. An estimator that regularises does not
    converge on the relationship the stream carries — it converges on a shrunk
    version of it, and how shrunk depends on how much evidence it has kept,
    which is what the forgetting factor decides. Measuring the distance to the
    stream's own coefficient would therefore measure the shrinkage and call it
    the rate, and would rank a slow agent above a fast one for the wrong
    reason. What is asked here instead is how long each agent took to arrive
    wherever *it* was going.
    """
    learned = trajectory.learned
    if len(learned) <= BREAK_AT:
        return None
    settled_at = sum(learned[-PLATEAU_BARS:]) / PLATEAU_BARS
    started_from = learned[BREAK_AT - 2]
    travel = abs(settled_at - started_from)
    if travel == 0.0:
        return None
    for offset, weight in enumerate(learned[BREAK_AT - 1 :]):
        if abs(weight - settled_at) <= SETTLED_WITHIN * travel:
            return offset
    return None


def _norm(values: Iterable[float]) -> float:
    """Euclidean length of an iterable of numbers."""
    return sum(float(value) ** 2 for value in values) ** 0.5
