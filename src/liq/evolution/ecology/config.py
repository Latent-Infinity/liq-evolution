"""What one pass over history is configured by — and what cannot be configured.

A configuration schema is read twice: once by whoever fills it in, and once by
whoever later asks what a run could possibly have done. The second reading is
the reason this schema is small. Every key here is a knob a run can be turned
by, so a key that exists is a behaviour someone can ask for, whether or not
anybody has.

**There is no epoch, replay, rewind or pass count, and its absence is a
property rather than an oversight.** History is walked once, forward. A key
offering a second pass would not merely permit a mistake — it would make the
mistake invisible, because a figure computed over history counted twice looks
exactly like a figure computed over twice as much history, and every number
downstream of it inherits the repetition silently. The claim that no such key
exists is asserted mechanically against this schema, so adding one is a visible
change to what the run is allowed to be rather than a line in a file.

**Continuity across a split boundary is stated, not assumed.** A walk-forward
boundary divides what history is *used for*; it is not an event in an agent's
life. An agent alive on either side of one is the same agent with the same
learned state, which is what makes the boundary a test of generalisation rather
than a restart in disguise. The alternative reading — that everything begins
again at a fold — is not implemented, so asking for it is refused here instead
of being quietly ignored: a setting that is read, disregarded and recorded in
provenance is worse than one that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_ABSOLUTE_PENALTY",
    "DEFAULT_COLD_START_WEIGHT",
    "DEFAULT_FORGETTING_MAXIMUM",
    "DEFAULT_FORGETTING_MINIMUM",
    "DEFAULT_HISTORY_CAPACITY",
    "DEFAULT_SQUARED_PENALTY",
    "DEFAULT_STATE_BOUND",
    "DEFAULT_STEP_OFFSET",
    "DEFAULT_STEP_SCALE",
    "DEFAULT_WEIGHT_NORM_BOUND",
    "STATE_BOUND_KEY",
    "WEIGHT_NORM_BOUND_KEY",
    "EcologyConfig",
    "LearningConfig",
    "PopulationConfig",
]

#: The dotted names the two bounds are declared under, as a provenance record
#: carries them. Held here rather than built at a call site so the name a run
#: records a breach under and the name a reader looks the number up by are the
#: same string.
STATE_BOUND_KEY = "learning.state_bound"
WEIGHT_NORM_BOUND_KEY = "learning.weight_norm_bound"

#: How many of an agent's own outcomes its history retains, by default. The
#: ceiling is a declared *loss* — the oldest outcome is dropped when it is
#: reached — so it belongs to a key a run records rather than to a default at
#: whatever call site happened to build a population. That is why it is here and
#: not beside the array it bounds.
DEFAULT_HISTORY_CAPACITY = 32

#: What an agent's learned weights are before it has been shown any outcome.
#: Zero rather than a small number: an agent that has learned nothing carries
#: nothing, and any other value would be a prior nobody chose.
DEFAULT_COLD_START_WEIGHT = 0.0

#: The range a forgetting factor read from a genome is confined to. Strictly
#: inside (0, 1]: at zero an agent would remember only the bar in front of it,
#: and the lower end is held well above that so that a mutation cannot produce
#: an agent whose learned state is noise. The upper end is below one so every
#: agent forgets something — an agent at exactly one never adapts, which is the
#: behaviour the whole slice exists to vary.
DEFAULT_FORGETTING_MINIMUM = 0.90
DEFAULT_FORGETTING_MAXIMUM = 0.9999

#: The largest the estimator's internal state may get before the run says so.
#: A recursive estimator with exponential forgetting divides by a quantity that
#: decays when its input stops carrying information, so the state is the thing
#: that runs away first and the thing a bound has to watch.
DEFAULT_STATE_BOUND = 1.0e3

#: The largest the learned weight vector may get. A separate quantity from the
#: one above and separately bounded: an estimator can hold its internal state
#: and still produce weights that grow, and the two failures need telling apart.
DEFAULT_WEIGHT_NORM_BOUND = 1.0e2

#: The four tuned constants of the estimator family the V2.6.2 ablation
#: selected, at the values it measured them at. They are declared here rather
#: than inside the estimator so a finished run's configuration digest covers
#: them: a constant that only exists in code cannot be told apart, after the
#: fact, from a different constant in different code.
DEFAULT_STEP_SCALE = 0.1
DEFAULT_STEP_OFFSET = 1.0
DEFAULT_ABSOLUTE_PENALTY = 0.0
DEFAULT_SQUARED_PENALTY = 1.0


@dataclass(frozen=True)
class PopulationConfig:
    """What a population is held under, where holding it costs something.

    One key so far, and it is here rather than in the module that allocates the
    array because of what it is: a ceiling on an agent's outcome history is a
    declaration that the oldest outcome will be *dropped*. A run that dropped
    evidence at a depth chosen by whichever call site built the population
    could not say afterwards how much it dropped, and a reader of the history
    could not tell a short life from a truncated record.

    Attributes:
        history_capacity: How many of an agent's own outcomes its history
            retains before the oldest is dropped.

    Raises:
        ValueError: If the ceiling retains nothing.
    """

    history_capacity: int = DEFAULT_HISTORY_CAPACITY

    def __post_init__(self) -> None:
        """Refuse a ceiling that keeps no outcome at all."""
        if self.history_capacity < 1:
            raise ValueError(
                "an outcome history that retains nothing is not a history; "
                f"history_capacity was {self.history_capacity}"
            )


@dataclass(frozen=True)
class LearningConfig:
    """Every quantity the online update is allowed to be turned by.

    All nine are keys rather than literals, and that is the point of the class
    existing at all. A bound written into the estimator is a bound nobody can
    read off a finished run; a bound written into a check is a bound whoever
    wrote the check chose. Here they are part of the configuration a run is
    identified by — and therefore of the digest its provenance carries — so the
    numbers a result was produced under are recoverable from the record rather
    than from the code of the day. The four tuned constants of the selected
    family are in here for the same reason, and not because anybody expects to
    turn them: a constant nobody can recover is a constant nobody can check.

    Attributes:
        cold_start_weight: What every learned weight is before the agent has
            been shown an outcome.
        forgetting_minimum: The smallest forgetting factor a genome may carry.
        forgetting_maximum: The largest. Strictly inside (0, 1] together with
            the minimum, so no agent either remembers only the last bar or
            never forgets anything.
        state_bound: The largest the estimator's internal state may get.
        weight_norm_bound: The largest the learned weight vector may get.
        step_scale: The selected family's per-coordinate step scale.
        step_offset: Its per-coordinate step offset. The gain the state bound
            watches is at most ``step_scale / step_offset`` by construction, so
            these two also fix where the guardrail sits relative to the
            arithmetic it guards.
        absolute_penalty: Its absolute-value penalty, which sets weights that
            have earned nothing to exactly zero.
        squared_penalty: Its squared penalty.

    Raises:
        ValueError: If the forgetting range is empty or reaches outside (0, 1],
            if either bound is not positive, or if a step constant is not.
    """

    cold_start_weight: float = DEFAULT_COLD_START_WEIGHT
    forgetting_minimum: float = DEFAULT_FORGETTING_MINIMUM
    forgetting_maximum: float = DEFAULT_FORGETTING_MAXIMUM
    state_bound: float = DEFAULT_STATE_BOUND
    weight_norm_bound: float = DEFAULT_WEIGHT_NORM_BOUND
    step_scale: float = DEFAULT_STEP_SCALE
    step_offset: float = DEFAULT_STEP_OFFSET
    absolute_penalty: float = DEFAULT_ABSOLUTE_PENALTY
    squared_penalty: float = DEFAULT_SQUARED_PENALTY

    def __post_init__(self) -> None:
        """Refuse a range or a bound that would make the declaration meaningless."""
        if not 0.0 < self.forgetting_minimum <= self.forgetting_maximum <= 1.0:
            raise ValueError(
                "the forgetting range must be a non-empty interval strictly "
                "inside (0, 1]; it was "
                f"[{self.forgetting_minimum}, {self.forgetting_maximum}]"
            )
        if self.state_bound <= 0.0 or self.weight_norm_bound <= 0.0:
            raise ValueError(
                "a bound of zero or less admits no state at all, so nothing "
                "could run inside it; the bounds were "
                f"state {self.state_bound} and weight norm "
                f"{self.weight_norm_bound}"
            )
        if self.step_scale <= 0.0 or self.step_offset <= 0.0:
            raise ValueError(
                "the step scale and offset divide the update; at zero or below "
                "there is no step to take. They were "
                f"{self.step_scale} and {self.step_offset}"
            )
        if self.absolute_penalty < 0.0 or self.squared_penalty < 0.0:
            raise ValueError(
                "a negative penalty rewards a larger weight, which is the "
                "opposite of what regularisation is for; they were "
                f"{self.absolute_penalty} and {self.squared_penalty}"
            )


@dataclass(frozen=True)
class EcologyConfig:
    """How one pass over history is driven.

    Attributes:
        run_id: Identity the pass is recorded under. Required, because a result
            that cannot be named cannot be reproduced or compared.
        window_carries_across_segments: Whether an agent's view of history
            survives a walk-forward split boundary. Continuous is the only
            reading implemented; it is carried here so a run's provenance
            records which reading it ran under rather than leaving a reader to
            infer it from the code of the day.
        population: What the population is held under. Nested rather than
            flattened in here, so a provenance record carries each quantity
            under a name that says which part of the run it turns.
        learning: What the online update is turned by.

    Raises:
        ValueError: If the pass could not be identified, or asks for a
            discontinuity at split boundaries that nothing implements.
    """

    run_id: str
    window_carries_across_segments: bool = True
    population: PopulationConfig = field(default_factory=PopulationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

    def __post_init__(self) -> None:
        """Refuse a pass that cannot be named or asks for what does not exist."""
        if not self.run_id:
            raise ValueError(
                "a run must carry an identity: a result nothing names cannot be "
                "reproduced, compared or withdrawn"
            )
        if not self.window_carries_across_segments:
            raise ValueError(
                "window_carries_across_segments=False asks for an agent's view of "
                "history to restart at a walk-forward split boundary; nothing "
                "implements that, and accepting the setting would record a policy "
                "in provenance that the run did not follow"
            )
