"""The oracle for two claims about the online update: bounded state, causal scaling.

This module decides what "the estimator's state stayed within its bound" and
"the standardisation consumed only earlier bars" are each *measured as*. It is
declared inside the surfaces of the bindings that lean on it, because a probe
that sampled the wrong thing would leave both claims reading green while the
guardrail they name was gone.

**Where the bound comes from.** Not from here. The bound is a declared
configuration value, and :func:`declared_configuration` hands back every value
one configuration declares, by dotted name, so a check reads the number the run
was configured with rather than a number a test author chose. A key that is not
there is a :class:`KeyError` naming the key, which is the honest report that the
quantity is not configurable yet.

**Why the constant column is inside the surface rather than an input to it.**
Covariance wind-up is defined by the *absence* of excitation: a recursive
estimator with exponential forgetting divides by a quantity that decays when
nothing new arrives, and the gain grows without bound. No stream anybody could
find is perfectly unexciting, so the case has to be constructed — one feature
held at a value it already takes, everything else untouched. That is an invalid
mutation of the declared stream in exactly the sense the fixture register uses
the term, and it is not market data, was never market data, and is not evidence
of anything except that a rejection or a guardrail fires.

**What the causal check compares against.** An independent reimplementation of
the standardisation, written here rather than imported, in two versions: one
that uses only bars strictly before the one being processed, and one that is
allowed to peek at the bar itself. A production trajectory that matches the
first and differs from the second has been shown to be causal; a production
trajectory matching both would mean the comparison cannot tell them apart, and
the check says so. Shifting the statistics window one bar forward *is* the
peeking version, so one mechanism carries both halves of the claim.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from liq.evolution.ecology import agent
from liq.evolution.ecology.types import BarWindow, InstrumentId, UtcTimestamp

#: Dotted names of the configured quantities the checks below read. Held here
#: so the key a check asks for and the key a configuration declares are the same
#: string in one place, and a rename is one edit rather than a silent miss.
STATE_BOUND_KEY = "learning.state_bound"
WEIGHT_NORM_BOUND_KEY = "learning.weight_norm_bound"
FORGETTING_MINIMUM_KEY = "learning.forgetting_minimum"
FORGETTING_MAXIMUM_KEY = "learning.forgetting_maximum"
COLD_START_WEIGHT_KEY = "learning.cold_start_weight"
HISTORY_CAPACITY_KEY = "population.history_capacity"


def declared_configuration(config: object) -> Mapping[str, object]:
    """Every value ``config`` declares, by dotted name.

    Nested configuration objects are flattened rather than returned whole, so a
    quantity buried one level down is addressable by the name a provenance
    record would carry it under, and a quantity that is a literal in the code
    has no name here at all.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(
            f"{config!r} declares nothing: a configuration is a dataclass of "
            "named values, and anything else has no keys to read"
        )
    return MappingProxyType(dict(_flatten(dataclasses.asdict(config), prefix="")))


def _flatten(
    values: Mapping[str, object], *, prefix: str
) -> Iterable[tuple[str, object]]:
    """Walk a nested mapping, yielding dotted name and value."""
    for name, value in values.items():
        dotted = f"{prefix}{name}"
        if isinstance(value, dict):
            yield from _flatten(value, prefix=f"{dotted}.")
            continue
        yield dotted, value


def constant_column(
    stream: Sequence[BarWindow],
    *,
    feature: str,
    instrument: InstrumentId,
) -> tuple[BarWindow, ...]:
    """The same decision points with one feature held at the value it starts on.

    INVALID MUTATION of a declared arithmetic stream. Exactly one thing is
    changed — one feature stops varying — and every instant, bar, segment and
    other feature is the stream's own. It exists so the absence of excitation
    can be exercised, and it is not representative of anything.
    """
    if not stream:
        raise ValueError("a stream with no decision points cannot be mutated")
    held = stream[0].features[instrument][feature]
    return tuple(
        replace(
            window,
            features=MappingProxyType(
                {
                    instrument: MappingProxyType(
                        {**window.features[instrument], feature: held}
                    )
                }
            ),
        )
        for window in stream
    )


@dataclass(frozen=True)
class BoundReport:
    """What the estimator's state did across one whole pass.

    Attributes:
        bars: How many decision points were sampled. Reported so a check that
            silently ran over a prefix can be told from one that ran over the
            pass.
        largest_state_norm: The biggest the estimator's internal state got.
        largest_weight_norm: The biggest the learned weight vector got.
        first_breach_index: Index of the first decision point at which either
            bound was exceeded, or ``None`` if neither ever was.
        first_breach_at: The instant of that decision point, or ``None``.
        first_breach_of: Which bound was the first to be exceeded, by the
            dotted name it is declared under, or ``None``.
    """

    bars: int
    largest_state_norm: float
    largest_weight_norm: float
    first_breach_index: int | None
    first_breach_at: UtcTimestamp | None
    first_breach_of: str | None

    @property
    def within_bounds(self) -> bool:
        """Whether the pass stayed inside both declared bounds throughout."""
        return self.first_breach_index is None


def judge(
    *,
    stamps: Sequence[UtcTimestamp],
    state_norms: Sequence[float],
    weight_norms: Sequence[float],
    state_bound: float,
    weight_norm_bound: float,
) -> BoundReport:
    """Report what the sampled norms did against the two declared bounds.

    Both bounds are judged in one pass and the first one to be exceeded is
    named, because "the state stayed bounded" and "the weights stayed bounded"
    are two claims and a report that collapsed them would not say which
    guardrail was the one that held.
    """
    if not (len(stamps) == len(state_norms) == len(weight_norms)):
        raise ValueError(
            f"the probe was handed {len(stamps)} instants, {len(state_norms)} "
            f"state norms and {len(weight_norms)} weight norms; a report over "
            "rows that do not line up would name the wrong bar"
        )
    breach_index: int | None = None
    breach_of: str | None = None
    for index, (state_norm, weight_norm) in enumerate(
        zip(state_norms, weight_norms, strict=True)
    ):
        if state_norm > state_bound:
            breach_index, breach_of = index, STATE_BOUND_KEY
            break
        if weight_norm > weight_norm_bound:
            breach_index, breach_of = index, WEIGHT_NORM_BOUND_KEY
            break
    return BoundReport(
        bars=len(stamps),
        largest_state_norm=max(state_norms, default=0.0),
        largest_weight_norm=max(weight_norms, default=0.0),
        first_breach_index=breach_index,
        first_breach_at=None if breach_index is None else stamps[breach_index],
        first_breach_of=breach_of,
    )


def reference_standardised(
    values: Sequence[float],
    *,
    forgetting: float,
    peek: bool,
) -> tuple[float, ...]:
    """Standardise ``values`` with forgetting-weighted statistics, twice over.

    With ``peek`` false the statistics behind bar *t* are accumulated from bars
    strictly before it, which is what causality requires. With ``peek`` true the
    bar being processed is folded in first — the statistics window shifted one
    bar forward, which is the defect the claim exists to exclude. The two are
    the same arithmetic in a different order, deliberately: a comparison against
    a reference that differed in any other way would not isolate the ordering.
    """
    mass = 0.0
    total = 0.0
    squares = 0.0
    standardised: list[float] = []
    for value in values:
        if peek:
            mass, total, squares = _accumulate(
                mass, total, squares, value, forgetting=forgetting
            )
        standardised.append(_scale(mass, total, squares, value))
        if not peek:
            mass, total, squares = _accumulate(
                mass, total, squares, value, forgetting=forgetting
            )
    return tuple(standardised)


def _accumulate(
    mass: float,
    total: float,
    squares: float,
    value: float,
    *,
    forgetting: float,
) -> tuple[float, float, float]:
    """Fold one observation into forgetting-weighted first and second moments."""
    return (
        forgetting * mass + 1.0,
        forgetting * total + value,
        forgetting * squares + value * value,
    )


def _scale(mass: float, total: float, squares: float, value: float) -> float:
    """Centre and scale one value by the statistics accumulated so far."""
    if mass <= 0.0:
        return 0.0
    mean = total / mass
    variance = max(squares / mass - mean * mean, 0.0)
    spread = variance**0.5
    if spread <= 0.0:
        return 0.0
    return (value - mean) / spread


def sampled(
    state: agent.PopulationState,
    agent_id: str,
) -> tuple[float, float]:
    """The size of one agent's estimator state and of its learned weights.

    Two numbers rather than one, because FR-3 bounds two different things: the
    gain, covariance or accumulator the estimator carries, and the weight vector
    it produces. An estimator can hold one while losing the other.
    """
    return (
        _norm(state.estimator_state(agent_id).values()),
        _norm(state.learned_weights(agent_id).values()),
    )


def _norm(values: Iterable[float]) -> float:
    """Euclidean length of an iterable of numbers."""
    return sum(float(value) ** 2 for value in values) ** 0.5
