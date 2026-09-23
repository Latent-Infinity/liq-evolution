"""What the run computes from bars, and how much history each computation needs.

A feature is declared before it is computed, and the declaration carries the two
numbers that decide when it may be read: how many completed bars it reads, and
how long after the bar it was computed from it becomes available. Both are
properties of the feature rather than of the code that happens to compute it,
which is why they are written down beside the computation instead of being
inferred from it. A computation that quietly read more bars than it declared
would be readable before it could be right.

**Why the maximum is configuration and not a constant.** The rolling window the
substrate holds is sized by it, and that size is the run's memory floor —
instruments times bars times fields, held for the whole pass. A feature deeper
than the configured maximum is therefore not a feature that warms up slowly, it
is a feature the run cannot afford, and it is refused where it is declared. The
alternative is worse in the way that matters: a run that accepted it would
discover the problem at the first decision point that could not serve it, by
either computing the feature on a short window or excluding it forever, and
neither failure announces itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from liq.evolution.ecology.types import Bar

__all__ = [
    "DEFAULT_MAXIMUM_LOOKBACK",
    "FeatureComputation",
    "FeatureDeclaration",
    "FeatureSet",
]

#: How deep a rolling window a run holds unless it configures otherwise. A
#: default, not a limit anyone has justified: it is the depth the approved
#: fixtures were exported against, so a set left unconfigured lines up with the
#: warm-up the fixture manifest records.
DEFAULT_MAXIMUM_LOOKBACK = 200

#: What computes one feature's value. It is handed exactly the bars the
#: declaration asked for, oldest first, and returns one number. It is never
#: handed the whole window, so it cannot read further back than it declared.
type FeatureComputation = Callable[[Sequence[Bar]], float]


@dataclass(frozen=True)
class FeatureDeclaration:
    """One feature: what it is called, what it needs, and when it may be read.

    Attributes:
        name: What agents read the value under. Names belong to a feature
            vocabulary whose version travels with every decision point.
        lookback: How many consecutive completed bars the computation reads.
            One means the bar that has just closed and nothing before it.
        availability_lag: How many bar periods pass between the bar the value
            was computed from and the instant it may be read. Zero means the
            value is readable at the decision point it was computed at; one
            means a value computed from a bar's close is not readable until the
            next decision point.
        compute: What turns those bars into the value.

    Raises:
        ValueError: If the declaration could not describe a real feature — an
            unnamed one, one reading no bars, or one available before it exists.
    """

    name: str
    lookback: int
    availability_lag: int
    compute: FeatureComputation

    def __post_init__(self) -> None:
        """Refuse a declaration no bar sequence could satisfy."""
        if not self.name:
            raise ValueError("a feature declaration must name the feature")
        if self.lookback < 1:
            raise ValueError(
                f"feature {self.name!r} declares a lookback of {self.lookback}; "
                "a feature reads at least the bar that has just closed"
            )
        if self.availability_lag < 0:
            raise ValueError(
                f"feature {self.name!r} declares an availability lag of "
                f"{self.availability_lag}; a value cannot be readable before the "
                "bar it was computed from closed"
            )


@dataclass(frozen=True)
class FeatureSet:
    """The features one run computes, and the deepest window it will hold.

    Attributes:
        declarations: The features, in the order they are computed.
        maximum_lookback: The deepest run of bars the substrate holds per
            instrument. No declaration may ask for more.

    Raises:
        ValueError: If the set computes nothing, names one feature twice, or
            holds a declaration deeper than the configured maximum.
    """

    declarations: tuple[FeatureDeclaration, ...]
    maximum_lookback: int = DEFAULT_MAXIMUM_LOOKBACK

    def __post_init__(self) -> None:
        """Refuse a set the configured window could not serve."""
        if self.maximum_lookback < 1:
            raise ValueError(
                f"the maximum lookback is {self.maximum_lookback}; a run holds at "
                "least the bar that has just closed"
            )
        if not self.declarations:
            raise ValueError("a feature set computes at least one feature")
        seen: set[str] = set()
        for declaration in self.declarations:
            if declaration.name in seen:
                raise ValueError(
                    f"feature {declaration.name!r} is declared twice; one name "
                    "cannot carry two values in one broadcast"
                )
            seen.add(declaration.name)
            if declaration.lookback > self.maximum_lookback:
                raise ValueError(
                    f"feature {declaration.name!r} declares a lookback of "
                    f"{declaration.lookback}, deeper than the configured maximum "
                    f"of {self.maximum_lookback}; raise the maximum deliberately "
                    "or declare a shallower feature, because the window this run "
                    "holds is sized by that maximum"
                )

    @property
    def names(self) -> tuple[str, ...]:
        """What every declared feature is called, in declaration order."""
        return tuple(declaration.name for declaration in self.declarations)
