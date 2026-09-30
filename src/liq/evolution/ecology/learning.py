"""What an agent learns from an outcome, and the discipline that holds it.

**The family is not this module's choice.** Which of the three online-linear
families the platform learns with was settled by measurement, in the V2.6.2
ablation, against the properties the update has to hold rather than against a
preference: follow-the-regularised-leader with per-coordinate adaptive steps —
FTRL-Proximal — running on forgetting-decayed accumulators. Recursive least
squares holds the same properties and costs an order of magnitude more per
agent per bar, because its state is a covariance and this one's is two vectors.
Normalised online gradient descent is cheaper again and was rejected for a
reason worth restating here, because it is the kind of thing that gets
re-proposed: in that family the forgetting factor enters only the scale
estimate, so it does not control how fast the weights move, and two agents
differing only in it adapt at the same rate. A gene with no effect is not
genetic control of anything.

**The three properties, and where each one lives.**

*The forgetting factor is genetic.* It is read from the agent's genome, never
configured per run, and it decays both accumulators — so an agent at the fast
end of the declared range discards older evidence sooner and follows a change
in the relationship sooner. The range itself is configuration, and a genome
outside it is refused rather than pulled to the edge.

*The state is bounded.* The quantity that runs away in a recursive estimator
with exponential forgetting is the **gain** — the multiplier applied to the next
surprise — because the forgetting divides by something that decays when the
input stops carrying information. Here that gain is
``step_scale / (step_offset + sqrt(energy))``, which is bounded above by
``step_scale / step_offset`` by construction; the configured bound is a second,
outer guarantee that does not rely on that algebra staying true, and reaching it
is recorded and *acted on* rather than logged. The forgetting-weighted counters
are carried too, as learned state a checkpoint captures, but they are not what
the bound watches: a counter that converges to ``1 / (1 - forgetting)`` is
bounded by arithmetic and carries no information about the estimator's health.

*The scaling is causal.* Each agent standardises what it is shown using
statistics accumulated from bars strictly before the one being processed, and
folds the bar in afterwards, with the same forgetting factor the weights use. At
the first decision point there is nothing behind the bar, so there is no scale,
and the standardised value is zero rather than the bar divided by a statistic it
created.

**What this module is not allowed to know.** It never imports the population. It
is array arithmetic over columns whose indices it is handed, which is what lets
the population own one representation of an agent rather than two, and what
keeps the population free to import this.

**Cost.** Measured in the V2.6.2 ablation at 10⁴ agents and 64 features:
roughly 8.6e-07 seconds per agent per bar, against 8.6e-06 for regularised
recursive least squares on the same machine and streams. The figure is recorded
because it multiplies by population size and by bar count, and a family chosen
on cost should carry the number it was chosen on.
"""

from __future__ import annotations

from liq.evolution.ecology.learning_state import (
    DRIFT_PREFIX,
    ENERGY_PREFIX,
    ESTIMATOR_FAMILY,
    GAIN_PREFIX,
    MEASURED_SECONDS_PER_AGENT_PER_BAR,
    SCALE_MASS,
    SCALE_SQUARES_PREFIX,
    SCALE_TOTAL_PREFIX,
    WEIGHT_PREFIX,
    BoundsApplied,
    cold_start,
    declared_columns,
    gain_names,
    weight_names,
)
from liq.evolution.ecology.learning_update import OnlineUpdate

__all__ = [
    "DRIFT_PREFIX",
    "ENERGY_PREFIX",
    "ESTIMATOR_FAMILY",
    "GAIN_PREFIX",
    "MEASURED_SECONDS_PER_AGENT_PER_BAR",
    "SCALE_MASS",
    "SCALE_SQUARES_PREFIX",
    "SCALE_TOTAL_PREFIX",
    "WEIGHT_PREFIX",
    "BoundsApplied",
    "OnlineUpdate",
    "cold_start",
    "declared_columns",
    "gain_names",
    "weight_names",
]
