"""A deflator double that leaves some agents unassessed while giving them a real figure.

A C-06 test double. It stands in for the fitness-deflation port and is not a
model of any deflation: every number it returns is this module's declaration,
not a statistic about anything.

**Why it exists.** An agent the deflation did not assess carries no deflated
figure, and selection must be handed none for it. Where such an agent also
carries no *undeflated* figure, "handed its undeflated figure" and "handed
none" are the same ``None``, and a check cannot tell them apart. This double
gives every agent it leaves unassessed a real, non-``None`` undeflated figure,
so the two are different numbers.

**What it declares**, per agent and per trigger — the trigger being the
ordinal of the call, counted from zero, since the loop calls a deflator once per
trigger that has anyone eligible:

* whether the agent is assessed: every agent except those
  :func:`left_unassessed` names, a declared subset fixed by identity alone;
* an assessed agent's deflated figure (:func:`deflated_figure`) and its
  standing (worse below :data:`WORSE_BELOW`, otherwise indistinguishable);
* every agent's undeflated figure, assessed or not, by a second declared
  function handed in per run (:data:`HIGH_UNDEFLATED`,
  :data:`LOW_UNDEFLATED`).

So two runs differing only in that second function are handed the same
deflated figures and standings at every trigger, whatever their populations
do; only what is reported beside them differs.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from liq.evolution.ecology.ports import refuse_an_undeflatable_sample
from liq.evolution.ecology.types import AgentId, DeflatedFitness, FitnessSample

#: An undeflated figure, as a declared function of an agent and a trigger.
Undeflated = Callable[[AgentId, int], float]

#: One agent in this many, by identity, is left unassessed.
UNASSESSED_ONE_IN = 5

#: An assessed agent whose deflated figure is below this is worse.
WORSE_BELOW = 0.2

#: How far an undeflated figure sits from every deflated one: far enough that
#: an agent handed it would rank first, or last, whatever the deflated figures.
UNDEFLATED_OFFSET = 1.0e3


def _unit(text: str) -> float:
    """A declared number in [0, 1) read off ``text``: its CRC-32 over 2³²."""
    return zlib.crc32(text.encode()) / 2.0**32


def left_unassessed(agent_id: AgentId) -> bool:
    """Whether the double declines to assess ``agent_id``: by identity alone."""
    return zlib.crc32(agent_id.encode()) % UNASSESSED_ONE_IN == 0


def deflated_figure(agent_id: AgentId, trigger: int) -> float:
    """An assessed agent's deflated figure at ``trigger``."""
    return _unit(f"{agent_id}@{trigger}")


def standing_of(figure: float) -> str:
    """The standing the double declares for a deflated ``figure``."""
    return "worse" if figure < WORSE_BELOW else "indistinguishable"


def high_undeflated(agent_id: AgentId, trigger: int) -> float:
    """An undeflated figure above every deflated one."""
    return UNDEFLATED_OFFSET + deflated_figure(agent_id, trigger)


def low_undeflated(agent_id: AgentId, trigger: int) -> float:
    """An undeflated figure below every deflated one."""
    return -UNDEFLATED_OFFSET - deflated_figure(agent_id, trigger)


HIGH_UNDEFLATED: Undeflated = high_undeflated
LOW_UNDEFLATED: Undeflated = low_undeflated


@dataclass
class ReproduceDeflator:
    """The double. ``undeflated`` is the only thing two runs may differ in.

    Attributes:
        undeflated: Every agent's undeflated figure, per agent and trigger.
        minimum_window: The fewest outcomes this double assesses on.
        calls: Every call's results, in order; a call's place is its trigger.
    """

    undeflated: Undeflated
    minimum_window: int = 2
    calls: list[tuple[DeflatedFitness, ...]] = field(default_factory=list)

    def deflate(self, samples: Sequence[FitnessSample]) -> tuple[DeflatedFitness, ...]:
        """The declared verdict on each sample at this call's trigger."""
        refuse_an_undeflatable_sample(samples)
        trigger = len(self.calls)
        results = tuple(
            self._verdict(sample.agent_id, trigger, sample, len(samples))
            for sample in samples
        )
        self.calls.append(results)
        return results

    def _verdict(
        self, agent_id: AgentId, trigger: int, sample: FitnessSample, compared: int
    ) -> DeflatedFitness:
        undeflated = self.undeflated(agent_id, trigger)
        if left_unassessed(agent_id):
            return DeflatedFitness(
                agent_id=agent_id,
                undeflated=undeflated,
                deflated=None,
                compared=compared,
                independent_outcomes=sample.independent_outcomes,
                standing="indistinguishable",
            )
        figure = deflated_figure(agent_id, trigger)
        return DeflatedFitness(
            agent_id=agent_id,
            undeflated=undeflated,
            deflated=figure,
            compared=compared,
            independent_outcomes=sample.independent_outcomes,
            standing=standing_of(figure),  # type: ignore[arg-type]
        )
