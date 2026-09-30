"""Single-agent façade over the shared array-backed population."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from liq.evolution.ecology.config import LearningConfig, PopulationConfig
from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    Genome,
    InstrumentId,
    Intent,
    UtcTimestamp,
)

from .agent_contracts import AgentBirth, BoundReached

if TYPE_CHECKING:
    from .agent import PopulationState


@dataclass(frozen=True)
class Agent:
    """One agent, addressed by name: a population of one and the rule it shares.

    There is one decision rule in this module and this is not a second copy of
    it. What an evaluation account follows is a single agent, and what the rule
    is written over is a population, so something has to stand between the two;
    this is that something, and it computes nothing. Every wish it hands back
    was formed by :meth:`PopulationState.step` over the same arrays a run of ten
    thousand agents would use, which is why a one-agent run and a population run
    cannot drift apart in the last bit of a sum.

    Attributes:
        agent_id: Who formed the wish. Carried into every intent, so an exposure
            can be traced back to the agent that wanted it rather than to a
            position in a list.
        state: The population of one this agent is the whole of. Reachable,
            because what an agent has learned is inspectable state and hiding
            it behind the wish would make the learning unobservable again.
    """

    agent_id: AgentId
    state: PopulationState

    @classmethod
    def founded(
        cls,
        *,
        agent_id: AgentId,
        genome: Genome,
        feature_schema_version: str,
        model_version: str,
        learning: LearningConfig,
        population: PopulationConfig | None = None,
    ) -> Agent:
        """Start one agent, carrying ``genome`` and having learned nothing yet.

        Args:
            agent_id: Identity the agent is known by.
            genome: The heritable part, including the forgetting gene the update
                reads and the entry gene the wish is compared against.
            feature_schema_version: Vocabulary its gene names refer to. A
                decision point stamped with a different one is refused rather
                than answered, because weights applied to names that vocabulary
                does not define produce a meaningless reading rather than a
                worse one.
            model_version: Decision model it is born under.
            learning: What the online update is turned by. Required, not
                optional: the rule reads what the agent has learned, so an agent
                with no update has nothing to decide on.
            population: What the agent is held under — the outcome-history
                ceiling among it.

        Returns:
            Agent: The agent, before any decision point.
        """
        from .agent import PopulationState

        return cls(
            agent_id=agent_id,
            state=PopulationState.founded(
                (
                    AgentBirth(
                        agent_id=agent_id,
                        genome=genome,
                        learned_state={},
                        feature_schema_version=feature_schema_version,
                        model_version=model_version,
                    ),
                ),
                population=population,
                learning=learning,
            ),
        )

    def intend(self, window: BarWindow, instrument: InstrumentId) -> Intent:
        """Form what this agent wants to hold in ``instrument`` at ``window``.

        The decision point is handed over whole rather than unpacked by the
        caller, so the wish's instant is the decision point's instant by
        construction. A caller passing the two apart could stamp a wish later
        than the reading it was formed from, which is the cheapest look-ahead
        there is.

        Args:
            window: The decision point, carrying the broadcast view.
            instrument: The instrument the wish is formed in.

        Returns:
            Intent: What the agent wants to hold, stamped at ``window.as_of``.

        Raises:
            AgentBornUnderAnotherVocabulary: If the decision point speaks a
                feature vocabulary this agent was not born under.
        """
        return self.state.step(window, instrument).intents()[0]

    def observe(self, as_of: UtcTimestamp, outcome: float) -> tuple[BoundReached, ...]:
        """Show this agent what the wish it last formed went on to earn.

        Args:
            as_of: The instant the outcome became known (UTC). Strictly after
                the decision point the reading was formed at, because what
                holding a reading earned is only known once the bar it was
                acted on over has finished.
            outcome: What that reading earned.

        Returns:
            Records of any declared bound that acted.

        Raises:
            NothingWasShown: If the agent has formed no wish yet.
            OutcomeFromTheSameBar: If ``as_of`` is not after the decision point
                the reading was formed at.
        """
        return self.state.observe(as_of, {self.agent_id: outcome})

    def genome(self) -> Genome:
        """Return the heritable part of this agent, whole."""
        return self.state.genome(self.agent_id)

    def learned_weights(self) -> Mapping[str, float]:
        """Return what the online update has arrived at, by feature."""
        return self.state.learned_weights(self.agent_id)
