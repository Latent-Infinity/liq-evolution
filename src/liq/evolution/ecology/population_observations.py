"""Population observations behavior."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.config import (
    STATE_BOUND_KEY,
    WEIGHT_NORM_BOUND_KEY,
)
from liq.evolution.ecology.types import (
    AgentId,
    UtcTimestamp,
)

from .agent_contracts import (
    BoundReached,
    NothingWasShown,
    OutcomeFromTheSameBar,
    OutcomeNotFinite,
)
from .agent_genes import (
    STORAGE_DTYPE,
)
from .population_state_fields import _PopulationStateFields


class _ObservationsMixin(_PopulationStateFields):
    def observe(
        self, as_of: UtcTimestamp, outcomes: Mapping[AgentId, float]
    ) -> tuple[BoundReached, ...]:
        """Show every living agent what its last decision point actually earned.

        The pairing, and the refusal that protects it, are the point. An
        outcome belongs to the reading that produced it, and it can only be
        known after the bar that reading was acted on over — so an outcome
        stamped at the reading's own instant is refused rather than absorbed.

        Args:
            as_of: The instant the outcome became known (UTC). Must be strictly
                after the decision point it follows.
            outcomes: Every living agent's realised outcome. Every one of them:
                a mapping that left somebody out would update part of a
                population and leave the rest a bar behind, with nothing saying
                which agents were which.

        Returns:
            Records of any declared bound that acted, which are also appended
            to :meth:`bounds_reached`.

        Raises:
            PopulationDoesNotLearn: If this population has no online update.
            NothingWasShown: If no decision point has been answered yet.
            OutcomeFromTheSameBar: If ``as_of`` is not after that decision
                point.
            OutcomeNotFinite: If any outcome is not a finite number. Nothing
                is learned from the rest, and the reading stays paired.
            KeyError: If an outcome names an agent that is not alive, or if a
                living agent was left out.
        """
        self._refuse_if_nothing_learns()
        assert self._update is not None
        if self._shown is None:
            raise NothingWasShown(
                "an outcome was offered before the population had been shown a "
                "decision point, so there is no reading it could belong to"
            )
        shown_at, standardised = self._shown
        if as_of <= shown_at:
            raise OutcomeFromTheSameBar(
                f"an outcome stamped {as_of.isoformat()} was offered for the "
                f"reading formed at {shown_at.isoformat()}; what holding that "
                "reading earned is only known after the bar it was acted on "
                "over, so this one could not have been"
            )
        missing = tuple(agent_id for agent_id in self._ids if agent_id not in outcomes)
        if missing:
            raise KeyError(
                f"{list(missing)} are alive and were given no outcome; a "
                "population updated in part would leave the rest a bar behind "
                "with nothing recording which were which"
            )
        realised = np.fromiter(
            (outcomes[agent_id] for agent_id in self._ids),
            dtype=STORAGE_DTYPE,
            count=len(self._ids),
        )
        if not np.isfinite(realised).all():
            raise OutcomeNotFinite(
                "outcomes that are not finite numbers were offered for "
                f"{self._not_finite(realised)}; one would turn every weight and "
                "gain it reached into not-a-number for good, so the whole "
                "observation is refused and nothing is put in its place"
            )
        applied = self._update.observe(
            self._learned, standardised, realised, self._forgetting()
        )
        self._shown = None
        return self._record(as_of, applied)

    def _not_finite(self, realised: np.ndarray) -> dict[AgentId, float]:
        """Who was offered an outcome that is not a finite number, and what it was."""
        return {
            agent_id: float(value)
            for agent_id, value in zip(self._ids, realised.tolist(), strict=True)
            if not np.isfinite(value)
        }

    def bounds_reached(self) -> tuple[BoundReached, ...]:
        """Return every declared bound that acted, in the order it acted."""
        return tuple(self._bounds_reached)

    def _record(
        self, as_of: UtcTimestamp, applied: online.BoundsApplied
    ) -> tuple[BoundReached, ...]:
        """Append one observation's bound events and hand them back."""
        if not applied.any_applied():
            return ()
        events = tuple(
            BoundReached(agent_id=agent_id, as_of=as_of, bound=bound)
            for bound, flags in (
                (STATE_BOUND_KEY, applied.state.tolist()),
                (WEIGHT_NORM_BOUND_KEY, applied.weights.tolist()),
            )
            for agent_id, reached in zip(self._ids, flags, strict=True)
            if reached
        )
        self._bounds_reached.extend(events)
        return events
