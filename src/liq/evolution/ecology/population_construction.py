"""Population construction behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.config import (
    STATE_BOUND_KEY,
    LearningConfig,
    PopulationConfig,
)
from liq.evolution.ecology.types import (
    UtcTimestamp,
)

from .agent_contracts import (
    AgentBirth,
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    ForgettingFactorOutsideItsRange,
    Lineage,
    PopulationSnapshot,
    StartingStateOutsideItsBound,
)
from .agent_genes import (
    ENTRY_THRESHOLD,
    FLAT,
    FORGETTING_FACTOR,
    STORAGE_DTYPE,
    WEIGHT_PREFIX,
    _gene_layout,
)

if TYPE_CHECKING:
    pass


from .population_state_fields import _PopulationStateFields


class _ConstructionMixin(_PopulationStateFields):
    def __init__(self, snapshot: PopulationSnapshot) -> None:
        """Hold the agents ``snapshot`` describes, refusing a set that is not one.

        Args:
            snapshot: The agents to hold and the shape to hold them in.

        Raises:
            ValueError: If nobody is alive, if two agents share an identity, if
                the agents do not share one gene vocabulary, one learned-state
                vocabulary and one gene schema version, if a genome switches on
                a feature it carries no weight for or carries no entry gene, if
                an outcome history is longer than the ceiling it is restored
                under, or if that ceiling retains nothing.
            ValueError: Also if a reading awaiting its outcome is carried into
                a population that does not learn, or does not name exactly the
                living agents at the population's feature width.
            ForgettingFactorOutsideItsRange: If the population learns and some
                genome carries a forgetting factor the configuration disallows.
            KeyError: If the population learns and the agents were not born
                declaring the columns the update writes, or carrying the gene
                it reads.
        """
        agents = snapshot.agents
        if not agents:
            raise ValueError(
                "a population with no agents is not a population: nothing would "
                "be stepped, and every figure computed over it would be a "
                "figure over an empty set rather than an absent one"
            )
        if snapshot.history_capacity < 1:
            raise ValueError(
                "an outcome history that retains nothing is not a history; "
                f"history_capacity was {snapshot.history_capacity}"
            )
        self._ids = tuple(held.agent_id for held in agents)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError(
                "two agents share an identity, so neither could be scored, "
                "snapshotted or inherited from without the other"
            )
        self._row_of = {agent_id: row for row, agent_id in enumerate(self._ids)}
        self.history_capacity = snapshot.history_capacity
        self.gene_schema_version = agents[0].genome.schema_version
        self.gene_names, self.feature_names = _gene_layout(agents[0].genome)
        self.learned_names = tuple(sorted(agents[0].learned_state))
        self._learned_at = {
            name: column for column, name in enumerate(self.learned_names)
        }
        self._entry_at = self.gene_names.index(ENTRY_THRESHOLD)
        self._features = len(self.feature_names)

        count = len(agents)
        self._genes = np.zeros((count, len(self.gene_names)), dtype=STORAGE_DTYPE)
        self._learned = np.zeros((count, len(self.learned_names)), dtype=STORAGE_DTYPE)
        self._intended = np.zeros(count, dtype=STORAGE_DTYPE)
        self._realised = np.zeros(count, dtype=STORAGE_DTYPE)
        self._history = np.zeros((count, self.history_capacity), dtype=STORAGE_DTYPE)
        self._retained = np.zeros(count, dtype=np.int64)
        for row, held in enumerate(agents):
            self._admit(row, held)
        self._lineage = tuple(
            Lineage(agent_id=held.agent_id, parents=held.parents, born_at=held.born_at)
            for held in agents
        )
        self._versions = tuple(
            AgentVersions(
                feature_schema_version=held.feature_schema_version,
                model_version=held.model_version,
            )
            for held in agents
        )
        self._born_under = {
            version.feature_schema_version for version in self._versions
        }
        self.learning = snapshot.learning
        self._update: online.OnlineUpdate | None = None
        self._forgetting_at: int | None = None
        self._bounds_reached: list[BoundReached] = list(snapshot.bounds_reached)
        if self.learning is not None:
            self._start_learning(self.learning)
        self._shown: tuple[UtcTimestamp, np.ndarray] | None = self._resume_pending(
            snapshot.pending
        )

    def _start_learning(self, configured: LearningConfig) -> None:
        """Bind the update to its columns and refuse a rate nobody declared."""
        if FORGETTING_FACTOR not in self.gene_names:
            raise KeyError(
                f"a population that learns reads {FORGETTING_FACTOR!r} from "
                f"every genome and these carry {sorted(self.gene_names)}; how "
                "fast an agent forgets is inherited, never defaulted"
            )
        self._forgetting_at = self.gene_names.index(FORGETTING_FACTOR)
        rates = self._genes[:, self._forgetting_at]
        outside = (rates < configured.forgetting_minimum) | (
            rates > configured.forgetting_maximum
        )
        if outside.any():
            strangers = tuple(
                agent_id
                for agent_id, out in zip(self._ids, outside.tolist(), strict=True)
                if out
            )
            raise ForgettingFactorOutsideItsRange(
                f"{strangers} carry forgetting factors outside the declared "
                f"range [{configured.forgetting_minimum}, "
                f"{configured.forgetting_maximum}]: "
                f"{sorted(set(rates[outside].tolist()))}"
            )
        self._update = online.OnlineUpdate(
            config=configured,
            features=self.feature_names,
            column_of=self._learned_at,
        )

    @classmethod
    def founded(
        cls,
        births: Sequence[AgentBirth],
        *,
        population: PopulationConfig | None = None,
        learning: LearningConfig | None = None,
    ) -> Self:
        """Start a population from ``births``, each holding nothing and wanting nothing.

        Args:
            births: Who is alive at the start, and what each was born with.
            population: What the population is held under — the outcome-history
                ceiling among it. Defaults to the schema's own declaration
                rather than to a number written at this call, so the ceiling a
                run dropped outcomes at is one of the keys its provenance
                carries.
            learning: What the online update is turned by, or ``None`` for a
                population that does not learn. When it is given, every agent
                carries exactly the update's own columns. A birth carrying no
                learned state starts them from its own genome's ``weight.``
                genes — its prior; a birth carrying all of them starts from
                exactly what it carried, and its own ``weight.`` genes are not
                read.

        A birth carrying nothing is founded with the family's largest gain. If
        the declared state bound is below that gain, the gain is held at
        founding: the evidence starts at the least value the bound admits, the
        prior is seeded against that evidence so the starting weight is still
        the gene exactly, and the hold is recorded in :meth:`bounds_reached`
        before any decision point is answered. A birth carrying learned state
        is admitted verbatim and never held.

        Returns:
            PopulationState: The population, before any decision point.

        Raises:
            ValueError: If a learning birth carries part of the update's
                columns, or a name the update does not declare.
            StartingStateOutsideItsBound: If a learning agent would start with
                a learned weight that is not finite, or a weight vector whose
                norm exceeds the declared weight-norm bound, or if an inherited
                state carries a gain outside the declared state bound.
        """
        held = PopulationConfig() if population is None else population
        starts = tuple(_starting_state(birth, learning) for birth in births)
        founded = cls(
            PopulationSnapshot(
                agents=tuple(
                    AgentSnapshot(
                        agent_id=birth.agent_id,
                        genome=birth.genome,
                        learned_state=start.learned_state,
                        intended=FLAT,
                        realised=FLAT,
                        history=(),
                        parents=birth.parents,
                        born_at=birth.born_at,
                        feature_schema_version=birth.feature_schema_version,
                        model_version=birth.model_version,
                    )
                    for birth, start in zip(births, starts, strict=True)
                ),
                history_capacity=held.history_capacity,
                learning=learning,
                bounds_reached=tuple(
                    BoundReached(
                        agent_id=birth.agent_id,
                        as_of=birth.born_at,
                        bound=STATE_BOUND_KEY,
                        at_founding=True,
                    )
                    for birth, start in zip(births, starts, strict=True)
                    if start.held_at_founding
                ),
            )
        )
        founded._refuse_a_start_outside_the_bound(
            np.asarray([start.carried for start in starts], dtype=np.bool_)
        )
        return founded

    def _refuse_a_start_outside_the_bound(self, carried: np.ndarray) -> None:
        """Refuse agents whose starting state their bounds would not admit.

        Applied at a birth and not at a restore: a restored agent resumes a
        state the bound already held, and a restore never re-reads a prior.
        ``carried`` marks the births that inherited a learned state, whose gain
        is checked as well as their weights.
        """
        if self._update is None:
            return
        unfit = self._update.unfit_to_start(self._learned, carried)
        if not unfit.any():
            return
        weight_norms = self._update.weight_norms(self._learned).tolist()
        gain_norms = self._update.state_norms(self._learned).tolist()
        strangers = {
            agent_id: {"weight_norm": weight, "gain_norm": gain}
            for agent_id, weight, gain, out in zip(
                self._ids, weight_norms, gain_norms, unfit.tolist(), strict=True
            )
            if out
        }
        assert self.learning is not None
        raise StartingStateOutsideItsBound(
            f"agents would start with learned weights outside the declared "
            f"weight-norm bound {self.learning.weight_norm_bound}, or not "
            f"finite, or with an inherited gain outside the declared state "
            f"bound {self.learning.state_bound}; starting norms {strangers}"
        )

    @classmethod
    def restored(cls, snapshot: PopulationSnapshot) -> Self:
        """Take a population back out of ``snapshot``, ready to go on deciding.

        What comes back stands on its own: it holds no reference to the
        population the snapshot was taken from, and nothing it goes on to do
        consults one.

        Args:
            snapshot: The population as it was put away.

        Returns:
            PopulationState: The same population, mid-walk.
        """
        return cls(snapshot)


@dataclass(frozen=True)
class _Start:
    """What one birth starts from, and how it came to start there.

    Attributes:
        learned_state: The learned columns the agent is admitted with.
        carried: Whether they were inherited rather than built from a prior.
        held_at_founding: Whether the state bound acted on the founding gain.
    """

    learned_state: Mapping[str, float]
    carried: bool
    held_at_founding: bool


def _starting_state(birth: AgentBirth, learning: LearningConfig | None) -> _Start:
    """What ``birth`` starts out having learned, all or nothing.

    Without an update, what the birth declares is what it carries. With one, a
    birth carries either nothing — and starts from its own genome's
    ``weight.`` genes as its prior, at the founding evidence the state bound
    admits — or exactly the columns the update declares, admitted verbatim
    with no prior applied and no hold. A part of that set is
    refused, because a weight carried without the accumulator it is computed
    from is a state no update could resume; so is a name the update does not
    declare, because a column nothing chose would be silently grown rather
    than refused.

    Raises:
        ValueError: If the birth carries some but not all of the update's
            columns, or any name the update does not declare.
    """
    if learning is None:
        return _Start(birth.learned_state, carried=True, held_at_founding=False)
    _, features = _gene_layout(birth.genome)
    if not birth.learned_state:
        genes = birth.genome.genes
        energy, held = online.founding_energy(learning, features)
        return _Start(
            online.cold_start(
                {name: genes[f"{WEIGHT_PREFIX}{name}"] for name in features},
                learning,
                energy,
            ),
            carried=False,
            held_at_founding=held,
        )
    declared = set(online.declared_columns(features))
    carried = set(birth.learned_state)
    if carried != declared:
        raise ValueError(
            f"{birth.agent_id!r} is born carrying learned state that is neither "
            "nothing nor the whole of what the update declares: missing "
            f"{sorted(declared - carried)}, undeclared {sorted(carried - declared)}"
        )
    return _Start(birth.learned_state, carried=True, held_at_founding=False)
