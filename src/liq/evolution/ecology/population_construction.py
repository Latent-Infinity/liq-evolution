"""Population construction behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.config import (
    STATE_BOUND_KEY,
    LearningConfig,
    PopulationConfig,
)
from liq.evolution.ecology.types import (
    BOOK_ACCOUNT_ID,
    AgentId,
    UtcTimestamp,
)

from .agent_contracts import (
    AgentBirth,
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    ForgettingFactorOutsideItsRange,
    LearnedStateDisposition,
    Lineage,
    PopulationSnapshot,
    StartingStateOutsideItsBound,
)
from .agent_genes import (
    ENTRY_THRESHOLD,
    FLAT,
    FORGETTING_FACTOR,
    WEIGHT_PREFIX,
    _gene_layout,
)
from .population_state_fields import _PopulationStateFields


class _ConstructionMixin(_PopulationStateFields):
    def __init__(self, snapshot: PopulationSnapshot) -> None:
        """Hold the agents ``snapshot`` describes, refusing a set that is not one.

        Args:
            snapshot: The agents to hold and the shape to hold them in.

        Raises:
            ValueError: If nobody is alive, if two agents share an identity, if
                an agent carries the aggregate book's reserved identity, if the
                agents do not share one gene vocabulary, one learned-state
                vocabulary and one gene schema version, if a genome switches on
                a feature it carries no weight for or carries no entry gene, if
                an outcome history is longer than the ceiling it is restored
                under, or if that ceiling retains nothing.
            ValueError: Also if a reading awaiting its outcome is carried into
                a population that does not learn, or does not name exactly the
                living agents at the population's feature width; or if a
                departed agent is recorded twice, is among the living, or
                carries no death instant; or if a refusal count is negative.
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
        for agent_id in self._ids:
            _refuse_the_book_identity(agent_id)
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

        self._stores = {}
        self._allocate(len(agents))
        self._reallocations = 0
        for row, held in enumerate(agents):
            self._admit(row, held)
        self._hold_the_reference_returns(snapshot.reference_returns)
        self._lineage_of = {held.agent_id: _lineage(held) for held in agents}
        self._departed = []
        self._refusals = _a_count(snapshot.admission_refusals)
        self._hold_the_departed(snapshot.departed)
        self._versions = [
            AgentVersions(
                feature_schema_version=held.feature_schema_version,
                model_version=held.model_version,
            )
            for held in agents
        ]
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
        self._refuse_rates_outside_the_range(
            self._genes[:, self._forgetting_at], self._ids
        )
        self._update = online.OnlineUpdate(
            config=configured,
            features=self.feature_names,
            column_of=self._learned_at,
        )

    def _hold_the_departed(self, departed: Sequence[Lineage]) -> None:
        """Take back the lineage of the dead, refusing a record that is not one."""
        for lineage in departed:
            if lineage.died_at is None:
                raise ValueError(
                    f"{lineage.agent_id!r} is recorded as departed with no death "
                    "instant, so nothing says when it stopped being alive"
                )
            if lineage.death_kind is None:
                raise ValueError(
                    f"{lineage.agent_id!r} is recorded as departed with no death "
                    "kind, so nothing says whether evidence or age decided it"
                )
            if lineage.agent_id in self._lineage_of:
                raise ValueError(
                    f"{lineage.agent_id!r} is recorded as departed and is also "
                    "alive, or departed twice; one identity is one life"
                )
            self._lineage_of[lineage.agent_id] = lineage
            self._departed.append(lineage.agent_id)

    def _refuse_rates_outside_the_range(
        self, rates: np.ndarray, ids: tuple[AgentId, ...]
    ) -> None:
        """Refuse forgetting factors the configuration does not allow."""
        assert self.learning is not None
        configured = self.learning
        outside = (rates < configured.forgetting_minimum) | (
            rates > configured.forgetting_maximum
        )
        if outside.any():
            strangers = tuple(
                agent_id
                for agent_id, out in zip(ids, outside.tolist(), strict=True)
                if out
            )
            raise ForgettingFactorOutsideItsRange(
                f"{strangers} carry forgetting factors outside the declared "
                f"range [{configured.forgetting_minimum}, "
                f"{configured.forgetting_maximum}]: "
                f"{sorted(set(rates[outside].tolist()))}"
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
        born = tuple(_born(birth, learning) for birth in births)
        founded = cls(
            PopulationSnapshot(
                agents=tuple(one.held for one in born),
                history_capacity=held.history_capacity,
                learning=learning,
                bounds_reached=tuple(
                    one.founding_hold for one in born if one.founding_hold
                ),
            )
        )
        founded._refuse_a_start_outside_the_bound(
            founded._learned,
            np.asarray([one.carried for one in born], dtype=np.bool_),
            founded._ids,
        )
        return founded

    def _refuse_a_start_outside_the_bound(
        self, learned: np.ndarray, carried: np.ndarray, ids: tuple[AgentId, ...]
    ) -> None:
        """Refuse agents whose starting state their bounds would not admit.

        The one refusal every birth path applies, founding and admission
        alike, over the learned rows ``learned`` of the agents ``ids``.
        Applied at a birth and not at a restore: a restored agent resumes a
        state the bound already held, and a restore never re-reads a prior.
        ``carried`` marks the births that inherited a learned state, whose gain
        is checked as well as their weights.
        """
        if self._update is None:
            return
        unfit = self._update.unfit_to_start(learned, carried)
        if not unfit.any():
            return
        weight_norms = self._update.weight_norms(learned).tolist()
        gain_norms = self._update.state_norms(learned).tolist()
        strangers = {
            agent_id: {"weight_norm": weight, "gain_norm": gain}
            for agent_id, weight, gain, out in zip(
                ids, weight_norms, gain_norms, unfit.tolist(), strict=True
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
class _Born:
    """One birth, made ready for whichever path admits it.

    Attributes:
        held: The agent as it enters the population.
        carried: Whether its learned state was inherited, so its gain is
            checked against the state bound as well as its weights.
        founding_hold: The record of its founding gain being held, or
            ``None`` where the bound did not act.
    """

    held: AgentSnapshot
    carried: bool
    founding_hold: BoundReached | None


def _refuse_the_book_identity(agent_id: AgentId) -> None:
    """Refuse an agent that would carry the aggregate book's reserved identity."""
    if agent_id == BOOK_ACCOUNT_ID:
        raise ValueError(
            f"{BOOK_ACCOUNT_ID!r} is reserved for the aggregate book's account and "
            "is no agent's identity: admitted, it would be scored, traded and "
            "reported under the book's own name"
        )


def _born(birth: AgentBirth, learning: LearningConfig | None) -> _Born:
    """The single birth rule: what ``birth`` starts from, and what it records.

    Every path that brings an agent into being — :meth:`founded` and an
    admission into a population already walking — goes through here and
    nowhere else, so there is one starting state, one founding hold and one
    record of it.
    """
    _refuse_the_book_identity(birth.agent_id)
    start = _starting_state(birth, learning)
    return _Born(
        held=AgentSnapshot(
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
            mutation_scale=birth.mutation_scale,
            spawn_seed=birth.spawn_seed,
            learned_state_disposition=(
                LearnedStateDisposition.CARRIED
                if birth.learned_state
                else LearnedStateDisposition.NOTHING_CARRIED
            ),
        ),
        carried=start.carried,
        founding_hold=(
            BoundReached(
                agent_id=birth.agent_id,
                as_of=birth.born_at,
                bound=STATE_BOUND_KEY,
                at_founding=True,
            )
            if start.held_at_founding
            else None
        ),
    )


def _lineage(held: AgentSnapshot) -> Lineage:
    """Where a living agent came from, as its snapshot records it."""
    return Lineage(
        agent_id=held.agent_id,
        parents=held.parents,
        born_at=held.born_at,
        mutation_scale=held.mutation_scale,
        spawn_seed=held.spawn_seed,
        learned_state_disposition=held.learned_state_disposition,
    )


def _a_count(count: int) -> int:
    """``count``, refused if it is not one."""
    if count < 0:
        raise ValueError(f"a count of refusals cannot be negative; it was {count}")
    return count


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
