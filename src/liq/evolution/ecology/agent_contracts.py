from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC

from liq.evolution.ecology.config import LearningConfig
from liq.evolution.ecology.errors import EcologyError
from liq.evolution.ecology.types import (
    AgentId,
    Genome,
    InstrumentId,
    Intent,
    UtcTimestamp,
    _UtcTimestampMixin,
)


class AgentBornUnderAnotherVocabulary(EcologyError):
    """A decision point offered a feature vocabulary some agent was not born under.

    Raised rather than recorded, and raised for the whole step rather than for
    the agent. A gene named ``weight.momentum_20`` means whatever the feature
    vocabulary of the day says ``momentum_20`` is; evaluate an agent under a
    later vocabulary and its weights are being applied to numbers they were
    never selected against, which is not a worse score but a meaningless one.
    Nothing downstream can tell that apart from a real one, so the walk stops
    here instead of producing it.
    """


class ForgettingFactorOutsideItsRange(EcologyError):
    """A genome carries a forgetting factor the configuration does not allow.

    Refused rather than pulled to the nearest edge. A gene silently clipped is
    an agent learning at a rate nobody chose and nothing records, and every
    figure computed from it would be attributed to the gene it was supposed to
    carry. The range is configuration precisely so that being outside it is a
    statable condition rather than a judgement.
    """


class StartingStateOutsideItsBound(EcologyError):
    """An agent would start with a learned state its configuration disallows.

    Checked when an agent is admitted at birth, over whatever it starts from —
    the prior its genome declares, or a state it inherited — and computed
    exactly as the bound applied after each outcome computes it. Refused rather
    than clipped or rescaled: a starting weight silently pulled inside the
    bound is a prior nobody chose, and would make the declared gene decorative.
    A non-finite weight is refused for the same reason, because no norm of it
    is a number the bound could be compared with.

    An *inherited* state is also refused when its gain is outside the state
    bound. Holding it would change what crossed the birth boundary, which is
    the thing inheritance is measured by. A birth carrying nothing is never
    refused on its gain: its founding gain is held and recorded instead.
    """


class PopulationDoesNotLearn(EcologyError):
    """Something asked a population with no configured update to learn.

    Raised rather than ignored. A population built without a learning
    configuration has no columns to write and no rate to write them at, and
    absorbing the call would leave a run that believes it is learning and is
    not.
    """


class OutcomeFromTheSameBar(EcologyError):
    """An outcome was offered at the instant the reading it follows was formed.

    The cheapest look-ahead there is. A wish formed at a bar's close is acted
    on over the bar after it, so the outcome of holding it cannot be known at
    the instant it was formed; an update fed one would be fitting the answer to
    itself. The refusal is here rather than in a reviewer's attention.
    """


class FeatureNotFinite(EcologyError):
    """A feature value the view offered was not a finite number.

    Refused rather than absorbed, and never replaced. Folded into the
    standardiser, one not-a-number makes that feature's moments not-a-number
    for good. Standardised, it makes the reading not-a-number, and the first
    outcome learned from that reading turns every learned weight it touches
    into one. Substituting a value, whether zero, the last one or a mean,
    would be inventing a feature nobody computed. The whole decision point is
    refused before anything moves, so the population is left exactly as it
    was.
    """


class OutcomeNotFinite(EcologyError):
    """An outcome offered to the update was not a finite number.

    Refused rather than absorbed, and never replaced. One not-a-number folded
    into the accumulators makes every weight and gain it touches not-a-number
    from then on: the agent goes flat at every later decision point, no bound
    records anything because no comparison with it is true, and nothing says
    why. Substituting a value — zero, the last one, a mean — would be inventing
    what a bar earned. The whole observation is refused and the population is
    left exactly as it was, so a finite outcome for the same reading is still
    accepted.
    """


class NothingWasShown(EcologyError):
    """An outcome was offered before the population had been shown anything.

    An update needs the reading the outcome belongs to. Without one there is no
    pairing to learn from, and inventing a reading — the last one, or zero —
    would attribute an outcome to a decision point that did not produce it.
    """


@dataclass(frozen=True)
class AgentVersions:
    """What an agent was born under, so it is never silently judged under else.

    Attributes:
        feature_schema_version: Vocabulary the feature names its genes refer to
            belonged to at its birth. A decision point stamped with a different
            one is refused rather than answered.
        model_version: Version of the decision model it was born under, carried
            so a population assembled from more than one of them can be told
            apart afterwards rather than averaged over.
    """

    feature_schema_version: str
    model_version: str


@dataclass(frozen=True)
class Lineage:
    """Where one agent came from.

    Kept beside the agent rather than inside its genome, because it is not
    heritable: an offspring inherits its parents' genes, not their parents.

    Attributes:
        agent_id: The agent this describes.
        parents: Identities the agent was born of, in the order they were given
            to the birth. Empty for a founder, which is a different thing from
            a birth whose parents were lost.
        born_at: The decision instant the agent was born at (UTC), or ``None``
            for a founder, which was alive before the walk began.
    """

    agent_id: AgentId
    parents: tuple[AgentId, ...]
    born_at: UtcTimestamp | None

    def __post_init__(self) -> None:
        """Refuse a birth instant that does not say which zone it is in."""
        if self.born_at is None:
            return
        if self.born_at.tzinfo is None or self.born_at.utcoffset() is None:
            raise ValueError("born_at must be timezone-aware")
        object.__setattr__(self, "born_at", self.born_at.astimezone(UTC))


@dataclass(frozen=True)
class AgentBirth:
    """One agent entering the population, with both its parts stated apart.

    What the agent starts out having learned is required rather than defaulted.
    An empty mapping is a declaration that it starts out having learned nothing;
    an omitted one would be the same population with nobody having said so.

    Attributes:
        agent_id: Identity the agent is known by for the rest of its life.
        genome: The heritable part.
        learned_state: What the agent starts out having learned, by name. The
            names declared here are the names it can ever learn under. In a
            population that learns this is either empty — the agent starts
            from its own genome's priors — or exactly the columns the update
            declares, carried verbatim; anything else is refused.
        feature_schema_version: Feature vocabulary it is born under.
        model_version: Decision model it is born under.
        parents: Who it was born of. Empty for a founder.
        born_at: The decision instant it was born at (UTC), or ``None`` for a
            founder.
    """

    agent_id: AgentId
    genome: Genome
    learned_state: Mapping[str, float]
    feature_schema_version: str
    model_version: str
    parents: tuple[AgentId, ...] = ()
    born_at: UtcTimestamp | None = None


@dataclass(frozen=True)
class AgentSnapshot:
    """One agent, whole, at one instant — everything needed to go on being it.

    The parts a resume would change are ``learned_state``, ``intended``,
    ``realised`` and ``history``; the rest is what the agent is and does not
    move. Both halves are carried, because a snapshot that held only the moving
    half could be written but not restored.

    Attributes:
        agent_id: Who this is.
        genome: The heritable part, as it stood.
        learned_state: What had been learned, by name.
        intended: The exposure most recently wanted, as a signed fraction of the
            agent's own evaluation-account equity.
        realised: The exposure actually held, on the same scale. Not the same
            number as ``intended`` and not derivable from it: a wish that was
            refused, reduced or never reached leaves the two apart, and it is
            the realised side an agent is scored on.
        history: The agent's own retained outcomes, oldest first.
        parents: Who it was born of.
        born_at: The decision instant it was born at (UTC), or ``None``.
        feature_schema_version: Feature vocabulary it was born under.
        model_version: Decision model it was born under.
    """

    agent_id: AgentId
    genome: Genome
    learned_state: Mapping[str, float]
    intended: float
    realised: float
    history: tuple[float, ...]
    parents: tuple[AgentId, ...]
    born_at: UtcTimestamp | None
    feature_schema_version: str
    model_version: str


@dataclass(frozen=True)
class PendingReading(_UtcTimestampMixin):
    """A decision point answered whose outcome is not known yet.

    The one part of a population that belongs to no agent's columns and cannot
    be rebuilt from them. An outcome is learned from by pairing it with the
    standardised reading it followed, and that reading was formed — and the
    statistics it was scaled by were moved past it — at the decision point.
    A snapshot taken before the outcome arrives that did not carry it would
    restore to a population refusing the outcome it is owed, and re-forming
    the reading would scale it by statistics that already include it.

    Attributes:
        as_of: The decision instant the reading was formed at (UTC).
        readings: Each living agent's standardised reading, in the
            population's feature order.
    """

    as_of: UtcTimestamp
    readings: Mapping[AgentId, tuple[float, ...]]

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class PopulationSnapshot:
    """Every living agent at one instant, and the shape they were held in.

    Attributes:
        agents: One entry per living agent, in the order the population holds
            them.
        history_capacity: How many outcomes each agent's history retained. Part
            of the snapshot because restoring under a different ceiling would
            silently change what a later reader of the history sees.
        learning: What the population's online update was turned by, or
            ``None`` for a population that did not learn. Carried for the same
            reason as the ceiling above, and the reason is stronger here: the
            learned columns are meaningless without the constants that produced
            them, so a resume under a different configuration would be a
            different agent wearing the same weights.
        bounds_reached: Every declared bound that had acted, in the order it
            acted. A run held under a bound says so, and a resume that forgot
            it had been held would describe the same run as one that never was.
        pending: The reading awaiting its outcome, or ``None`` at a settled
            boundary. See :class:`PendingReading`.
    """

    agents: tuple[AgentSnapshot, ...]
    history_capacity: int
    learning: LearningConfig | None = None
    bounds_reached: tuple[BoundReached, ...] = ()
    pending: PendingReading | None = None


@dataclass(frozen=True)
class BoundReached(_UtcTimestampMixin):
    """One agent meeting one declared bound, at one instant.

    Recorded *and acted on*: the bound is applied at the same moment this is
    written, so the record describes a run that was held rather than a run that
    was warned about. A guardrail that only logged would leave every figure
    after it computed from a state nobody was watching.

    A bound can also act when an agent is founded. A newly founded agent that
    has learned nothing carries its family's largest gain, so a state bound
    below that gain is met before the first decision point. The gain is held
    then, and the hold is recorded then, marked ``at_founding``. No instant is
    invented for it. It carries the agent's birth instant where the birth
    states one, and ``None`` otherwise, which means "at founding, before any
    decision point".

    Attributes:
        agent_id: Whose state met the bound.
        as_of: The instant the outcome that met it became known (UTC). For a
            founding hold, the agent's birth instant, or ``None`` for a founder
            born at no stated instant.
        bound: Which bound, by the dotted name its configuration declares it
            under, so a reader can look the number up rather than guess it.
        at_founding: Whether the bound acted when the agent was founded rather
            than at an outcome.

    Raises:
        ValueError: If a record that is not a founding hold carries no instant.
    """

    agent_id: AgentId
    as_of: UtcTimestamp | None
    bound: str
    at_founding: bool = False

    def __post_init__(self) -> None:
        if self.as_of is None:
            if not self.at_founding:
                raise ValueError(
                    f"a bound that acted on {self.agent_id!r} at an outcome must "
                    "say when; only a founding hold may carry no instant"
                )
            return
        self._normalize_timestamps("as_of")


@dataclass(frozen=True)
class PopulationStep(_UtcTimestampMixin):
    """What the whole population did at one decision point.

    Both sides are reported, and the pair is the point. ``wanted`` is the wish
    the rule formed here; ``held`` is the exposure each agent was carrying when
    it formed it. A wish equal to what is already held implies no trade and a
    wish different from it implies one, so reporting only the wish would leave
    every reader to reconstruct the distinction from state it cannot see — and
    scoring reads the held side regardless of what was wanted.

    Attributes:
        as_of: The decision instant (UTC).
        instrument: The instrument the wishes are in.
        agent_ids: Who the two rows below belong to, positionally.
        wanted: Exposure each agent wants, aligned with ``agent_ids``.
        held: Exposure each agent was holding, aligned with ``agent_ids``.
    """

    as_of: UtcTimestamp
    instrument: InstrumentId
    agent_ids: tuple[AgentId, ...]
    wanted: tuple[float, ...]
    held: tuple[float, ...]

    def __post_init__(self) -> None:
        self._normalize_timestamps("as_of")

    def intents(self) -> tuple[Intent, ...]:
        """The same wishes as the value type the rest of the ecology speaks.

        Materialised on request rather than produced by the step, because the
        step's own arithmetic is over arrays and a population of ten thousand
        should not pay for ten thousand objects at a boundary that may not want
        them.
        """
        return tuple(
            Intent(
                agent_id=agent_id,
                instrument=self.instrument,
                target_exposure=exposure,
                as_of=self.as_of,
            )
            for agent_id, exposure in zip(self.agent_ids, self.wanted, strict=True)
        )


@dataclass(frozen=True)
class StorageReport:
    """The shape the population is actually held in.

    Reported rather than asserted in a docstring, so the claim that a population
    is arrays and not objects is something a caller can read and a check can
    fail on.

    Attributes:
        agents: How many agents are alive.
        gene_columns: How many genes each agent carries.
        feature_columns: How many features the genes switch on and weight.
        learned_columns: How many named values each agent can learn.
        history_capacity: How many outcomes each agent's history retains.
        dtype: The element type every array is held in.
        contiguous: Whether the gene, learned-state and outcome arrays are each
            one contiguous block rather than a gathered collection of rows.
    """

    agents: int
    gene_columns: int
    feature_columns: int
    learned_columns: int
    history_capacity: int
    dtype: str
    contiguous: bool
