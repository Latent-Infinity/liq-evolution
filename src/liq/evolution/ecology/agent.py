"""What an agent has inherited and what it has learned, turned into what it wants.

**The contract, in full.** At a decision point an agent standardises the
features the view offers against its own prior-bar statistics, multiplies each
feature its genome switches on by *the weight it has learned for that feature*,
adds them, and wants the whole of its evaluation account long in the instrument
if that sum is strictly above its entry gene. Otherwise it wants to be flat. The
wish is stamped at the decision point's own instant. That is the entire rule,
and it is written here rather than in a comment beside the code because a
rebuild that changed it should have to change this paragraph first.

**Which weight the rule reads, and why it is the learned one.** The genome
carries a ``weight.`` gene per feature and the online update carries a
``learned.weight.`` column per feature, and they are different quantities: the
gene multiplies raw feature levels, the learned weight is fitted against the
standardised view to predict what holding a reading earned. The decision reads
the learned one. That is a ruling rather than a preference
(`liq-docs/plans/oracle-learned-weights-in-the-decision-2026-09-27.md`), and the
reason is the experiment: the arm this platform exists to run is *which of the
genome and the learned state crosses a birth boundary*, and a rule that read
only genes would make the learned half consequence-free — an agent that learns
and cannot act. What the genome's per-feature weight is for now, if anything, is
open and is settled by ablation before the genome schema freezes; until then it
is inherited, handed back, and not read here.

**Why the reading is standardised before it is weighted.** The learned weights
are fitted against standardised inputs, so multiplying them by raw levels would
be arithmetically incoherent — the two live in different units. Standardising
first is therefore forced, and it is causal: each feature is scaled by
statistics accumulated from bars strictly *before* the one being processed, and
the bar joins those statistics only afterwards. The ordering was an internal
detail of the update while nothing read what it produced; it now decides every
wish, which is why the check that pins it asserts on the wish and not only on
the number.

**Why the wish is all-or-nothing.** An intent is a wish, not an order, and
deciding *how much* of a wish is warranted belongs to two places that are not
this one: the mandate decides what may be held, and the allocator decides what
share of the book an agent gets. A rule that scaled its own exposure would be
making one of those decisions early, in the one module with no visibility of
either, and the number would then be adjusted twice. Wanting all of it or none
of it keeps every magnitude decision downstream of here, where it can be
reasoned about.

**Why a feature the view withholds contributes nothing.** An agent is a
function of what it was shown. A feature still warming up, or not yet available,
is not a value the agent can be right or wrong about — it is a value it does not
have. Treating it as zero and treating it as an error are both defensible; what
is not defensible is reaching for it, and nothing here can, because only the
names the view offers are ever read.

Once the reading is standardised, "contributes nothing" stops being free and has
to be arranged. A withheld value enters the arithmetic as a raw zero, and a raw
zero standardises to wherever zero sits among the bars behind it — which is a
perfectly definite, perfectly wrong opinion about a value nobody has. So what
the view offers is carried alongside the mask and multiplied into the reading:
a feature nobody broadcast contributes exactly nothing, whatever the agent has
learned about it and whatever the statistics behind it say. *Known narrowness,
recorded rather than papered over:* the withheld zero is still folded into that
feature's own forgetting-weighted moments, because the standardiser's mass is
one column shared by every feature and separating it is a change to what an
agent's learned state **is**. So an agent's scale for a late-arriving feature
carries a decaying memory of the bars it was absent for. That biases a scale; it
cannot reach the wish through a feature the view is withholding, and it belongs
with the learned-state schema question rather than with this wiring.

**Why a population that does not learn cannot be stepped.** The rule above reads
a learned weight, so a population built with no online update has nothing to
read: it has no weight columns, no rate to write them at, and no statistics to
standardise against. Such a population is still constructible, because holding
agents and stepping them are different capabilities and a snapshot of the first
is worth taking. Asking it for a wish is refused rather than answered from a
fallback, because a fallback would be a second decision rule that nobody
declared and that no fact describes.

**The population is the rule; one agent is a view onto it.**
:class:`PopulationState` is a handful of arrays: one for the genes, one for what
each agent has learned, one for what each wants, one for what each holds, one
for what each has seen. A decision point is answered for everyone at once, by
arithmetic over those arrays rather than by a loop over objects, because ten
thousand agents reading sixty-four features is a matrix the machine already
knows how to multiply and is ten thousand attribute lookups it does not.
:class:`Agent` is not a second implementation of that arithmetic — it is a
population of one, addressed by name, for the evaluation account that follows a
single agent. There was a second implementation, and it was deleted rather than
kept in step: two forms of one rule agree until floating-point addition is
performed in two orders, and then a wish sitting exactly on its entry gene falls
either side of it depending on which form asked.

**Why the parts are named rather than bundled.** An agent's parts have different
lifetimes and different owners. What it inherited does not change while it
lives; what it learned changes every bar; what it wants is remade at every
decision point; what it holds is whatever actually traded; where it came from
and what vocabulary it was born under never change at all. The experiment this
platform exists to run turns on *which* of those a birth carries across, and a
representation that held them as one state object could not express the
question, let alone answer it. So each is reachable, and writable, on its own.

**Why a snapshot carries the heritable part too.** The wording that matters is
"restore": what comes back has to be able to go on deciding, and a bundle of
mutable parts with no genes decides nothing. A snapshot here is therefore the
whole of an agent, marked by which parts a resume would change and which it
would not, and restoring one builds a population that stands on its own rather
than one that needs the population it came from to still exist.
"""

from __future__ import annotations

from .agent_contracts import (
    AgentBirth,
    AgentBornUnderAnotherVocabulary,
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    ForgettingFactorOutsideItsRange,
    Lineage,
    NothingWasShown,
    OutcomeFromTheSameBar,
    PopulationDoesNotLearn,
    PopulationSnapshot,
    PopulationStep,
    StorageReport,
)
from .agent_facade import Agent
from .agent_genes import (
    ENTRY_THRESHOLD,
    FORGETTING_FACTOR,
    FULL_EXPOSURE,
    MASK_PREFIX,
    STORAGE_DTYPE,
    SWITCHED_ON_AT,
    WEIGHT_PREFIX,
)
from .population_access import _AccessMixin
from .population_construction import _ConstructionMixin
from .population_decisions import _DecisionsMixin
from .population_observations import _ObservationsMixin
from .population_snapshot import _SnapshotMixin

__all__ = [
    "ENTRY_THRESHOLD",
    "FORGETTING_FACTOR",
    "FULL_EXPOSURE",
    "MASK_PREFIX",
    "STORAGE_DTYPE",
    "SWITCHED_ON_AT",
    "WEIGHT_PREFIX",
    "Agent",
    "AgentBirth",
    "AgentBornUnderAnotherVocabulary",
    "AgentSnapshot",
    "AgentVersions",
    "BoundReached",
    "ForgettingFactorOutsideItsRange",
    "Lineage",
    "NothingWasShown",
    "OutcomeFromTheSameBar",
    "PopulationDoesNotLearn",
    "PopulationSnapshot",
    "PopulationState",
    "PopulationStep",
    "StorageReport",
]


class PopulationState(
    _ConstructionMixin,
    _AccessMixin,
    _DecisionsMixin,
    _ObservationsMixin,
    _SnapshotMixin,
):
    """Every living agent, held as arrays, with each part addressable on its own.

    A population comes into being in one of two ways and there is no third:
    :meth:`founded`, from a set of births, which is how a run starts; and
    :meth:`restored`, from a snapshot, which is how a run resumes. Both go
    through the same construction, so a restored population cannot be a
    differently-shaped thing that happens to answer the same questions.

    The gene columns are laid out so that stepping reads slices rather than
    gathers: every ``mask.`` gene first, in feature order, then every
    ``weight.`` gene in the same order, then whatever else the vocabulary
    carries. A gene the step does not read — a forgetting factor, say — is still
    held, still inherited and still handed back by :meth:`genome`; it simply is
    not one of the two blocks the arithmetic slices.


    """

    pass


# Keep the established module path used by imports and serialized values.
for _public_type in (
    Agent,
    AgentBirth,
    AgentBornUnderAnotherVocabulary,
    AgentSnapshot,
    AgentVersions,
    BoundReached,
    ForgettingFactorOutsideItsRange,
    Lineage,
    NothingWasShown,
    OutcomeFromTheSameBar,
    PopulationDoesNotLearn,
    PopulationSnapshot,
    PopulationStep,
    PopulationState,
    StorageReport,
):
    _public_type.__module__ = __name__
