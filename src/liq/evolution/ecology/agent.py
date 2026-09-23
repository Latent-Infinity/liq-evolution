"""One agent's heritable rule for turning what it was shown into what it wants.

**The contract, in full.** At a decision point an agent reads the features its
genome switches on and the view actually offers, multiplies each by that
feature's weight gene, adds them, and wants the whole of its evaluation account
long in the instrument if that sum is strictly above its entry gene. Otherwise
it wants to be flat. The wish is stamped at the decision point's own instant.
That is the entire rule, and it is written here rather than in a comment beside
the code because a rebuild that changed it should have to change this paragraph
first.

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

**What is deliberately absent.** Nothing here learns, so the same genome and the
same view give the same wish forever. Nothing here knows about other agents, so
there is no allocation. Nothing here knows about the account, costs or the
mandate, so there is no sizing. This is one agent's genome and one rule; the
struct-of-arrays layout a whole population needs is a different shape and
arrives with the population.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    Genome,
    InstrumentId,
    Intent,
)

__all__ = [
    "ENTRY_THRESHOLD",
    "FULL_EXPOSURE",
    "MASK_PREFIX",
    "SWITCHED_ON_AT",
    "WEIGHT_PREFIX",
    "Agent",
]

#: Gene name prefix: whether one feature is read at all. Prefixed rather than
#: positional so a reader asks for the trait it means, and so a genome carrying
#: genes for features this run does not compute is not silently misaligned.
MASK_PREFIX = "mask."

#: Gene name prefix: how much one feature counts once it is read.
WEIGHT_PREFIX = "weight."

#: Gene name: the sum a reading must be strictly above before the agent wants
#: exposure at all.
ENTRY_THRESHOLD = "entry_threshold"

#: A mask gene at or above this switches its feature on. Genes are numbers, so
#: the cut has to be written down somewhere; here, once.
SWITCHED_ON_AT = 0.5

#: What an agent wants when it wants anything: the whole of its own evaluation
#: account, long. See the module docstring for why there is nothing between this
#: and flat.
FULL_EXPOSURE = 1.0

#: What it wants otherwise.
FLAT = 0.0


@dataclass(frozen=True)
class Agent:
    """One agent: an identity and the heritable rule it forms wishes by.

    Attributes:
        agent_id: Who formed the wish. Carried into every intent, so an exposure
            can be traced back to the agent that wanted it rather than to a
            position in a list.
        genome: The heritable part. It does not change during the agent's life;
            whatever the agent learns is held separately, because inheritance
            has to be able to carry one without the other.
    """

    agent_id: AgentId
    genome: Genome

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
            KeyError: If the genome carries no entry gene, or switches on a
                feature it carries no weight for. A genome that cannot be read
                is a defect in whatever produced it, not a condition to handle:
                treating either as a default would score an agent on a rule
                nobody wrote down.
        """
        reading = self._read(window.features.get(instrument, {}))
        wanted = FULL_EXPOSURE if reading > self._entry_gene() else FLAT
        return Intent(
            agent_id=self.agent_id,
            instrument=instrument,
            target_exposure=wanted,
            as_of=window.as_of,
        )

    def _entry_gene(self) -> float:
        """The sum a reading must be strictly above before exposure is wanted."""
        if ENTRY_THRESHOLD not in self.genome.genes:
            raise KeyError(
                f"the genome carries no {ENTRY_THRESHOLD!r} gene, so there is no "
                "reading at which this agent would want exposure and none at which "
                "it would not"
            )
        return self.genome.genes[ENTRY_THRESHOLD]

    def _read(self, view: Mapping[str, float]) -> float:
        """Sum the features this genome switches on and this view offers."""
        total = 0.0
        for name in view:
            if self.genome.genes.get(f"{MASK_PREFIX}{name}", FLAT) < SWITCHED_ON_AT:
                continue
            weight = f"{WEIGHT_PREFIX}{name}"
            if weight not in self.genome.genes:
                raise KeyError(
                    f"the genome switches {name!r} on and carries no {weight!r} "
                    "gene; how much a feature counts is inherited, never defaulted"
                )
            total += self.genome.genes[weight] * view[name]
        return total
