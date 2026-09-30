"""Heritable gene names, thresholds, storage type, and column layout."""

from __future__ import annotations

import numpy as np

from liq.evolution.ecology.types import Genome

MASK_PREFIX = "mask."

#: Gene name prefix: the prior for one feature — the value that agent's learned
#: weight for the feature starts from when it is born without an inherited
#: learned state. In the learned weight's own units: outcome per standard
#: deviation of the feature's standardised view. The decision never reads the
#: gene; it reads the learned weight, which the update moves away from the prior
#: at the agent's own forgetting rate. An agent born carrying inherited learned
#: state starts from that instead, and its own ``weight.`` genes are not read.
WEIGHT_PREFIX = "weight."

#: Gene name: the sum a reading must be strictly above before the agent wants
#: exposure at all.
ENTRY_THRESHOLD = "entry_threshold"

#: Gene name: how fast an agent discards what it learned from older bars. Read
#: by the online update and by nothing else — the wish is formed from the mask
#: and entry genes and the learned weights, and this gene reaches it only
#: through how those weights move — but carried in the genome because it is
#: inherited, which is the whole of what makes one agent adapt faster than
#: another. It also sets how long a prior outlives the evidence against it.
FORGETTING_FACTOR = "forgetting_factor"

#: A mask gene at or above this switches its feature on. Genes are numbers, so
#: the cut has to be written down somewhere; here, once.
SWITCHED_ON_AT = 0.5

#: What an agent wants when it wants anything: the whole of its own evaluation
#: account, long. See the module docstring for why there is nothing between this
#: and flat.
FULL_EXPOSURE = 1.0

#: What it wants otherwise.
FLAT = 0.0

#: The element type every array below is held in. One type, stated once: a
#: population whose genes were single precision and whose outcomes were double
#: would produce results that depend on which array a number came from.
STORAGE_DTYPE = np.float64


def _gene_layout(genome: Genome) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The column order the genes are held in, and the features they refer to.

    Every ``mask.`` gene first in feature order, then every ``weight.`` gene in
    the same order, then the rest sorted. Ordering them so lets the step take
    two slices rather than two gathers, and keeps the two blocks aligned column
    for column with each other and with the feature values.
    """
    features = tuple(
        sorted(
            name.removeprefix(MASK_PREFIX)
            for name in genome.genes
            if name.startswith(MASK_PREFIX)
        )
    )
    masks = tuple(f"{MASK_PREFIX}{name}" for name in features)
    weights = tuple(f"{WEIGHT_PREFIX}{name}" for name in features)
    missing = tuple(name for name in weights if name not in genome.genes)
    if missing:
        raise ValueError(
            f"the population's genomes switch on features they carry no weight "
            f"gene for: {missing}; that gene is the prior an agent's learned "
            "weight for the feature starts from, and it is inherited, never "
            "defaulted"
        )
    if ENTRY_THRESHOLD not in genome.genes:
        raise ValueError(
            f"the population's genomes carry no {ENTRY_THRESHOLD!r} gene, so "
            "there is no reading at which an agent would want exposure and none "
            "at which it would not"
        )
    rest = tuple(sorted(set(genome.genes) - set(masks) - set(weights)))
    return (*masks, *weights, *rest), features
