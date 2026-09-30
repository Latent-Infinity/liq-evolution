"""Population decisions behavior."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from liq.evolution.ecology import learning as online
from liq.evolution.ecology.types import (
    AgentId,
    BarWindow,
    InstrumentId,
    UtcTimestamp,
)

from .agent_contracts import (
    NothingWasShown,
    PopulationDoesNotLearn,
    PopulationStep,
)
from .agent_genes import (
    FLAT,
    FULL_EXPOSURE,
    STORAGE_DTYPE,
    SWITCHED_ON_AT,
)
from .population_state_fields import _PopulationStateFields


class _DecisionsMixin(_PopulationStateFields):
    def step(self, window: BarWindow, instrument: InstrumentId) -> PopulationStep:
        """Answer one decision point for every agent at once.

        The whole population's reading is one array product: each agent's
        learned weights against the standardised feature values the window
        offers, under the mask its genome carries, in the layout's own column
        order. A feature the view withholds contributes nothing, for the reason
        the module docstring gives, and a name the view offers that no gene
        refers to is not read at all.

        The scaling happens *before* the wish and is what the wish is formed
        from, which is why the standardiser is run here rather than after. It
        remains causal — each value is scaled by statistics accumulated from
        earlier bars and the bar joins them only afterwards — so nothing the
        wish consults depends on the bar the wish is formed at beyond that
        bar's own broadcast view.

        Args:
            window: The decision point, carrying the broadcast view.
            instrument: The instrument the wishes are formed in.

        Returns:
            PopulationStep: What every agent wants and what each was holding.

        Raises:
            PopulationDoesNotLearn: If this population has no online update,
                and therefore no learned weight for the rule to read.
            AgentBornUnderAnotherVocabulary: If the decision point's feature
                vocabulary is not the one every living agent was born under.
        """
        self._refuse_if_nothing_learns()
        assert self._update is not None
        self._refuse_a_vocabulary_nobody_was_born_under(window)
        view = window.features.get(instrument, {})
        readable = tuple(name in view for name in self.feature_names)
        # Asked for by name only where the view says it has one. A broadcast
        # view refuses a feature that is not readable yet — that refusal is the
        # as-of guard and it is raised, not returned — so `get` with a default
        # does not quietly produce a zero here: it produces the refusal. The
        # membership test is the only way to ask a view what it is offering
        # without asking it for something it is withholding.
        values = np.fromiter(
            (
                view[name] if offered else FLAT
                for name, offered in zip(self.feature_names, readable, strict=True)
            ),
            dtype=STORAGE_DTYPE,
            count=self._features,
        )
        offered = np.asarray(readable, dtype=np.bool_)
        standardised = self._show(window.as_of, values)
        switched_on = self._genes[:, : self._features] >= SWITCHED_ON_AT
        learned = self._update.weights(self._learned)
        reading = np.einsum("af,af->a", switched_on * offered * learned, standardised)
        wanted = np.where(reading > self._genes[:, self._entry_at], FULL_EXPOSURE, FLAT)
        held = tuple(self._realised.tolist())
        self._intended[:] = wanted
        return PopulationStep(
            as_of=window.as_of,
            instrument=instrument,
            agent_ids=self._ids,
            wanted=tuple(wanted.tolist()),
            held=held,
        )

    def _show(self, as_of: UtcTimestamp, values: np.ndarray) -> np.ndarray:
        """Scale what the wish and the update both consume, and put it aside.

        Nothing is learned here. The reading is scaled by each agent's own
        prior-bar statistics and put aside with the instant it belongs to, so
        that when the outcome arrives it can be paired with the reading that
        produced it rather than with whatever is in front of the population by
        then. The same scaled reading is handed back, because the wish is
        formed from it: one standardisation, consumed twice, so the value a
        decision was taken on and the value an outcome is attributed to cannot
        be two different numbers.
        """
        assert self._update is not None
        raw = np.broadcast_to(values, (len(self._ids), self._features))
        standardised = self._update.standardise(self._learned, raw, self._forgetting())
        self._shown = (as_of, standardised)
        return standardised

    def standardised(self, agent_id: AgentId) -> tuple[float, ...]:
        """Return what the update would consume for ``agent_id``, in feature order.

        The scaled reading, not the raw one: this is the value the estimator
        actually sees, which is the thing a causality claim has to be made
        about. Available only between a decision point and the outcome that
        follows it, because outside that pair there is nothing it would be a
        reading of.

        Raises:
            PopulationDoesNotLearn: If nothing here standardises anything.
            NothingWasShown: If no decision point has been answered yet.
        """
        self._refuse_if_nothing_learns()
        if self._shown is None:
            raise NothingWasShown(
                "no decision point has been answered, so there is no reading "
                "the update would consume"
            )
        return tuple(self._shown[1][self._row(agent_id)].tolist())

    def learned_weights(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return what the online update has learned for ``agent_id``, by feature.

        A named slice of the learned state rather than a second copy of it: the
        weights live in the same array a checkpoint captures, and this reads
        them under the feature names they belong to instead of under the column
        names they are stored beside.

        Raises:
            PopulationDoesNotLearn: If this population has no online update.
        """
        self._refuse_if_nothing_learns()
        assert self._update is not None
        row = self._update.weights(self._learned)[self._row(agent_id)]
        return MappingProxyType(
            dict(zip(self.feature_names, row.tolist(), strict=True))
        )

    def estimator_state(self, agent_id: AgentId) -> Mapping[str, float]:
        """Return the gain the update would apply to ``agent_id``'s next surprise.

        This, and not the learned weights, is what a state bound watches: the
        failure a recursive estimator with exponential forgetting has is that
        its gain grows without limit when the input stops carrying information,
        and a large weight is the symptom rather than the disease. The
        forgetting-weighted counters the gain is computed from are learned state
        and are handed back by :meth:`learned_state`; they are bounded by
        arithmetic rather than by a guard, so they are not reported here.

        Raises:
            PopulationDoesNotLearn: If this population has no online update.
        """
        self._refuse_if_nothing_learns()
        assert self._update is not None
        row = self._update.gains(self._learned)[self._row(agent_id)]
        return MappingProxyType(
            dict(
                zip(
                    online.gain_names(self.feature_names),
                    row.tolist(),
                    strict=True,
                )
            )
        )

    def _forgetting(self) -> np.ndarray:
        """Every agent's forgetting factor, read off the genes it inherited."""
        assert self._forgetting_at is not None
        return self._genes[:, self._forgetting_at]

    def _refuse_if_nothing_learns(self) -> None:
        """Refuse a question only a population with an update could answer."""
        if self._update is None:
            raise PopulationDoesNotLearn(
                "this population was built without a learning configuration, "
                "so it has no update, no columns to write and no rate to write "
                "them at"
            )
