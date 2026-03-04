"""Stage-16 tests for CFG-lite template expansion."""

from __future__ import annotations

import pytest

from liq.evolution.config import PrimitiveConfig
from liq.evolution.program import TerminalNode
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.template_expander import (
    CFGLiteTemplateExpander,
    RegimeExpansionTrace,
)
from liq.gp import BoolSeries, Series, compile_regime_model_to_program


class TestCFGLiteTemplateExpander:
    def test_expansion_produces_valid_regime_model_candidate(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=42)

        model, trace = expander.expand(expert_count=2, include_risk=True)

        assert isinstance(trace, RegimeExpansionTrace)
        assert len(model.experts) == 2
        assert model.risk is not None
        assert len(trace.expert_templates) == 2
        assert trace.risk_template is not None

        program = compile_regime_model_to_program(model, registry)
        assert program.output_type is Series

    def test_expansion_is_deterministic_under_fixed_seed(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander_a = CFGLiteTemplateExpander(registry, seed=7)
        expander_b = CFGLiteTemplateExpander(registry, seed=7)

        _, trace_a = expander_a.expand(expert_count=3, include_risk=True)
        _, trace_b = expander_b.expand(expert_count=3, include_risk=True)

        assert trace_a == trace_b

    def test_invalid_production_paths_are_rejected(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=1)

        with pytest.raises(ValueError, match="unknown template"):
            expander.set_role_probabilities(
                "expert",
                {"not_a_template": 1.0},
            )

        with pytest.raises(ValueError, match="expert_count must be >= 1"):
            expander.expand(expert_count=0)

    def test_template_level_probability_adaptation_is_deterministic(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander_a = CFGLiteTemplateExpander(registry, seed=99)
        expander_b = CFGLiteTemplateExpander(registry, seed=99)

        base = expander_a.get_role_probabilities("expert")
        target_template = sorted(base.keys())[0]

        for expander in (expander_a, expander_b):
            expander.adapt_role_probabilities(
                "expert",
                {target_template: 1.0},
                learning_rate=0.5,
            )

        updated = expander_a.get_role_probabilities("expert")
        assert updated[target_template] > base[target_template]
        assert pytest.approx(sum(updated.values()), rel=1e-9) == 1.0

        _, trace_a = expander_a.expand(expert_count=1, include_risk=False)
        _, trace_b = expander_b.expand(expert_count=1, include_risk=False)
        assert trace_a == trace_b

    def test_probability_api_validation(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=4)
        detector_template = sorted(expander.get_role_probabilities("detector"))[0]

        with pytest.raises(ValueError, match="unknown role"):
            expander.get_role_probabilities("unknown")

        with pytest.raises(ValueError, match="unknown role"):
            expander.set_role_probabilities("unknown", {detector_template: 1.0})

        with pytest.raises(ValueError, match="must be >= 0"):
            expander.set_role_probabilities("detector", {detector_template: -1.0})

        with pytest.raises(ValueError, match="finite numbers"):
            expander.set_role_probabilities(
                "detector",
                {detector_template: float("inf")},
            )

        with pytest.raises(ValueError, match="sum to > 0"):
            expander.set_role_probabilities("detector", {detector_template: 0.0})

    def test_adaptation_validation(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=6)
        expert_template = sorted(expander.get_role_probabilities("expert"))[0]

        with pytest.raises(ValueError, match="learning_rate"):
            expander.adapt_role_probabilities(
                "expert",
                {expert_template: 1.0},
                learning_rate=0.0,
            )

        with pytest.raises(ValueError, match="reward values must be finite numbers"):
            expander.adapt_role_probabilities(
                "expert",
                {expert_template: float("nan")},
                learning_rate=0.5,
            )

    def test_expand_without_risk_sets_trace_to_none(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=8)

        model, trace = expander.expand(expert_count=1, include_risk=False)

        assert model.risk is None
        assert trace.risk_template is None

    def test_build_failures_raise_value_errors(self, monkeypatch) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=10)
        detector_template = sorted(expander.get_role_probabilities("detector"))[0]
        gate_template = sorted(expander.get_role_probabilities("gate"))[0]
        expert_template = sorted(expander.get_role_probabilities("expert"))[0]
        expander.set_role_probabilities("detector", {detector_template: 1.0})
        expander.set_role_probabilities("gate", {gate_template: 1.0})
        expander.set_role_probabilities("expert", {expert_template: 1.0})

        import liq.evolution.template_expander as template_expander_module

        original_builder = template_expander_module.build_strategy_seed

        def _raise_on_detect(name: str, reg):
            if name == detector_template:
                raise RuntimeError("boom")
            return original_builder(name, reg)

        monkeypatch.setattr(
            template_expander_module,
            "build_strategy_seed",
            _raise_on_detect,
        )
        with pytest.raises(ValueError, match="failed to build template"):
            expander.expand(expert_count=1, include_risk=False)

        monkeypatch.setattr(
            template_expander_module,
            "build_strategy_seed",
            lambda _name, _reg: TerminalNode(name="close", output_type=Series),
        )
        with pytest.raises(ValueError, match="must output BoolSeries"):
            expander.expand(expert_count=1, include_risk=False)

        monkeypatch.setattr(
            template_expander_module,
            "build_strategy_seed",
            lambda _name, _reg: TerminalNode(name="detector", output_type=BoolSeries),
        )
        model, trace = expander.expand(expert_count=1, include_risk=False)
        assert model.risk is None
        assert trace.risk_template is None

    def test_empty_role_helpers_and_choose_template_guard(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        expander = CFGLiteTemplateExpander(registry, seed=13)

        expander._templates["risk"] = ()
        assert expander._uniform_role_probabilities("risk") == {}
        assert expander._normalize_probability_map("risk", {}) == {}
        with pytest.raises(
            ValueError,
            match="does not support weighted templates",
        ):
            expander._normalize_probability_map("risk", {"carry_spread_expansion": 1.0})

        expander._probabilities["risk"] = {}
        assert expander.get_role_probabilities("risk") == {}
        expander.adapt_role_probabilities("risk", {"carry_spread_expansion": 1.0})

        with pytest.raises(ValueError, match="no templates available for role 'risk'"):
            expander._choose_template("risk")

    def test_constructor_rejects_empty_non_risk_template_set(self, monkeypatch) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        import liq.evolution.template_expander as template_expander_module

        original = template_expander_module.list_seed_templates_by_role

        def _detector_empty(role):
            if str(role) == "detector":
                return []
            return original(role)

        monkeypatch.setattr(
            template_expander_module,
            "list_seed_templates_by_role",
            _detector_empty,
        )
        with pytest.raises(
            ValueError,
            match="no templates available for role 'detector'",
        ):
            template_expander_module.CFGLiteTemplateExpander(registry, seed=2)
