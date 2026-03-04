"""CFG-lite regime template expansion for deterministic structured priors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from liq.evolution.program import ConstantNode, FunctionNode, Program, TerminalNode
from liq.evolution.regime_model import (
    RegimeDetector,
    RegimeExpert,
    RegimeGate,
    RegimeId,
    RegimeModel,
    RegimeRisk,
    RegimeWeights,
)
from liq.evolution.seed_catalog import (
    SeedTemplateRole,
    build_strategy_seed,
    list_seed_templates_by_role,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series

_RoleName = Literal["detector", "gate", "expert", "risk"]


@dataclass(frozen=True)
class RegimeExpansionTrace:
    """Deterministic trace metadata for a single expansion event."""

    detector_template: str
    gate_template: str
    expert_templates: tuple[str, ...]
    risk_template: str | None


class CFGLiteTemplateExpander:
    """Expand typed regime candidates from role-constrained template productions."""

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        seed: int = 0,
    ) -> None:
        self._registry = registry
        self._rng = np.random.default_rng(seed)
        all_templates: dict[_RoleName, tuple[str, ...]] = {
            "detector": tuple(
                list_seed_templates_by_role(SeedTemplateRole.detector)
            ),
            "gate": tuple(list_seed_templates_by_role(SeedTemplateRole.detector)),
            "expert": tuple(list_seed_templates_by_role(SeedTemplateRole.expert)),
            "risk": tuple(list_seed_templates_by_role(SeedTemplateRole.risk)),
        }
        self._templates: dict[_RoleName, tuple[str, ...]] = {
            role: tuple(
                name for name in templates if self._can_build_template(name)
            )
            for role, templates in all_templates.items()
        }

        for role, templates in self._templates.items():
            if role == "risk":
                continue
            if not templates:
                raise ValueError(f"no templates available for role {role!r}")

        self._probabilities: dict[_RoleName, dict[str, float]] = {
            role: self._uniform_role_probabilities(role)
            for role in self._templates
        }

    def _can_build_template(self, template_name: str) -> bool:
        try:
            program = build_strategy_seed(template_name, self._registry)
        except Exception:
            return False
        return program.output_type is BoolSeries

    def _uniform_role_probabilities(self, role: _RoleName) -> dict[str, float]:
        templates = self._templates[role]
        if not templates:
            return {}
        uniform = 1.0 / float(len(templates))
        return {template: uniform for template in templates}

    def _normalize_role(self, role: SeedTemplateRole | _RoleName | str) -> _RoleName:
        normalized = str(role).strip().lower()
        if normalized not in {"detector", "gate", "expert", "risk"}:
            raise ValueError(f"unknown role {role!r}")
        return normalized  # type: ignore[return-value]

    def _normalize_probability_map(
        self,
        role: _RoleName,
        weights: dict[str, float],
    ) -> dict[str, float]:
        available = set(self._templates[role])
        if not available:
            if weights:
                raise ValueError(f"role {role!r} does not support weighted templates")
            return {}

        merged = {name: 0.0 for name in self._templates[role]}
        for name, value in weights.items():
            if name not in available:
                raise ValueError(f"unknown template {name!r} for role {role!r}")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("template probability weights must be finite numbers")
            if float(value) < 0.0:
                raise ValueError("template probability weights must be >= 0")
            merged[name] = float(value)

        total = sum(merged.values())
        if total <= 0.0:
            raise ValueError("template probability weights must sum to > 0")

        return {name: value / total for name, value in merged.items()}

    def set_role_probabilities(
        self,
        role: SeedTemplateRole | _RoleName | str,
        weights: dict[str, float],
    ) -> None:
        """Set normalized production probabilities for a role."""
        normalized_role = self._normalize_role(role)
        self._probabilities[normalized_role] = self._normalize_probability_map(
            normalized_role,
            weights,
        )

    def adapt_role_probabilities(
        self,
        role: SeedTemplateRole | _RoleName | str,
        rewards: dict[str, float],
        *,
        learning_rate: float = 0.25,
    ) -> None:
        """Adapt role probabilities with deterministic multiplicative updates."""
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        normalized_role = self._normalize_role(role)
        current = self.get_role_probabilities(normalized_role)
        if not current:
            return

        updated: dict[str, float] = {}
        for name, prob in current.items():
            reward = rewards.get(name, 0.0)
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise ValueError("reward values must be finite numbers")
            adjusted = float(prob) * math.exp(learning_rate * float(reward))
            updated[name] = adjusted

        self._probabilities[normalized_role] = self._normalize_probability_map(
            normalized_role,
            updated,
        )

    def get_role_probabilities(
        self,
        role: SeedTemplateRole | _RoleName | str,
    ) -> dict[str, float]:
        """Get a copy of normalized production probabilities for a role."""
        normalized_role = self._normalize_role(role)
        return dict(self._probabilities[normalized_role])

    def _choose_template(self, role: _RoleName) -> str:
        templates = self._templates[role]
        probs = self._probabilities[role]
        if not templates:
            raise ValueError(f"no templates available for role {role!r}")
        ordered = sorted(templates)
        weights = np.asarray([probs[name] for name in ordered], dtype=np.float64)
        choice = int(self._rng.choice(len(ordered), p=weights))
        return ordered[choice]

    def _build_signal_template(self, template_name: str) -> Program:
        try:
            program = build_strategy_seed(template_name, self._registry)
        except Exception as exc:
            raise ValueError(f"failed to build template {template_name!r}") from exc
        if program.output_type is not BoolSeries:
            raise ValueError(f"template {template_name!r} must output BoolSeries")
        return program

    def _lift_signal_to_series(self, signal_program: Program) -> Program:
        primitive = self._registry.get("if_then_else")
        return FunctionNode(
            primitive=primitive,
            children=(
                signal_program,
                TerminalNode(name="close", output_type=Series),
                ConstantNode(value=0.0, output_type=Series),
            ),
        )

    def expand(
        self,
        *,
        expert_count: int = 2,
        include_risk: bool = False,
    ) -> tuple[RegimeModel, RegimeExpansionTrace]:
        """Expand one regime-model candidate from CFG-lite production rules."""
        if expert_count < 1:
            raise ValueError("expert_count must be >= 1")

        detector_template = self._choose_template("detector")
        gate_template = self._choose_template("gate")
        expert_templates = tuple(
            self._choose_template("expert")
            for _ in range(expert_count)
        )
        risk_template = self._choose_template("risk") if include_risk else None

        detector_signal = self._build_signal_template(detector_template)
        gate_signal = self._build_signal_template(gate_template)
        expert_programs = tuple(
            self._lift_signal_to_series(self._build_signal_template(name))
            for name in expert_templates
        )
        risk_program = (
            self._lift_signal_to_series(self._build_signal_template(risk_template))
            if risk_template is not None
            else None
        )

        model = RegimeModel(
            detector=RegimeDetector(RegimeId.trend, detector_signal),
            gate=RegimeGate(RegimeId.trend, gate_signal),
            experts=tuple(
                RegimeExpert(RegimeId.trend, program) for program in expert_programs
            ),
            risk=(
                RegimeRisk(RegimeId.fallback, risk_program)
                if risk_program is not None
                else None
            ),
            weights=RegimeWeights(tuple(1.0 for _ in expert_programs)),
        )
        trace = RegimeExpansionTrace(
            detector_template=detector_template,
            gate_template=gate_template,
            expert_templates=expert_templates,
            risk_template=risk_template,
        )
        return model, trace


__all__ = ["CFGLiteTemplateExpander", "RegimeExpansionTrace"]
