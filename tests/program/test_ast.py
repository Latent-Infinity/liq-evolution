"""Tests for AST node re-exports from liq-gp."""

from __future__ import annotations

from liq.evolution.program import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import Series


class TestReExports:
    def test_terminal_node_importable(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.name == "close"
        assert node.output_type == Series

    def test_constant_node_importable(self) -> None:
        node = ConstantNode(value=3.14)
        assert node.value == 3.14

    def test_function_node_importable(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(
            "add",
            lambda a, b: a + b,
            input_types=(Series, Series),
            output_type=Series,
        )
        prim = reg.get("add")
        t1 = TerminalNode(name="close", output_type=Series)
        t2 = TerminalNode(name="open", output_type=Series)
        node = FunctionNode(primitive=prim, children=(t1, t2))
        assert node.output_type == Series
        assert node.depth == 1
        assert node.size == 3

    def test_parameterized_node_importable(self) -> None:
        from liq.gp.types import ParamSpec

        reg = PrimitiveRegistry()
        reg.register(
            "sma",
            lambda a, *, period=20: a,
            input_types=(Series,),
            output_type=Series,
            param_specs=[ParamSpec("period", int, 20, 2, 50)],
        )
        prim = reg.get("sma")
        t = TerminalNode(name="close", output_type=Series)
        node = ParameterizedNode(
            primitive=prim,
            children=(t,),
            params={"period": 10},
        )
        assert node.params == {"period": 10}
        assert node.depth == 1

    def test_terminal_is_frozen(self) -> None:
        import pytest

        node = TerminalNode(name="close", output_type=Series)
        with pytest.raises(AttributeError):
            node.name = "open"  # type: ignore[misc]

    def test_program_is_base_type(self) -> None:
        # All node types are subtypes of Program (union)
        t = TerminalNode(name="close", output_type=Series)
        c = ConstantNode(value=1.0)
        assert isinstance(t, Program)
        assert isinstance(c, Program)
