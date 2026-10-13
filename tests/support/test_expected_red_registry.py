"""Every expected-red marker in the ecology test trees is registered, strict and pinned.

A strict ``xfail`` keeps a red that cannot yet be cleared inside the gate
without letting it redden the gate. It is safe only while three things hold: an
unexpected pass fails (``strict=True``), a different failure fails (``raises=``
is a ``pytest.RaisesExc`` carrying the observed red's ``match``), and nobody
can add or loosen a marker without it showing. The last is what this module is
for. :data:`EXPECTED_RED` is a literal, so adding a marker, loosening one or
re-pinning one is an edit to it — an edit to evidence, visible as such — and
the cases below fail when the markers in the trees and the literal disagree.
The mechanism and its conditions are recorded in
``plans/oracle-v3-expected-red-markers-2026-10-01.md``.

Mechanism: an AST scan of the source, not a collection-time read of
``item.iter_markers``. The scan sees every module whether or not this run
collects it, including the consumer repository's evidence trees, which this
repository's test run never collects; and it can tell a node that no longer
exists from one that merely lost its mark. A mark is found wherever pytest
would apply it — a decorator on a function, method or class (directly or
through a module-level name bound to the mark) and the ``marks=`` of a
``pytest.param`` inside a ``parametrize`` decorator — and any other appearance
of ``xfail`` in the trees, a module ``pytestmark`` or an imperative
``pytest.xfail()`` among them, is reported as an unregistered marker in its own
right, because it cannot be registered.

A marker may only override in a specific adapter's registration subclass, or
sit on a standalone evidence node: never in a method a contract base declares,
which every registered adapter would inherit, and never on a class registering
a stand-in, which the contract is held to unmarked.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
CONSUMER = REPOSITORY.parent / "liq-experiments"

#: The trees the scan covers, each as its repository and the tree within it:
#: every test module here, and the consumer's evidence trees. The consumer's
#: other trees hold strict ``xfail`` cases of research history that are not
#: ecology evidence and are not scanned. A tree that does not exist yet holds
#: no marker; the two consumer trees that do exist are required to.
SCANNED_TREES = (
    (REPOSITORY, "tests"),
    (CONSUMER, "tests/facts"),
    (CONSUMER, "tests/support"),
    (CONSUMER, "tests/ecology"),
)
REQUIRED_TREES = (CONSUMER / "tests" / "facts", CONSUMER / "tests" / "support")

#: Every marked node, and the exception type and message its marker is pinned to.
#: A node is named as pytest names it, prefixed by its repository.
EXPECTED_RED: dict[str, tuple[str, str]] = {}


@dataclass(frozen=True)
class Marker:
    """One ``xfail`` mark as the source declares it.

    ``raises`` and ``match`` are set only when ``raises=`` is a
    ``RaisesExc(<type>, match=<literal>)``; anything else leaves them ``None``.
    ``owner`` is the class a marked method is declared in.
    """

    node: str
    where: str
    strict: bool
    raises: str | None
    match: str | None
    owner: str | None = None


@dataclass(frozen=True)
class Scan:
    """What the trees declare: their marks, their nodes and their class layout."""

    markers: tuple[Marker, ...]
    nodes: frozenset[str]
    bases: frozenset[str]
    null_registrations: frozenset[str]


def _source(path: Path) -> str:
    """The text of ``path``; the one place the scan reads a file."""
    return path.read_text(encoding="utf-8")


def _is_xfail_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "xfail"


def _callee_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


class _ModuleScan:
    """The marks, nodes and classes of one module."""

    def __init__(self, tree: ast.Module, prefix: str) -> None:
        self.prefix = prefix
        self.bindings: dict[str, ast.expr] = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target, value = statement.target, statement.value
            else:
                continue
            if isinstance(target, ast.Name):
                self.bindings[target.id] = value
        self.consumed: set[int] = set()
        self.markers: list[Marker] = []
        self.nodes: set[str] = set()
        self.bases: set[str] = set()
        self.null_registrations: set[str] = set()
        self._walk(tree.body, owner=None)
        for found in ast.walk(tree):
            if _is_xfail_attribute(found) and id(found) not in self.consumed:
                self.markers.append(
                    Marker(
                        node=f"{prefix}:{found.lineno} (xfail outside any decorator "
                        "or parameter)",
                        where=f"{prefix}:{found.lineno}",
                        strict=False,
                        raises=None,
                        match=None,
                    )
                )

    def _resolve(self, expression: ast.expr) -> ast.expr:
        """``expression``, or what a module-level name it is bound to is bound to."""
        seen: set[str] = set()
        while isinstance(expression, ast.Name) and expression.id in self.bindings:
            if expression.id in seen:
                break
            seen.add(expression.id)
            expression = self.bindings[expression.id]
        return expression

    def _string(self, expression: ast.expr) -> str | None:
        resolved = self._resolve(expression)
        if isinstance(resolved, ast.Constant) and isinstance(resolved.value, str):
            return resolved.value
        if isinstance(resolved, ast.JoinedStr):
            return None
        if isinstance(resolved, ast.BinOp) and isinstance(resolved.op, ast.Add):
            left, right = self._string(resolved.left), self._string(resolved.right)
            if left is not None and right is not None:
                return left + right
        return None

    def _marker(self, mark: ast.expr, node: str, owner: str | None) -> Marker | None:
        """A :class:`Marker` if ``mark`` is an ``xfail`` mark, else ``None``."""
        resolved = self._resolve(mark)
        call = resolved if isinstance(resolved, ast.Call) else None
        attribute = call.func if call is not None else resolved
        if not _is_xfail_attribute(attribute):
            return None
        self.consumed.add(id(attribute))
        keywords = {k.arg: k.value for k in call.keywords} if call is not None else {}
        strict = keywords.get("strict")
        raises, match = None, None
        pin = keywords.get("raises")
        if (
            isinstance(pin, ast.Call)
            and _callee_name(pin.func) == "RaisesExc"
            and len(pin.args) == 1
        ):
            raises = _callee_name(pin.args[0])
            pattern = {k.arg: k.value for k in pin.keywords}.get("match")
            match = self._string(pattern) if pattern is not None else None
            if raises is None or match is None:
                raises, match = None, None
        return Marker(
            node=node,
            where=f"{self.prefix}:{attribute.lineno}",
            strict=isinstance(strict, ast.Constant) and strict.value is True,
            raises=raises,
            match=match,
            owner=owner,
        )

    def _marks_of(self, expression: ast.expr) -> list[ast.expr]:
        resolved = self._resolve(expression)
        if isinstance(resolved, ast.List | ast.Tuple):
            return list(resolved.elts)
        return [expression]

    def _parameter_marks(
        self, decorator: ast.expr, node: str, owner: str | None
    ) -> Iterator[Marker]:
        """The marks a ``parametrize`` decorator's ``pytest.param`` entries carry."""
        if not (
            isinstance(decorator, ast.Call)
            and _callee_name(decorator.func) == "parametrize"
            and len(decorator.args) >= 2
        ):
            return
        values = self._resolve(decorator.args[1])
        if not isinstance(values, ast.List | ast.Tuple):
            return
        for value in values.elts:
            if not (
                isinstance(value, ast.Call) and _callee_name(value.func) == "param"
            ):
                continue
            keywords = {k.arg: k.value for k in value.keywords}
            if "marks" not in keywords:
                continue
            given = keywords.get("id")
            label = self._string(given) if given is not None else None
            if label is None and len(value.args) == 1:
                label = self._string(value.args[0])
            for mark in self._marks_of(keywords["marks"]):
                found = self._marker(mark, f"{node}[{label or '?'}]", owner)
                if found is not None:
                    yield found

    def _decorated(
        self,
        definition: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        node: str,
        owner: str | None,
    ) -> None:
        for decorator in definition.decorator_list:
            found = self._marker(decorator, node, owner)
            if found is not None:
                self.markers.append(found)
            self.markers.extend(self._parameter_marks(decorator, node, owner))

    def _walk(
        self, body: list[ast.stmt], owner: str | None, scope: str | None = None
    ) -> None:
        """Every function and class in ``body``; ``scope`` is the classes around it."""
        for statement in body:
            if not isinstance(
                statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                continue
            qualified = f"{scope}::{statement.name}" if scope else statement.name
            node = f"{self.prefix}::{qualified}"
            self.nodes.add(node)
            if not isinstance(statement, ast.ClassDef):
                self._decorated(statement, node, owner)
                continue
            # A mark on a class is the class's own: it is placed where the
            # class stands, a base or a stand-in's registration included.
            self._decorated(statement, node, statement.name)
            for base in statement.bases:
                name = _callee_name(base)
                if name is not None:
                    self.bases.add(name)
            for member in statement.body:
                targets = (
                    member.targets
                    if isinstance(member, ast.Assign)
                    else [member.target]
                    if isinstance(member, ast.AnnAssign) and member.value is not None
                    else []
                )
                if any(
                    isinstance(t, ast.Name) and t.id == "adapter_factory"
                    for t in targets
                ) and "Null" in ast.unparse(member):
                    self.null_registrations.add(statement.name)
            self._walk(statement.body, owner=statement.name, scope=qualified)


def scan_source(source: str, prefix: str) -> Scan:
    """The marks, nodes and class layout one module's ``source`` declares."""
    module = _ModuleScan(ast.parse(source), prefix)
    return Scan(
        markers=tuple(module.markers),
        nodes=frozenset(module.nodes),
        bases=frozenset(module.bases),
        null_registrations=frozenset(module.null_registrations),
    )


def scan_trees() -> Scan:
    """Every module under :data:`SCANNED_TREES`, scanned and merged."""
    markers: list[Marker] = []
    nodes: set[str] = set()
    bases: set[str] = set()
    null_registrations: set[str] = set()
    for repository, tree in SCANNED_TREES:
        for path in sorted((repository / tree).rglob("*.py")):
            prefix = f"{repository.name}/{path.relative_to(repository).as_posix()}"
            found = scan_source(_source(path), prefix)
            markers.extend(found.markers)
            nodes.update(found.nodes)
            bases.update(found.bases)
            null_registrations.update(found.null_registrations)
    return Scan(
        markers=tuple(markers),
        nodes=frozenset(nodes),
        bases=frozenset(bases),
        null_registrations=frozenset(null_registrations),
    )


# ---------------------------------------------------------------------------
# The trees against the registry.
# ---------------------------------------------------------------------------


def test_the_consumer_evidence_trees_are_where_the_scan_looks() -> None:
    """A scan of trees that are not there would find no marker and say nothing."""
    missing = [str(tree) for tree in REQUIRED_TREES if not tree.is_dir()]
    assert missing == [], f"the scan cannot see {missing}"


def test_every_xfail_marker_is_registered() -> None:
    """A marker the registry does not name is a red quietened without a record."""
    unregistered = [
        f"{marker.node} ({marker.where})"
        for marker in scan_trees().markers
        if marker.node not in EXPECTED_RED
    ]
    assert unregistered == [], f"xfail markers not in EXPECTED_RED: {unregistered}"


def test_every_registered_marker_is_strict() -> None:
    """Without ``strict=True`` an implementation that makes the case true passes unseen."""
    loose = [
        f"{marker.node} ({marker.where})"
        for marker in scan_trees().markers
        if marker.node in EXPECTED_RED and not marker.strict
    ]
    assert loose == [], f"registered xfail markers not strict=True: {loose}"


def test_every_registered_marker_is_pinned_to_its_registered_message() -> None:
    """A type alone admits any failure of that type; the pin is the registered one."""
    unpinned = [
        f"{marker.node} ({marker.where}): raises={marker.raises!r}, "
        f"match={marker.match!r}; registered {EXPECTED_RED[marker.node]!r}"
        for marker in scan_trees().markers
        if marker.node in EXPECTED_RED
        and (marker.raises, marker.match) != EXPECTED_RED[marker.node]
    ]
    assert unpinned == [], (
        f"registered xfail markers not pinned by RaisesExc(<type>, match=<the "
        f"registered pattern>): {unpinned}"
    )


def test_no_registry_entry_is_stale() -> None:
    """An entry whose node lost its mark, or no longer exists, is removed with it."""
    scan = scan_trees()
    marked = {marker.node for marker in scan.markers}
    stale = [
        f"{node}: "
        + (
            "no longer carries an xfail mark"
            if node.split("[")[0] in scan.nodes
            else "no longer exists"
        )
        for node in EXPECTED_RED
        if node not in marked
    ]
    assert stale == [], f"stale EXPECTED_RED entries: {stale}"


def test_no_marker_sits_in_a_base_or_on_a_stand_in_registration() -> None:
    """A base's marker is inherited by every adapter; a stand-in is held unmarked."""
    scan = scan_trees()
    misplaced = [
        f"{marker.node} ({marker.where}): {marker.owner} "
        + ("is a base" if marker.owner in scan.bases else "registers a stand-in")
        for marker in scan.markers
        if marker.owner is not None
        and (marker.owner in scan.bases or marker.owner in scan.null_registrations)
    ]
    assert misplaced == [], f"xfail markers outside the places allowed: {misplaced}"


# ---------------------------------------------------------------------------
# The scan itself, exercised on declared sources rather than trusted.
# ---------------------------------------------------------------------------

PINNED = (
    "pytest.mark.xfail(strict=True, "
    "raises=pytest.RaisesExc(AssertionError, match=r'^0 of 1'), reason='r')"
)


def _markers(source: str) -> dict[str, Marker]:
    return {m.node: m for m in scan_source(source, "m.py").markers}


def test_the_scan_reads_a_pinned_decorator() -> None:
    """Strictness, the pinned type and the pattern, read off the source."""
    found = _markers(f"import pytest\n@{PINNED}\ndef test_a(): ...\n")
    assert found == {
        "m.py::test_a": Marker(
            node="m.py::test_a",
            where="m.py:2",
            strict=True,
            raises="AssertionError",
            match="^0 of 1",
        )
    }


def test_the_scan_follows_a_bound_name_and_a_concatenated_pattern() -> None:
    """A mark bound to a module name is applied wherever the name decorates."""
    source = (
        "import pytest\n"
        "P = 'a' 'b'\n"
        "Q = P + 'c'\n"
        "MARK = pytest.mark.xfail(strict=True, "
        "raises=pytest.RaisesExc(AttributeError, match=Q))\n"
        "@MARK\ndef test_a(): ...\n"
        "@MARK\ndef test_b(): ...\n"
    )
    found = _markers(source)
    assert set(found) == {"m.py::test_a", "m.py::test_b"}
    assert {(m.raises, m.match) for m in found.values()} == {("AttributeError", "abc")}


def test_the_scan_reads_parameter_marks_by_their_id() -> None:
    """Each ``pytest.param`` carrying a mark is a node of its own."""
    source = (
        "import pytest\n"
        "ONE = 'one'\n"
        "class TestX:\n"
        "    @pytest.mark.parametrize('s', [\n"
        f"        pytest.param(ONE, marks={PINNED}),\n"
        f"        pytest.param(2, id='two', marks=[{PINNED}]),\n"
        "        pytest.param('three'),\n"
        "        'four',\n"
        "    ])\n"
        "    def test_a(self, s): ...\n"
    )
    found = _markers(source)
    assert set(found) == {"m.py::TestX::test_a[one]", "m.py::TestX::test_a[two]"}
    assert {m.owner for m in found.values()} == {"TestX"}


@pytest.mark.parametrize(
    ("mark", "strict", "raises"),
    [
        ("pytest.mark.xfail", False, None),
        ("pytest.mark.xfail(strict=False, raises=AssertionError)", False, None),
        ("pytest.mark.xfail(strict=True, raises=AssertionError)", True, None),
        (
            "pytest.mark.xfail(strict=True, raises=pytest.RaisesExc(AssertionError))",
            True,
            None,
        ),
        (
            "pytest.mark.xfail(strict=True, "
            "raises=pytest.RaisesExc(AssertionError, match=f'{x}'))",
            True,
            None,
        ),
    ],
)
def test_the_scan_reads_a_loose_mark_as_loose(
    mark: str, strict: bool, raises: str | None
) -> None:
    """Bare, non-strict, type-only and computed pins are each read as unpinned."""
    (found,) = _markers(f"import pytest\n@{mark}\ndef test_a(): ...\n").values()
    assert (found.strict, found.raises, found.match) == (strict, raises, None)


def test_the_scan_reports_an_xfail_it_cannot_place() -> None:
    """A module mark and an imperative call are reported where they stand."""
    source = (
        "import pytest\n"
        "pytestmark = [pytest.mark.xfail(strict=True)]\n"
        "def test_a():\n"
        "    pytest.xfail('later')\n"
        "UNUSED = pytest.mark.xfail(strict=True)\n"
    )
    assert sorted(m.where for m in scan_source(source, "m.py").markers) == [
        "m.py:2",
        "m.py:4",
        "m.py:5",
    ]


def test_the_scan_names_bases_and_stand_in_registrations() -> None:
    """The class layout the placement case reads."""
    source = (
        "class Contract: ...\n"
        "class TestNull(Contract):\n"
        "    adapter_factory = staticmethod(NullThing)\n"
        "class TestReal(mod.Contract):\n"
        "    adapter_factory = staticmethod(real_thing)\n"
    )
    scan = scan_source(source, "m.py")
    assert scan.bases == {"Contract"}
    assert scan.null_registrations == {"TestNull"}
    assert {"m.py::Contract", "m.py::TestNull", "m.py::TestReal"} <= scan.nodes


def test_the_scan_places_a_class_mark_on_the_class_it_marks() -> None:
    """A mark on a stand-in's registration is owned by that registration."""
    source = (
        "import pytest\n"
        "class Contract: ...\n"
        f"@{PINNED}\n"
        "class TestNull(Contract):\n"
        "    adapter_factory: object = staticmethod(NullThing)\n"
        "    class TestInner:\n"
        "        def test_a(self): ...\n"
    )
    scan = scan_source(source, "m.py")
    (marker,) = scan.markers
    assert (marker.node, marker.owner) == ("m.py::TestNull", "TestNull")
    assert scan.null_registrations == {"TestNull"}
    assert "m.py::TestNull::TestInner::test_a" in scan.nodes
