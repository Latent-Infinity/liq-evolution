"""The oracle behind the ecology's import boundaries.

Two mechanisms guard those boundaries and they do not overlap:

* **This module** is the real enforcement for first-party packages. `liq` is a
  namespace package, and `import-linter` rejects any subpackage of one as a
  forbidden module — so `liq.sim`, `liq.gp`, `liq.risk`, `liq.data` and the rest
  are unnameable in a contract and importable from anywhere with nothing to stop
  them. Parsing the AST is what is left. The pattern is not invented here:
  `liq-alert` hit the same wall and closed it the same way, in
  `liq-alert/tests/test_provider_containment.py`.
* **`[tool.importlinter]` in `pyproject.toml`** covers only what it can express
  — a forbidden contract over third-party broker, execution-client and HTTP
  packages.

What this module decides, in eight scans:

1. Domain and use-case code imports no first-party package but the ones a
   port-shaped design leaves it — the core value types and its own package.
   Adapters are deliberately out of scope: they are the one place a provider
   belongs.
2. No broker or execution client is imported anywhere in the ecology package,
   adapters included. This one is absolute: PRD AR-5 asks for a system that
   cannot submit orders by construction rather than by configuration.
3. No broker or execution client is reachable in the installed dependency
   graph either. Source imports are the half a reviewer can see; a dependency
   three edges away is the half that arrives without anyone deciding to add it.
4. Only the population adapter reaches `liq.gp.evolution` — the guard against a
   second evolution engine growing inside the ecology beside the one it is
   supposed to compose with.
5. No class named `*Archive` is defined here — the same guard, for the archive.
6. Only the execution adapter reaches `liq.sim` — the same guard again, for the
   execution model: fill and slippage mechanics are translated in one module,
   and a second importer is an execution assumption spreading inward.

The last two look for evolution machinery defined here rather than composed:

7. No configuration key counts generations. The walk starts at the ecology's
   root configuration and follows every nested configuration class, so a key
   added anywhere beneath it is seen. **Its limit, stated:** it matches key
   *names* only. A generational schedule under another name — a
   ``cycle_length`` that replaces half the population every hundred bars — is
   not caught here; turnover facts carry that, not this scan.
8. Objective directions are the platform's single schema's. No module here
   spells a direction (``"maximize"``/``"minimize"``) as a value, defines a
   loss-form conversion, or imports the direction vocabulary from anywhere but
   :mod:`liq.evolution.fitness.evaluation_schema`. ``liq.gp``'s archive
   declares a vocabulary of the same shape; importing that one is a second
   source, even though it spells the same two words.

Four more guard the descriptors, so that computing them starts neither a
second engine nor a hidden provider reach:

9. Transitive reach. No decision-making module reaches the simulator, the
   evolution engine, the GP stream's descriptor module or its QD orchestrator
   through any chain of first-party imports, not only directly. At full scope,
   no ecology module but the two that hold the provider (``adapters/liq_sim.py``
   and ``adapters/__init__.py``) reaches the simulator, the descriptor module or
   the orchestrator, and none but ``adapters/array_genome.py`` reaches the
   engine. **Its limit, stated:** the closure is static, over module files, and
   a green scan does not say a module never loads the simulator at run time.
   Three run-time reaches are outside it, each pre-existing and not repaired
   here, because each is an ancestor initialiser Python executes rather than
   the importing module's own dependency:

   a. ``liq/evolution/__init__.py`` loads ``liq.sim`` whenever anything under
      ``liq.evolution`` is imported — the GP stream's package initialiser.
   b. ``ecology/adapters/__init__.py`` re-exports ``LiqSimExecutionAdapter``, so
      importing any ``adapters`` submodule executes ``adapters.liq_sim``;
      consumers rely on that re-export.
   c. ``adapters/array_genome.py``'s import of
      ``liq.evolution.fitness.evaluation_schema`` executes
      ``liq/evolution/fitness/__init__.py``, which loads ``liq.sim`` and the GP
      stream's descriptor module.
10. One clip. The descriptor module holds the ecology's only clip, and nothing
    the ecology reaches names the GP stream's clamp helpers. A descriptor out of
    its bounds is an exclusion, never pulled inside them.
11. Record calls. No module but the loop calls the population port's
    ``record``: only the loop offers to the diversity record.
12. The descriptor vocabulary is spelled once, in the descriptor module, and
    read from there everywhere else.

These are static, and static is not enough on its own: a loop with its own
tournament, or an adapter holding the archive while writing to a list beside
it, passes every scan here. The traced half — interception where the adapter
looks the platform's primitives up — is asserted in
`tests/facts/test_boundaries.py`.

Each scan returns a sorted list of human-readable violations, empty when clean.
The assertions live in `tests/facts/test_boundaries.py`; this module only looks.

The near-identical file in `liq-experiments` is a deliberate copy rather than a
shared import: a test helper is not packaged, and the library must not learn
about the consumer to reach one.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import typing
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The tree this file guards. It may legitimately not exist in a repository
#: that has not created it yet, so every scan tolerates its absence — and
#: `test_the_scan_is_not_vacuous` is what stops that tolerance from quietly
#: becoming a green light.
ECOLOGY_ROOT = REPO_ROOT / "src" / "liq" / "evolution" / "ecology"

#: The distribution whose declared dependencies are walked in scan 3.
DISTRIBUTION = "liq-evolution"

#: Everything under this directory is exempt from the first-party containment
#: rule. Adapters exist to translate one library into a port; forbidding the
#: import there would forbid the job.
ADAPTER_DIRECTORY = "adapters"

#: The first-party packages domain and use-case code may import. `liq.core`
#: carries the shared value types every library speaks; `liq.evolution` is this
#: package's own home. Everything else `liq.*` — `liq.sim`, `liq.gp`,
#: `liq.risk`, `liq.data`, `liq.features`, `liq.datasets`, `liq.runner`,
#: `liq.validation` — arrives through a port or not at all.
ALLOWED_FIRST_PARTY = (
    "liq.core",
    "liq.evolution",
)

#: The one named exception, and the only reason it is named: agent output is
#: required to be expressible in the existing signal datamodels rather than a
#: private shape, and a conformance round-trip has to mention them to assert it.
#: The exception is per module, not blanket, and the tuple is empty because
#: nothing requires it yet. Adding a module here is an edit to a declared fact
#: surface, so it arrives attributed rather than as a convenience.
SIGNALS_DATAMODELS = "liq.signals"
SIGNALS_EXCEPTION_MODULES: tuple[str, ...] = ()

#: Brokers and execution clients, by import name. First-party entries first:
#: `liq.live` and `liq.trade` are the paper/live boundary this work never
#: crosses, and they are exactly the entries `import-linter` cannot express.
#: The third-party entries are order-submission SDKs; market-data vendors are
#: deliberately absent, because reading a tape is not trading.
BROKER_IMPORT_PREFIXES = (
    "liq.live",
    "liq.trade",
    "alpaca",
    "alpaca_trade_api",
    "binance",
    "cbpro",
    "ccxt",
    "coinbase",
    "ib_async",
    "ib_insync",
    "ibapi",
    "kiteconnect",
    "oandapyV20",
    "robin_stocks",
    "schwab",
    "tda",
    "tradestation",
)

#: The same set as distribution names, for the dependency-graph walk. Import
#: name and distribution name differ often enough that guessing one from the
#: other is a source of silent misses, so both are written down.
BROKER_DISTRIBUTIONS = frozenset(
    {
        "liq-live",
        "liq-trade",
        "alpaca-py",
        "alpaca-trade-api",
        "binance-connector",
        "cbpro",
        "ccxt",
        "coinbase-advanced-py",
        "ib-async",
        "ib-insync",
        "ibapi",
        "kiteconnect",
        "oandapyv20",
        "python-binance",
        "robin-stocks",
        "schwab-py",
        "tda-api",
        "tradestation",
    }
)

#: The evolution engine the ecology composes with rather than reimplements, and
#: the single module allowed to reach it. Paths are relative to the ecology
#: root. A second importer is a second engine starting.
GP_EVOLUTION_PACKAGE = "liq.gp.evolution"
GP_EVOLUTION_IMPORTERS = ("adapters/array_genome.py",)

#: The execution model the ecology acts through rather than reimplements, and the
#: single module allowed to reach it. A second importer is fill mechanics leaking
#: out of the one place they are translated, which is how an execution assumption
#: ends up inside the loop that selects agents.
EXECUTION_MODEL_PACKAGE = "liq.sim"
EXECUTION_MODEL_IMPORTERS = ("adapters/liq_sim.py",)

#: The archive is composed, never reimplemented, so a class whose name ends
#: this way inside the ecology is the divergence itself rather than a symptom.
ARCHIVE_CLASS_SUFFIX = "Archive"

#: Words that, as a component of a configuration key, make it a count of
#: generations or of passes over history. Matched on ``_``-separated components,
#: so ``n_generations`` and ``max_epochs`` are caught and ``regeneration_rate``
#: is not. Names only — see scan 7's stated limit.
GENERATION_KEY_WORDS = frozenset(
    {"generation", "generations", "epoch", "epochs", "replay", "replays"}
)

#: The platform's single direction schema, the only place the direction
#: vocabulary may be imported from.
DIRECTION_SCHEMA_MODULE = "liq.evolution.fitness.evaluation_schema"
DIRECTION_VOCABULARY = "ObjectiveDirection"

#: The two directions, as the schema spells them. A module here that writes
#: either as a value is declaring or interpreting a direction itself.
DIRECTION_WORDS = frozenset({"maximize", "minimize"})

#: A function whose name contains this converts an objective by its direction,
#: which is the schema's job (``to_loss_form``), not the ecology's.
LOSS_FORM_MARKER = "loss_form"

_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def ecology_files() -> list[Path]:
    """Every module in the ecology package, adapters included."""
    if not ECOLOGY_ROOT.is_dir():
        return []
    return sorted(ECOLOGY_ROOT.rglob("*.py"))


def decision_making_files() -> list[Path]:
    """Every module that decides behaviour — the ecology minus its adapters."""
    return [
        path
        for path in ecology_files()
        if ADAPTER_DIRECTORY not in path.relative_to(ECOLOGY_ROOT).parts
    ]


def imported_modules(path: Path) -> set[str]:
    """The absolute modules `path` imports.

    Relative imports are left out: they cannot reach beyond this package, so
    they can never be the violation being looked for. Imports under
    `TYPE_CHECKING` are counted like any other — a dependency taken only for a
    type annotation is still a dependency on that library's shape.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def covers(module: str, prefix: str) -> bool:
    """Whether `module` is `prefix` or something inside it.

    Written as an equality-or-dotted-prefix test rather than `startswith` so
    that `liq.database` is not read as being inside `liq.data`.
    """
    return module == prefix or module.startswith(f"{prefix}.")


def covered_by_any(module: str, prefixes: Iterable[str]) -> bool:
    return any(covers(module, prefix) for prefix in prefixes)


def _relative(path: Path) -> str:
    return path.relative_to(ECOLOGY_ROOT).as_posix()


def first_party_imports_outside_the_ports() -> list[str]:
    """Scan 1 — first-party containment in domain and use-case code."""
    violations: list[str] = []
    for path in decision_making_files():
        relative = _relative(path)
        allowed = list(ALLOWED_FIRST_PARTY)
        if relative in SIGNALS_EXCEPTION_MODULES:
            allowed.append(SIGNALS_DATAMODELS)
        for module in sorted(imported_modules(path)):
            if covers(module, "liq") and not covered_by_any(module, allowed):
                violations.append(f"{relative} imports {module}")
    return sorted(violations)


def broker_imports() -> list[str]:
    """Scan 2 — brokers and execution clients across the whole package."""
    violations: list[str] = []
    for path in ecology_files():
        for module in sorted(imported_modules(path)):
            if covered_by_any(module, BROKER_IMPORT_PREFIXES):
                violations.append(f"{_relative(path)} imports {module}")
    return sorted(violations)


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _requirements(
    distribution: str, extras: frozenset[str]
) -> list[tuple[str, frozenset[str]]]:
    """The dependencies `distribution` declares, under the extras requested.

    Environment markers other than `extra` are not evaluated. Honouring them
    would make the answer depend on the interpreter and platform the check
    happens to run on; ignoring them keeps the walk deterministic and can only
    make the reachable set larger, which is the safe direction for a check that
    asserts something is *absent*.
    """
    try:
        declared = metadata.distribution(distribution).requires or []
    except metadata.PackageNotFoundError:
        return []

    resolved: list[tuple[str, frozenset[str]]] = []
    for requirement in declared:
        specification, _, marker = requirement.partition(";")
        extra_match = _EXTRA_MARKER.search(marker)
        if extra_match and extra_match.group(1) not in extras:
            continue
        name_match = _REQUIREMENT_NAME.match(specification.strip())
        if name_match is None:
            continue
        name = name_match.group(0)
        requested = ""
        if "[" in specification and "]" in specification:
            requested = specification.split("[", 1)[1].split("]", 1)[0]
        wanted = frozenset(
            part.strip() for part in requested.split(",") if part.strip()
        )
        resolved.append((_normalise(name), wanted))
    return resolved


def reachable_distributions(distribution: str = DISTRIBUTION) -> set[str]:
    """Every distribution transitively reachable from `distribution`.

    Resolved from installed metadata, so it answers the question the acceptance
    criterion asks — what this environment can actually reach — rather than
    what a declaration file says. A dependency that is named but not installed
    is recorded and treated as a leaf; it cannot pull anything further in.
    """
    seen: set[str] = set()
    pending: list[tuple[str, frozenset[str]]] = [
        (_normalise(distribution), frozenset())
    ]
    visited: set[tuple[str, frozenset[str]]] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        seen.add(current[0])
        pending.extend(_requirements(*current))
    return seen


def brokers_in_dependency_graph() -> list[str]:
    """Scan 3 — brokers and execution clients in the resolved dependency graph."""
    reachable = reachable_distributions()
    return sorted(
        f"{DISTRIBUTION} reaches {name} through its installed dependencies"
        for name in reachable & BROKER_DISTRIBUTIONS
    )


def gp_evolution_imports_outside_the_population_adapter() -> list[str]:
    """Scan 4 — the anti-divergence guard on the evolution engine."""
    violations: list[str] = []
    for path in ecology_files():
        relative = _relative(path)
        if relative in GP_EVOLUTION_IMPORTERS:
            continue
        for module in sorted(imported_modules(path)):
            if covers(module, GP_EVOLUTION_PACKAGE):
                violations.append(f"{relative} imports {module}")
    return sorted(violations)


def execution_model_imports_outside_the_execution_adapter() -> list[str]:
    """Scan 6 — the anti-divergence guard on the execution model."""
    violations: list[str] = []
    for path in ecology_files():
        relative = _relative(path)
        if relative in EXECUTION_MODEL_IMPORTERS:
            continue
        for module in sorted(imported_modules(path)):
            if covers(module, EXECUTION_MODEL_PACKAGE):
                violations.append(f"{relative} imports {module}")
    return sorted(violations)


def archive_classes_defined_here() -> list[str]:
    """Scan 5 — the anti-divergence guard on the archive."""
    violations: list[str] = []
    for path in ecology_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith(
                ARCHIVE_CLASS_SUFFIX
            ):
                violations.append(f"{_relative(path)} defines {node.name}")
    return sorted(violations)


def _configuration_classes(annotation: object) -> list[type]:
    """Every dataclass a field annotated ``annotation`` can hold, unwrapped.

    Optional, union, tuple and mapping annotations are opened up, so a nested
    configuration is followed wherever it is held and not only where it is the
    whole annotation.
    """
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return [annotation]
    found: list[type] = []
    for argument in typing.get_args(annotation):
        found.extend(_configuration_classes(argument))
    return found


def generation_count_keys(root: type) -> list[str]:
    """Scan 7 — configuration keys that count generations, under ``root``.

    Every dataclass reachable from ``root`` through its fields' annotations is
    walked once. A key is reported by its path from ``root``, e.g.
    ``EcologyConfig.population.generations``.

    Raises:
        TypeError: If ``root`` is not a dataclass — a walk that cannot start
            would report nothing just as confidently as a clean one.
    """
    if not (isinstance(root, type) and dataclasses.is_dataclass(root)):
        raise TypeError(f"{root!r} is not a configuration class to walk")
    violations: list[str] = []
    pending: list[tuple[type, str]] = [(root, root.__name__)]
    seen: set[type] = set()
    while pending:
        current, path = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        hints = typing.get_type_hints(current)
        for item in dataclasses.fields(current):
            key = f"{path}.{item.name}"
            if GENERATION_KEY_WORDS & set(item.name.lower().split("_")):
                violations.append(key)
            pending.extend(
                (nested, key) for nested in _configuration_classes(hints[item.name])
            )
    return sorted(violations)


def configuration_classes_walked(root: type) -> list[str]:
    """The names of every configuration class scan 7 reaches from ``root``."""
    reached: set[str] = set()
    pending: list[type] = [root]
    while pending:
        current = pending.pop()
        if current.__name__ in reached:
            continue
        reached.add(current.__name__)
        hints = typing.get_type_hints(current)
        for item in dataclasses.fields(current):
            pending.extend(_configuration_classes(hints[item.name]))
    return sorted(reached)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """The ids of every docstring constant in ``tree``, which spell nothing."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def direction_definitions_in(paths: Iterable[Path], *, root: Path) -> list[str]:
    """Scan 8 over ``paths`` — directions spelled, converted or sourced here.

    Three shapes are reported: a direction written as a value (a constant, a
    ``Literal`` vocabulary, a comparison that branches on one); a function
    whose name says it converts to loss form; and the direction vocabulary
    imported from anywhere but the platform's single schema.
    """
    violations: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value in DIRECTION_WORDS
                and id(node) not in docstrings
            ):
                violations.append(
                    f"{relative}:{node.lineno} spells the direction {node.value!r}"
                )
            elif (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and LOSS_FORM_MARKER in node.name.lower()
            ):
                violations.append(
                    f"{relative}:{node.lineno} defines a loss-form conversion "
                    f"{node.name}"
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module != DIRECTION_SCHEMA_MODULE
                and any(alias.name == DIRECTION_VOCABULARY for alias in node.names)
            ):
                violations.append(
                    f"{relative}:{node.lineno} imports {DIRECTION_VOCABULARY} "
                    f"from {node.module}"
                )
    return sorted(violations)


def direction_definitions_here() -> list[str]:
    """Scan 8 — over the ecology package, adapters included."""
    return direction_definitions_in(ecology_files(), root=ECOLOGY_ROOT)


# ---------------------------------------------------------------------------
# Scans 9-12: the descriptors start no second engine and no hidden reach.
# ---------------------------------------------------------------------------

#: The source root first-party modules resolve against, and the ecology's
#: dotted name under it. ``ECOLOGY_ROOT`` is this package's file under this root.
SRC_ROOT = REPO_ROOT / "src"
ECOLOGY_PACKAGE = "liq.evolution.ecology"

#: The first-party package whose imports the transitive closure follows. Any
#: other import is a leaf: it is checked against the forbidden list and never
#: opened.
FIRST_PARTY_PACKAGE = "liq.evolution"

#: What no decision-making module may reach through any chain of first-party
#: imports: the execution model, the evolution engine, the GP stream's
#: descriptor module (which imports the execution model) and its QD
#: orchestrator (which carries a clamp the ecology declines and drives the
#: engine).
TRANSITIVELY_FORBIDDEN = (
    EXECUTION_MODEL_PACKAGE,
    GP_EVOLUTION_PACKAGE,
    "liq.evolution.fitness.behavior_descriptors",
    "liq.evolution.qd.orchestrator",
)

#: The two modules whose job is to hold the provider, relative to the ecology
#: root: the execution adapter, and the adapters' initialiser that re-exports it.
#: Neither is a full-scope source; both are followed like any other file.
REACH_EXEMPT_SOURCES = (*EXECUTION_MODEL_IMPORTERS, "adapters/__init__.py")

#: What no full-scope source may reach. The evolution engine is held separately,
#: against every source but its one permitted importer.
FULL_SCOPE_FORBIDDEN = tuple(
    module for module in TRANSITIVELY_FORBIDDEN if module != GP_EVOLUTION_PACKAGE
)

#: The descriptor module, relative to the ecology root: the home of the
#: ecology's one clip (beta's, on a finite estimate, with the raw value kept)
#: and the one place the descriptor vocabulary is spelled.
DESCRIPTOR_MODULE = "descriptors.py"

#: The GP stream's helpers that pull an out-of-range descriptor inside its
#: bounds. The ecology excludes such a value instead, so neither may be named
#: anywhere the ecology reaches.
CLAMP_HELPERS = frozenset({"_clamp", "_extract_descriptors"})

#: A call to one of these names bounds a value from both sides on its own.
_CLIP_CALLS = frozenset({"clip"})

#: Lower and upper bound calls; one nested in the other bounds from both sides.
_LOWER_BOUNDS = frozenset({"max", "maximum"})
_UPPER_BOUNDS = frozenset({"min", "minimum"})

#: A function whose name contains one of these is a clamp being defined.
_CLAMP_WORDS = ("clip", "clamp")

#: The modules allowed to call the population port's ``record``, relative to
#: the ecology root: the loop, and nothing else.
RECORD_CALLERS = ("steady_state.py",)
RECORD_METHOD = "record"

#: Keywords a run-log line always carries and the population port's
#: ``record(entry)`` cannot accept. A ``.record`` call passing both is a log
#: line, which could not be an offer that succeeds.
RUN_LOG_KEYWORDS = frozenset({"bar_ts", "stage"})


def descriptor_scan_files(ecology_root: Path = ECOLOGY_ROOT) -> list[Path]:
    """Every module scans 10-12 read: the whole ecology, adapters included."""
    if not ecology_root.is_dir():
        return []
    return sorted(ecology_root.rglob("*.py"))


def _decision_making_under(ecology_root: Path) -> list[Path]:
    return [
        path
        for path in descriptor_scan_files(ecology_root)
        if ADAPTER_DIRECTORY not in path.relative_to(ecology_root).parts
    ]


def module_name(path: Path, src_root: Path = SRC_ROOT) -> str:
    """The dotted name of the module in ``path``; a package names its directory."""
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_file(module: str, src_root: Path = SRC_ROOT) -> Path | None:
    """The file ``module`` is defined in under ``src_root``, or ``None``."""
    base = src_root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def named_modules(path: Path, src_root: Path = SRC_ROOT) -> set[str]:
    """Every module an import statement in ``path`` names, relative ones resolved.

    ``import a.b.c`` names ``a.b.c`` and not its ancestors: Python executes
    their initialisers, but they are not this module's own dependency.
    ``from a.b import c`` names ``a.b.c`` when that is a module or package
    under ``src_root``, and names ``a.b`` only when some imported name is not
    such a module (or the import is ``*``): ``a.b``'s initialiser is then the
    dependency; otherwise it is an ancestor Python executes, which is not this
    module's own dependency. A relative import is resolved against the
    module's own package, because a chain through a sibling is still a chain.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module_name(path, src_root).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            submodules = {
                f"{base}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
                and module_file(f"{base}.{alias.name}", src_root) is not None
            }
            found.update(submodules)
            if len(submodules) < len(node.names):
                found.add(base)
    return found


def _followed(module: str, src_root: Path) -> Path | None:
    if not covers(module, FIRST_PARTY_PACKAGE):
        return None
    return module_file(module, src_root)


def transitive_closure(starts: Iterable[Path], src_root: Path = SRC_ROOT) -> set[Path]:
    """Every first-party module file reachable from ``starts``, ``starts`` included."""
    reached: set[Path] = set()
    pending = list(starts)
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        for module in named_modules(current, src_root):
            target = _followed(module, src_root)
            if target is not None and target not in reached:
                pending.append(target)
    return reached


def _first_reach(
    start: Path, prefix: str, src_root: Path
) -> tuple[list[str], str] | None:
    """The shortest chain from ``start`` to an import covered by ``prefix``."""
    queue: list[tuple[Path, list[str]]] = [(start, [module_name(start, src_root)])]
    seen = {start}
    while queue:
        current, chain = queue.pop(0)
        named = sorted(named_modules(current, src_root))
        hit = next((module for module in named if covers(module, prefix)), None)
        if hit is not None:
            return chain, hit
        for module in named:
            target = _followed(module, src_root)
            if target is not None and target not in seen:
                seen.add(target)
                queue.append((target, [*chain, module_name(target, src_root)]))
    return None


def transitive_reach(
    starts: Iterable[Path],
    forbidden: Iterable[str],
    src_root: Path = SRC_ROOT,
) -> list[str]:
    """Each start that reaches a forbidden module, with the chain that carries it.

    A direct import reads ``<start> imports <module>``; one through other
    modules reads ``<start> reaches <module>: <start> -> ... -> <module>``.
    """
    violations: list[str] = []
    for start in starts:
        for prefix in forbidden:
            found = _first_reach(start, prefix, src_root)
            if found is None:
                continue
            chain, module = found
            if len(chain) == 1:
                violations.append(f"{chain[0]} imports {module}")
            else:
                route = " -> ".join([*chain, module])
                violations.append(f"{chain[0]} reaches {module}: {route}")
    return sorted(violations)


def decision_making_transitive_reach(
    ecology_root: Path = ECOLOGY_ROOT, src_root: Path = SRC_ROOT
) -> list[str]:
    """Scan 9 — no decision-making module reaches a provider through any chain."""
    return transitive_reach(
        _decision_making_under(ecology_root), TRANSITIVELY_FORBIDDEN, src_root
    )


def ecology_reach_sources(ecology_root: Path = ECOLOGY_ROOT) -> list[Path]:
    """The full-scope sources: every ecology module but the two holding the provider.

    The exempt modules are not sources, but the closure still follows them
    when a source names them as a dependency.
    """
    return [
        path
        for path in descriptor_scan_files(ecology_root)
        if path.relative_to(ecology_root).as_posix() not in REACH_EXEMPT_SOURCES
    ]


def ecology_transitive_reach(
    ecology_root: Path = ECOLOGY_ROOT, src_root: Path = SRC_ROOT
) -> list[str]:
    """Scan 9, full scope — no ecology module reaches a provider it does not hold.

    Every source is held against the simulator, the GP stream's descriptor
    module and its QD orchestrator; every source but the population adapter
    is held against the evolution engine, the transitive form of scan 4.
    """
    sources = ecology_reach_sources(ecology_root)
    engine_sources = [
        path
        for path in sources
        if path.relative_to(ecology_root).as_posix() not in GP_EVOLUTION_IMPORTERS
    ]
    return sorted(
        transitive_reach(sources, FULL_SCOPE_FORBIDDEN, src_root)
        + transitive_reach(engine_sources, (GP_EVOLUTION_PACKAGE,), src_root)
    )


def _name_of(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _bounds_both_sides(node: ast.Call) -> bool:
    """A lower-bound call around an upper-bound call, or the other way round."""
    outer = _name_of(node.func)
    if outer in _LOWER_BOUNDS:
        inner_kind = _UPPER_BOUNDS
    elif outer in _UPPER_BOUNDS:
        inner_kind = _LOWER_BOUNDS
    else:
        return False
    return any(
        isinstance(argument, ast.Call) and _name_of(argument.func) in inner_kind
        for argument in node.args
    )


def _clip_sites(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            _name_of(node.func) in _CLIP_CALLS or _bounds_both_sides(node)
        ):
            sites.append((node.lineno, "clips"))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
            word in node.name.lower() for word in _CLAMP_WORDS
        ):
            sites.append((node.lineno, f"defines {node.name}"))
    return sorted(sites)


def clips_outside_the_descriptor_module(
    ecology_root: Path = ECOLOGY_ROOT,
) -> list[str]:
    """Scan 10, first half — the descriptor module's clip is the ecology's only one.

    A clip is a call to ``clip`` (``np.clip``, an array's ``.clip``), an upper
    bound nested in a lower one or the reverse (``min(max(...))``,
    ``np.minimum(np.maximum(...))``), or a function defined with ``clip`` or
    ``clamp`` in its name. A one-sided floor, such as a guard against dividing
    by zero, bounds one side only and is not one.
    """
    violations: list[str] = []
    for path in descriptor_scan_files(ecology_root):
        relative = path.relative_to(ecology_root).as_posix()
        sites = _clip_sites(path)
        if relative == DESCRIPTOR_MODULE:
            if len(sites) > 1:
                lines = ", ".join(str(line) for line, _ in sites)
                violations.append(
                    f"{relative} holds {len(sites)} clips (lines {lines}); "
                    "the ruled one is the only one"
                )
            continue
        violations.extend(f"{relative}:{line} {what}" for line, what in sites)
    return sorted(violations)


def _names_used(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.add(node.name)
    return found


def clamp_helpers_reached(
    ecology_root: Path = ECOLOGY_ROOT, src_root: Path = SRC_ROOT
) -> list[str]:
    """Scan 10, second half — no GP-stream clamp helper is named in the ecology's reach.

    Read over the transitive closure of every ecology module, so a helper
    imported, called, copied under its own name, or reached through another
    first-party module is caught.
    """
    reached = transitive_closure(descriptor_scan_files(ecology_root), src_root)
    return sorted(
        f"{module_name(path, src_root)} names {helper}"
        for path in reached
        for helper in sorted(_names_used(path) & CLAMP_HELPERS)
    )


def _run_log_line(node: ast.Call) -> bool:
    return {keyword.arg for keyword in node.keywords} >= RUN_LOG_KEYWORDS


def _record_sites(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    log_lines = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _run_log_line(node)
    }
    sites: list[int] = []
    for node in ast.walk(tree):
        if (
            (
                isinstance(node, ast.Attribute)
                and node.attr == RECORD_METHOD
                and isinstance(node.ctx, ast.Load)
                and id(node) not in log_lines
            )
            or isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == RECORD_METHOD)
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == RECORD_METHOD
                )
            )
        ):
            sites.append(node.lineno)
    return sorted(set(sites))


def record_calls_outside_the_loop(ecology_root: Path = ECOLOGY_ROOT) -> list[str]:
    """Scan 11 — only the loop reaches the population port's ``record``.

    Every way to reach it is reported: ``x.record`` read or called, a bare
    ``record(...)`` after binding it to a name, and ``getattr(x, "record")``.
    A run-log line (``log.record(event, bar_ts=..., stage=...)``) is not one:
    the port's ``record(entry)`` cannot accept those keywords. **Its limit,
    stated:** names only, as every static scan here; a call built from a
    computed string is not seen, and the traced half of the claim is the
    interception of ``record`` by calling module.
    """
    violations: list[str] = []
    for path in descriptor_scan_files(ecology_root):
        relative = path.relative_to(ecology_root).as_posix()
        if relative in RECORD_CALLERS:
            continue
        violations.extend(
            f"{relative}:{line} reaches the population's {RECORD_METHOD}"
            for line in _record_sites(path)
        )
    return sorted(violations)


def _prose_nodes(tree: ast.AST) -> set[int]:
    """The ids of every string standing alone as a statement: prose, never a value."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def descriptor_version_restated(
    vocabulary: str, ecology_root: Path = ECOLOGY_ROOT
) -> list[str]:
    """Scan 12 — the vocabulary is spelled once, in the descriptor module.

    ``vocabulary`` is read by the caller from the descriptor module itself, so
    the scan restates nothing either. Any string value containing it is a
    spelling, including a part of an f-string; a docstring or other standalone
    prose is not.
    """
    violations: list[str] = []
    for path in descriptor_scan_files(ecology_root):
        relative = path.relative_to(ecology_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prose = _prose_nodes(tree)
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and vocabulary in node.value
            and id(node) not in prose
        ]
        allowed = 1 if relative == DESCRIPTOR_MODULE else 0
        if len(lines) > allowed:
            violations.extend(
                f"{relative}:{line} spells {vocabulary!r}"
                for line in sorted(lines)[allowed:]
            )
    return sorted(violations)
