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

What this module decides, in six scans:

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

Each scan returns a sorted list of human-readable violations, empty when clean.
The assertions live in `tests/facts/test_boundaries.py`; this module only looks.

The near-identical file in `liq-experiments` is a deliberate copy rather than a
shared import: a test helper is not packaged, and the library must not learn
about the consumer to reach one.
"""

from __future__ import annotations

import ast
import re
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
