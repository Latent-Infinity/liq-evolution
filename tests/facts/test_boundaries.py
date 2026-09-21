"""The ecology's import boundaries, asserted.

Evidence for two claims: that nothing but a port carries a provider into the
code that decides behaviour, and that no broker or execution client is reachable
from this package at all — by import or by dependency edge.

The scanning lives in `tests/support/boundary_scan.py`. Only the assertions and
the failure messages live here, so that a change to *what is scanned* and a
change to *what is claimed* are separate edits to separate files.
"""

from __future__ import annotations

from tests.support import boundary_scan


def test_domain_reaches_no_provider_package() -> None:
    """Domain and use-case code speaks value types and ports, nothing else."""
    violations = boundary_scan.first_party_imports_outside_the_ports()
    assert violations == [], (
        f"a provider package reached code that decides behaviour: {violations}. "
        "What is needed arrives through one of the five ports; if no port "
        "expresses it, widen a port or write an adapter, rather than importing "
        "the library here."
    )


def test_no_broker_reachable() -> None:
    """No order-submitting library is importable or installed under this tree.

    Both halves are asserted together because the claim is one claim. A source
    scan alone would miss a broker arriving as somebody else's dependency, and
    a graph walk alone would miss an import of something already installed for
    an unrelated reason.
    """
    imported = boundary_scan.broker_imports()
    assert imported == [], (
        f"a broker or execution client is imported by the ecology: {imported}. "
        "This system produces a weight vector in an artifact and nothing that "
        "can act on it; that has to hold by construction, not by configuration."
    )

    reachable = boundary_scan.brokers_in_dependency_graph()
    assert reachable == [], (
        f"a broker or execution client is installed under this package: "
        f"{reachable}. Nothing imports it today, which is exactly how the next "
        "person finds it already there and uses it."
    )


def test_only_the_population_adapter_reaches_the_evolution_engine() -> None:
    """The ecology composes with one evolution engine and never grows another."""
    violations = boundary_scan.gp_evolution_imports_outside_the_population_adapter()
    assert violations == [], (
        f"the evolution engine is reached outside the population adapter: "
        f"{violations}. Selection, mutation and archive primitives are consumed "
        "through that one module; a second importer is a second engine."
    )


def test_no_archive_is_defined_here() -> None:
    """The archive is composed as-is, so there is nothing here to define."""
    violations = boundary_scan.archive_classes_defined_here()
    assert violations == [], (
        f"an archive is defined inside the ecology: {violations}. The existing "
        "archive is consumed without forking, shadowing or subclassing it."
    )


def test_the_scan_is_not_vacuous() -> None:
    """Guard against every assertion above passing over an empty file list."""
    if not boundary_scan.ECOLOGY_ROOT.is_dir():
        assert boundary_scan.ecology_files() == []
        assert boundary_scan.decision_making_files() == []
        return
    assert boundary_scan.ecology_files()
    assert boundary_scan.decision_making_files()


def test_the_dependency_walk_is_not_vacuous() -> None:
    """A walk that resolved nothing would report no broker just as confidently."""
    reachable = boundary_scan.reachable_distributions()
    assert boundary_scan.DISTRIBUTION in reachable
    assert "liq-core" in reachable, (
        "the dependency walk did not reach a known first-order dependency, so "
        "it is resolving nothing and proving nothing"
    )


def test_the_detector_would_catch_a_violation() -> None:
    """The predicates are exercised, not merely defined.

    Without this, a mistake in `covers` would make every assertion above pass
    for as long as the mistake survived.
    """
    assert boundary_scan.covers("liq.sim", "liq.sim")
    assert boundary_scan.covers("liq.sim.fills", "liq.sim")
    assert not boundary_scan.covers("liq.simulation", "liq.sim")
    assert not boundary_scan.covers("liq.core", "liq.sim")

    forbidden = (
        "liq.sim",
        "liq.gp",
        "liq.risk",
        "liq.data",
        "liq.features",
        "liq.datasets",
        "liq.runner",
        "liq.validation",
    )
    for module in forbidden:
        assert boundary_scan.covers(module, "liq")
        assert not boundary_scan.covered_by_any(
            module, boundary_scan.ALLOWED_FIRST_PARTY
        )
    assert boundary_scan.covered_by_any(
        "liq.core.bars", boundary_scan.ALLOWED_FIRST_PARTY
    )
    assert boundary_scan.covered_by_any(
        "liq.evolution.ecology.types", boundary_scan.ALLOWED_FIRST_PARTY
    )

    assert boundary_scan.covered_by_any(
        "liq.live", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert boundary_scan.covered_by_any(
        "liq.trade.runner", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert boundary_scan.covered_by_any(
        "ccxt.binance", boundary_scan.BROKER_IMPORT_PREFIXES
    )
    assert not boundary_scan.covered_by_any(
        "polars", boundary_scan.BROKER_IMPORT_PREFIXES
    )


def test_adapters_are_exempt_from_containment_but_not_from_the_broker_rule() -> None:
    """The two scans have deliberately different scopes; pin the difference."""
    assert boundary_scan.ADAPTER_DIRECTORY not in boundary_scan.ALLOWED_FIRST_PARTY
    if not boundary_scan.ECOLOGY_ROOT.is_dir():
        return
    adapter_tree = boundary_scan.ECOLOGY_ROOT / boundary_scan.ADAPTER_DIRECTORY
    scanned_everywhere = set(boundary_scan.ecology_files())
    contained = set(boundary_scan.decision_making_files())
    assert all(not path.is_relative_to(adapter_tree) for path in contained)
    assert contained <= scanned_everywhere
