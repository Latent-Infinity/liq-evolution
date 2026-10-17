"""The declared two-regime per-expert loss stream, and the writer of its family.

Generated (authorized by Task V5.2) —
``liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md``. Non-signal-bearing,
``DEV_SMOKE_ONLY``, never citable.

**Nothing here is market data and nothing here may be read as market data.**
A stream is a declared sequence of dimensionless per-expert returns and a
declared reference series, built by pure arithmetic from one
``numpy.random.default_rng(seed)`` generator. It contains no OHLCV, order-book,
trade, quote, fill, price, instrument or market-regime quantity, and no
parameter is fitted to or derived from any real series. It enters only at the
aggregator's loss input: per-expert outcomes and reference returns.

The arithmetic is the one ``two-regime-generator-spec.md`` (sha256
``8a6777ab…c877``) states, implemented from that text. The draws, in order:
a Rademacher sign per decision point; a permutation of the experts whose first
two entries are the regime-1 and regime-2 best experts; and, only for an
``all_noise`` or ``idiosyncratic`` stream, a ``(T, P)`` standard-normal block.
Row 0 of the outcomes and element 0 of the reference are produced and never
consumed; every consumer starts at decision point 1.

The family a stream is asked for changes nothing in the arithmetic: it only
selects the seed range the generator admits, and a seed outside it is refused.

This module has no path into any data store. Its only writer is
:func:`materialise`, which writes a declared family to the directory it is
handed, normally ``tests/fixtures/generated/ecology/two_regime/``.
"""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "1.0.0"
AUTHORIZED_BY = "Task V5.2 / liq-docs/plans/oracle-v5-generated-streams-2026-10-15.md"

#: The seed range each family admits, inclusive at both ends.
FAMILY_SEED_RANGES: Mapping[str, tuple[int, int]] = {
    "E": (1_000_000, 1_999_999),
    "C": (0, 999_999),
}

KINDS = ("switch", "no_switch", "all_noise")
NOISES = ("pathwise", "idiosyncratic")

#: The first line of every rendered stream starts with this.
STREAM_HEADER_PREFIX = "# Generated (authorized by Task V5.2) by two_regime_stream"


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """The parameters of one stream, named as the spec names them."""

    seed: int
    P: int
    T: int
    kind: str
    noise: str | None
    switch_at: int | None
    Delta: float | None
    sigma: float
    lam: float


@dataclass(frozen=True, slots=True)
class TwoRegimeStream:
    """One generated stream: ``outcomes`` is ``(T, P)``, ``reference`` ``(T,)``."""

    spec: StreamSpec
    reference: np.ndarray
    outcomes: np.ndarray
    B1: int
    B2: int
    best: np.ndarray | None


def expert_ids(P: int) -> list[str]:
    """Expert identifiers, whose ascending order equals index order."""
    return [f"e{i:05d}" for i in range(P)]


def _refuse_seed_outside(family: str, seed: int) -> None:
    if family not in FAMILY_SEED_RANGES:
        raise ValueError(f"unknown family {family!r}")
    low, high = FAMILY_SEED_RANGES[family]
    if not low <= seed <= high:
        raise ValueError(
            f"seed {seed} is outside family {family}'s range [{low}, {high}]"
        )


def _validate(spec: StreamSpec) -> None:
    if spec.P < 2:
        raise ValueError("P must be at least 2")
    if spec.T < 2:
        raise ValueError("T must be at least 2")
    if spec.kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not spec.sigma > 0.0:
        raise ValueError("sigma must be positive")
    if not spec.lam >= 0.0:
        raise ValueError("lam must be non-negative")
    if spec.kind == "all_noise":
        return
    if spec.noise not in NOISES:
        raise ValueError(f"noise must be one of {NOISES}")
    if spec.Delta is None or not spec.Delta > 0.0:
        raise ValueError("Delta must be positive")
    if spec.kind == "switch" and (
        spec.switch_at is None or not 1 < spec.switch_at < spec.T
    ):
        raise ValueError("switch_at must satisfy 1 < switch_at < T")


def generate(spec: StreamSpec, *, family: str) -> TwoRegimeStream:
    """Generate one stream; refuse a seed outside ``family``'s range."""
    _refuse_seed_outside(family, spec.seed)
    _validate(spec)
    P, T, sigma = spec.P, spec.T, spec.sigma
    rng = np.random.default_rng(spec.seed)
    z = rng.integers(0, 2, size=T).astype(np.float64) * 2.0 - 1.0
    perm = rng.permutation(P)
    B1, B2 = int(perm[0]), int(perm[1])
    needs_eps = spec.kind == "all_noise" or spec.noise == "idiosyncratic"
    eps = rng.standard_normal((T, P)) if needs_eps else None

    reference = sigma * z
    if spec.kind == "all_noise":
        assert eps is not None
        return TwoRegimeStream(spec, reference, sigma * eps, B1, B2, None)

    assert spec.Delta is not None
    delta = spec.Delta * sigma
    if spec.kind == "no_switch":
        best = np.full(T, B1, dtype=np.int64)
    else:
        assert spec.switch_at is not None
        best = np.where(np.arange(T) < spec.switch_at, B1, B2).astype(np.int64)
    drift = np.full((T, P), -delta)
    drift[np.arange(T), best] = 0.0
    outcomes = sigma * z[:, None] + drift
    if spec.noise == "idiosyncratic":
        assert eps is not None
        outcomes = outcomes + (spec.lam * sigma) * eps
    return TwoRegimeStream(spec, reference, outcomes, B1, B2, best)


# --- text form ----------------------------------------------------------------


def _fmt(value: float) -> str:
    return repr(float(value))


def render(stream: TwoRegimeStream, *, stream_id: str) -> bytes:
    """The stream as labelled CSV text; floats in shortest round-trip form."""
    s = stream.spec
    names = expert_ids(s.P)
    described = " ".join(
        f"{key}={value}"
        for key, value in (
            ("stream", stream_id),
            ("seed", s.seed),
            ("kind", s.kind),
            ("noise", s.noise),
            ("P", s.P),
            ("T", s.T),
            ("switch_at", s.switch_at),
            ("Delta", s.Delta),
            ("sigma", s.sigma),
            ("lam", s.lam),
            ("B1", names[stream.B1]),
            ("B2", names[stream.B2]),
        )
        if value is not None
    )
    lines = [
        f"{STREAM_HEADER_PREFIX} {GENERATOR_VERSION}. Nothing here is market data.",
        f"# {described}",
        ",".join(["t", "ref", *names]),
    ]
    for t in range(s.T):
        row = [str(t), _fmt(stream.reference[t])]
        row.extend(_fmt(v) for v in stream.outcomes[t])
        lines.append(",".join(row))
    return ("\n".join(lines) + "\n").encode("ascii")


def parse(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read a rendered stream back: ``(reference, outcomes)``."""
    rows = [
        line.split(",")
        for line in data.decode("ascii").splitlines()
        if line and not line.startswith("#")
    ]
    body = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    return body[:, 0].copy(), body[:, 1:].copy()


# --- declaration and manifest -------------------------------------------------


def spec_from_declaration(
    family_fields: Mapping[str, object], stream: Mapping[str, object]
) -> StreamSpec:
    """A stream's spec from a family declaration's fields and its entry."""

    def field(name: str) -> object:
        return stream.get(name, family_fields.get(name))

    return StreamSpec(
        seed=int(stream["seed"]),  # type: ignore[arg-type]
        P=int(field("P")),  # type: ignore[arg-type]
        T=int(field("T")),  # type: ignore[arg-type]
        kind=str(stream["kind"]),
        noise=stream.get("noise"),  # type: ignore[arg-type]
        switch_at=stream.get("switch_at"),  # type: ignore[arg-type]
        Delta=stream.get("Delta"),  # type: ignore[arg-type]
        sigma=float(field("sigma")),  # type: ignore[arg-type]
        lam=float(field("lam")),  # type: ignore[arg-type]
    )


def spec_from_manifest(entry: Mapping[str, object]) -> StreamSpec:
    """A stream's spec from its manifest entry."""
    return StreamSpec(
        seed=int(entry["seed"]),  # type: ignore[arg-type]
        P=int(entry["K"]),  # type: ignore[arg-type]
        T=int(entry["T"]),  # type: ignore[arg-type]
        kind=str(entry["kind"]),
        noise=entry.get("noise"),  # type: ignore[arg-type]
        switch_at=entry.get("switch_at"),  # type: ignore[arg-type]
        Delta=entry.get("regime_gap"),  # type: ignore[arg-type]
        sigma=float(entry["sigma"]),  # type: ignore[arg-type]
        lam=float(entry["lam"]),  # type: ignore[arg-type]
    )


def _noise_spec(spec: StreamSpec) -> str:
    if spec.kind == "all_noise":
        return "outcomes = sigma * standard_normal((T, P)); reference = sigma * z"
    if spec.noise == "pathwise":
        return "common term sigma * z (Rademacher); no idiosyncratic draw"
    return (
        "common term sigma * z (Rademacher) + (lam * sigma) * standard_normal((T, P))"
    )


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_table(fields: Sequence[tuple[str, object]]) -> list[str]:
    return [f"{k} = {_toml_value(v)}" for k, v in fields if v is not None]


def materialise(declaration: Path, out_dir: Path, *, declaration_ref: str) -> Path:
    """Write every stream a declaration lists, and its manifest; return the manifest."""
    raw = declaration.read_bytes()
    decl = tomllib.loads(raw.decode("utf-8"))
    family = str(decl["family"])
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated (authorized by Task V5.2): manifest of the materialised",
        f"# family {family}. Nothing here is market data. Written by",
        "# tests/support/two_regime_stream.py; never edit by hand.",
        *_toml_table(
            [
                ("authorized_by", AUTHORIZED_BY),
                ("signal_bearing", False),
                ("citable", False),
                ("family", family),
                ("declaration", declaration_ref),
                ("declaration_sha256", hashlib.sha256(raw).hexdigest()),
                ("generator_spec_sha256", decl["generator_spec_sha256"]),
                ("generator_module", "tests/support/two_regime_stream.py"),
                ("generator_version", GENERATOR_VERSION),
                ("generator_module_sha256", module_sha),
            ]
        ),
    ]
    for entry in decl["stream"]:
        spec = spec_from_declaration(decl, entry)
        stream = generate(spec, family=family)
        name = f"{entry['id']}.csv"
        data = render(stream, stream_id=str(entry["id"]))
        (out_dir / name).write_bytes(data)
        names = expert_ids(spec.P)
        best_after = names[stream.B2] if spec.kind == "switch" else None
        lines += [
            "",
            "[[stream]]",
            *_toml_table(
                [
                    ("id", entry["id"]),
                    ("file", name),
                    ("sha256", hashlib.sha256(data).hexdigest()),
                    ("seed", spec.seed),
                    ("K", spec.P),
                    ("T", spec.T),
                    ("kind", spec.kind),
                    ("switch_at", spec.switch_at),
                    ("switch_fraction", entry.get("switch_fraction")),
                    ("regime_gap", spec.Delta),
                    ("noise", spec.noise),
                    ("noise_spec", _noise_spec(spec)),
                    ("sigma", spec.sigma),
                    ("lam", spec.lam),
                    ("best_before", None if stream.best is None else names[stream.B1]),
                    ("best_after", best_after),
                    ("generator_version", GENERATOR_VERSION),
                    ("generator_module_sha256", module_sha),
                ]
            ),
        ]
    manifest = out_dir / "manifest.toml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("declaration", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--declaration-ref", required=True)
    args = parser.parse_args(argv)
    print(
        materialise(
            args.declaration, args.out_dir, declaration_ref=args.declaration_ref
        )
    )


if __name__ == "__main__":
    main()
