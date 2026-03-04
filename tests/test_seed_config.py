"""Tests for seed configuration loading and builder utilities."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from liq.evolution.errors import ConfigurationError
from liq.evolution.seed import config as seed_config
from liq.evolution.seed.config import SeedManifest, SeedSpec


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_seed_spec_normalizes_strategy_name() -> None:
    spec = SeedSpec(strategy="Momentum-EMA", params={"lookback": 14})
    assert spec.strategy == "momentum_ema"


def test_seed_manifest_filters_disabled_entries() -> None:
    manifest = SeedManifest(
        seeds=(
            [
                SeedSpec(strategy="momentum"),
                SeedSpec(strategy="trend", enabled=False),
            ]
        )
    )
    enabled = manifest.enabled_seeds
    assert len(enabled) == 1
    assert enabled[0].strategy == "momentum"


def test_load_payload_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "seeds.txt"
    _write(path, "[]")

    with pytest.raises(
        ConfigurationError, match="Unsupported seed config extension for"
    ):
        seed_config.load_seed_manifest(path)


def test_load_payload_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    _write(path, "{")

    with pytest.raises(ConfigurationError, match="Invalid JSON in seed config"):
        seed_config.load_seed_manifest(path)


def test_load_payload_reports_json_read_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "seeds.json"
    _write(path, "[]")

    def _raise_read_error(*_args, **_kwargs):
        raise OSError("read denied")

    monkeypatch.setattr(Path, "read_text", _raise_read_error)
    with pytest.raises(ConfigurationError, match="Failed to read seed config"):
        seed_config.load_seed_manifest(path)


def test_load_payload_reports_yaml_import_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "seeds.yaml"
    _write(path, "strategy: momentum")
    monkeypatch.setattr(seed_config, "yaml", None)

    with pytest.raises(
        ConfigurationError, match="PyYAML is required to load .yaml seed configs"
    ):
        seed_config.load_seed_manifest(path)


def test_load_payload_handles_valid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "seeds.yaml"
    _write(path, "strategy: momentum\nparams: {}\n")

    manifest = seed_config.load_seed_manifest(path)
    assert len(manifest.seeds) == 1
    assert manifest.seeds[0].strategy == "momentum"


def test_load_payload_reports_yaml_read_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "seeds.yaml"
    _write(path, "strategy: momentum\n")

    def _raise_read_error(*_args, **_kwargs):
        raise OSError("read denied")

    monkeypatch.setattr(Path, "read_text", _raise_read_error)
    with pytest.raises(ConfigurationError, match="Failed to read seed config"):
        seed_config.load_seed_manifest(path)


def test_load_payload_reports_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "seeds.yaml"
    _write(path, "seed: [1, 2")

    with pytest.raises(ConfigurationError, match="Invalid YAML in seed config"):
        seed_config.load_seed_manifest(path)


def test_coerce_seed_entries_rejects_invalid_list_entry(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="Invalid seed entry in .*index 1; expected object",
    ):
        seed_config._coerce_seed_entries(
            [{"strategy": "momentum"}, 3], tmp_path / "x.json"
        )


def test_coerce_seed_entries_rejects_bad_dict_payload_variants(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    with pytest.raises(
        ConfigurationError, match="Invalid seed config in .*'seeds' must be a list"
    ):
        seed_config._coerce_seed_entries({"seeds": {"a": 1}}, source)

    with pytest.raises(
        ConfigurationError, match="Invalid seed config in .*'seed' must be an object"
    ):
        seed_config._coerce_seed_entries({"seed": 1}, source)

    with pytest.raises(
        ConfigurationError,
        match="Invalid seed config format in .*expected object with strategy/seed/seeds or array",
    ):
        seed_config._coerce_seed_entries({"unknown": 1}, source)


def test_coerce_seed_entries_accepts_seeds_and_seed_shapes(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    many = seed_config._coerce_seed_entries(
        {"seeds": [{"strategy": "momentum"}]},
        source,
    )
    single = seed_config._coerce_seed_entries(
        {"seed": {"strategy": "trend"}},
        source,
    )

    assert many == [{"strategy": "momentum"}]
    assert single == [{"strategy": "trend"}]


def test_load_manifest_from_payload_variants(tmp_path: Path) -> None:
    null_payload = tmp_path / "null.json"
    _write(null_payload, "null")
    null_manifest = seed_config.load_seed_manifest(str(null_payload))
    assert null_manifest.metadata["source"] == null_payload.as_posix()
    assert null_manifest.seeds == []

    list_payload = tmp_path / "list.json"
    _write(list_payload, '[{"strategy": "momentum", "enabled": false}]')
    list_manifest = seed_config.load_seed_manifest(list_payload)
    assert len(list_manifest.seeds) == 1
    assert not list_manifest.seeds[0].enabled

    manifest_payload = tmp_path / "manifest.json"
    _write(
        manifest_payload,
        '{"schema_version": 1, "seeds": [{"strategy": "trend", "enabled": false}]}',
    )
    manifest = seed_config.load_seed_manifest(manifest_payload)
    assert manifest.schema_version == 1
    assert len(manifest.seeds) == 1


def test_load_manifest_reports_entry_validation_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    _write(broken, '[{"strategy": {}}]')
    with pytest.raises(ConfigurationError, match="Invalid seed entry in .*"):
        seed_config.load_seed_manifest(broken)


def test_load_manifest_reports_manifest_validation_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken_manifest.json"
    _write(broken, '{"schema_version": "invalid", "seeds": []}')

    with pytest.raises(ConfigurationError, match="Invalid seed manifest in .*"):
        seed_config.load_seed_manifest(broken)


def test_load_manifest_reports_dict_entry_validation_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken_entry_dict.json"
    _write(broken, '{"strategy": {}}')

    with pytest.raises(ConfigurationError, match="Invalid seed entry in .*"):
        seed_config.load_seed_manifest(broken)


def test_load_seed_manifest_aggregates_directory(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    _write(cfg_dir / "01.json", '[{"strategy": "momentum"}]')
    _write(cfg_dir / "02.yaml", "strategy: trend\n")
    _write(cfg_dir / "ignored.txt", '[{"strategy": "momentum"}]')

    manifest = seed_config.load_seed_manifest(cfg_dir)
    assert manifest.metadata["directory"] == cfg_dir.as_posix()
    assert len(manifest.seeds) == 2


def test_load_seed_specs_returns_enabled(tmp_path: Path) -> None:
    path = tmp_path / "specs.json"
    _write(path, '[{"strategy":"momentum","enabled":false},{"strategy":"trend"}]')

    specs = seed_config.load_seed_specs(path)
    assert len(specs) == 1
    assert specs[0].strategy == "trend"


def test_load_seed_manifest_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(ConfigurationError, match="Seed path does not exist"):
        seed_config.load_seed_manifest(missing)


def test_load_seed_manifest_rejects_non_file_non_dir(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "opaque.node"
    _write(marker, "seed")

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(ConfigurationError, match="Seed path is not a file"):
        seed_config.load_seed_manifest(marker)


def test_build_seed_programs_skips_disabled_specs(monkeypatch) -> None:
    builder = MagicMock(side_effect=lambda *args, **kwargs: f"program:{kwargs}")
    template = MagicMock(builder=builder)
    monkeypatch.setattr(seed_config, "get_seed_template", lambda name: template)

    specs = [
        SeedSpec(strategy="momentum", params={"a": 1}),
        SeedSpec(strategy="trend", params={"b": 2}, enabled=False),
    ]
    out = seed_config.build_seed_programs(specs, MagicMock())
    assert out == ["program:{'a': 1}"]
    assert builder.call_count == 1


def test_build_seed_programs_from_path_uses_build_chain(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "specs.json"
    _write(path, '[{"strategy":"momentum", "enabled": false}, {"strategy":"trend"}]')
    builder = MagicMock(side_effect=lambda *args, **kwargs: f"program:{kwargs}")
    monkeypatch.setattr(
        seed_config, "get_seed_template", lambda name: MagicMock(builder=builder)
    )

    out = seed_config.build_seed_programs_from_path(path, MagicMock())
    assert out == ["program:{}"]
    assert builder.call_count == 1
