"""CLI tests for liq-evolution stage/gate configuration."""

from __future__ import annotations

import json
from pathlib import Path

import tomllib

from liq.evolution.config import EvolutionConfig


def _json_or_empty(text: str) -> dict:
    data = text.strip()
    return json.loads(data) if data else {}


def _run_cli(args: list[str], capsys) -> tuple[int, dict, dict]:
    from liq.evolution.cli.main import main

    exit_code = main(args)
    captured = capsys.readouterr()
    return (
        exit_code,
        _json_or_empty(captured.out),
        _json_or_empty(captured.err),
    )


def test_cli_success_default_config(capsys) -> None:
    exit_code, out, err = _run_cli([], capsys)
    expected = EvolutionConfig()

    assert exit_code == 0
    assert err == {}
    assert out["ok"] is True
    assert out["config"]["population_size"] == expected.population_size
    assert out["config"]["max_depth"] == expected.max_depth
    assert out["config"]["generations"] == expected.generations
    assert out["config"]["seed"] == expected.seed


def test_cli_config_file_overrides_and_flag_sync(tmp_path: Path, capsys) -> None:
    payload = {
        "population_size": 300,
        "run": {
            "stage_a_candidate_budget": 1500,
            "stage_a_threshold": 0.75,
            "stage_b_candidate_budget": 200,
        },
        "fitness_stages": {
            "use_backtest": False,
            "label_metric": "accuracy",
            "label_top_k": 0.2,
        },
        "fitness": {"stage_b_enabled": False},
        "regime": {
            "regime_confidence_threshold": 0.55,
            "regime_occupancy_threshold": 0.15,
            "regime_hysteresis_margin": 0.07,
            "regime_min_persistence": 4,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, out, err = _run_cli(
        [
            "--config",
            str(config_path),
            "--population-size",
            "512",
            "--stage-a-candidate-budget",
            "2222",
            "--label-top-k",
            "0.33",
            "--use-backtest",
            "--regime-min-persistence",
            "5",
        ],
        capsys,
    )

    assert exit_code == 0
    assert err == {}
    assert out["ok"] is True
    assert out["config"]["population_size"] == 512
    assert out["config"]["run"]["stage_a_candidate_budget"] == 2222
    assert out["config"]["run"]["stage_a_threshold"] == 0.75
    assert out["config"]["fitness_stages"]["label_top_k"] == 0.33
    assert out["config"]["fitness_stages"]["use_backtest"] is True
    assert out["config"]["fitness"]["stage_b_enabled"] is True
    assert out["config"]["regime"]["regime_min_persistence"] == 5


def test_cli_missing_config_file(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does-not-exist.json"
    exit_code, out, err = _run_cli(["--config", str(missing)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.config_missing"


def test_cli_invalid_config_payload(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"population_size": }', encoding="utf-8")

    exit_code, out, err = _run_cli(["--config", str(bad)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.config_parse_error"


def test_cli_validation_error(tmp_path: Path, capsys) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"population_size": 1}), encoding="utf-8")

    exit_code, out, err = _run_cli(["--config", str(invalid)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.config_validation_error"


def test_cli_direct_configuration_error_is_wrapped(tmp_path: Path, capsys) -> None:
    conflicting = tmp_path / "conflict.json"
    conflicting.write_text(
        json.dumps({"population_size": 300, "gp": {"population_size": 500}}),
        encoding="utf-8",
    )

    exit_code, out, err = _run_cli(["--config", str(conflicting)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.config_validation_error"
    assert isinstance(err["context"]["validation_errors"], list)
    assert err["context"]["validation_errors"][0]["type"] == "configuration_error"
    assert "must match" in err["context"]["validation_errors"][0]["msg"]


def test_cli_config_schema_violation(tmp_path: Path, capsys) -> None:
    bad_type = tmp_path / "bad_type.json"
    bad_type.write_text("[]", encoding="utf-8")

    exit_code, out, err = _run_cli(["--config", str(bad_type)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.config_schema_violation"


def test_cli_stage_b_mismatch(tmp_path: Path, capsys) -> None:
    payload = {
        "fitness_stages": {"use_backtest": True},
        "fitness": {"stage_b_enabled": False},
    }
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, out, err = _run_cli(["--config", str(path)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.stage_b_mismatch"


def test_cli_unsafe_budget_guard_and_override(tmp_path: Path, capsys) -> None:
    payload = {"run": {"stage_a_candidate_budget": 25_000}}
    path = tmp_path / "big_budget.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, out, err = _run_cli(["--config", str(path)], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.unsafe_budget"

    exit_code, out, err = _run_cli(
        ["--config", str(path), "--allow-expensive"],
        capsys,
    )
    assert exit_code == 0
    assert err == {}
    assert out["ok"] is True
    assert out["config"]["run"]["stage_a_candidate_budget"] == 25_000


def test_cli_argparse_error(capsys) -> None:
    exit_code, out, err = _run_cli(["--unknown-flag"], capsys)
    assert exit_code == 2
    assert out == {}
    assert err["error_code"] == "evo.cli.argparse_error"


def test_cli_script_entrypoint_declared() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as handle:
        project_data = tomllib.load(handle)

    scripts = project_data["project"]["scripts"]
    assert scripts["liq-evolution"] == "liq.evolution.cli.main:main"
