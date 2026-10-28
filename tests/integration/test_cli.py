import json

import pytest

from gpu_os_copilot.cli import main


@pytest.mark.parametrize("scenario", ["cpu_starvation", "transfer_bound", "compute_bound"])
def test_fixture_cli(scenario, capsys):
    assert main(["demo-fixture", "--scenario", scenario]) == 0
    output = capsys.readouterr().out
    assert "SIMULATED" in output
    assert "not a GPU measurement" in output


def test_json_cli(capsys):
    assert main(["demo-fixture", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["provenance"]["source"] == "fixture"


def test_meta_requires_explicit_paid_opt_in(capsys):
    assert main(["demo-fixture", "--mode", "meta"]) == 2
    assert "PAID_API_NOT_APPROVED" in capsys.readouterr().err


def test_trace_cli_is_only_a_plan(tmp_path, capsys):
    assert main(["trace", "--out", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert not list(tmp_path.iterdir())
