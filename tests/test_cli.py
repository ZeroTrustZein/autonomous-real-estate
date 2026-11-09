"""Unit tests for the CLI commands."""

import json
from autonomous_real_estate.cli import main


def test_cli_help(capsys):
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "are-underwrite" in captured.out
    assert "evaluate" in captured.out


def test_cli_no_args(capsys):
    ret = main([])
    assert ret == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err


def test_cli_evaluate_command(capsys):
    argv = [
        "evaluate",
        "--id", "TEST-PROP",
        "--price", "1000000",
        "--rent", "130000",
    ]
    ret = main(argv)
    assert ret == 0
    captured = capsys.readouterr()
    assert "UNDERWRITING REPORT: Commercial Asset (TEST-PROP)" in captured.out
    assert "DECISION:" in captured.out
    assert "BASE CASE METRICS" in captured.out


def test_cli_evaluate_json_command(capsys):
    argv = [
        "evaluate",
        "--id", "TEST-PROP",
        "--price", "1000000",
        "--rent", "130000",
        "--json",
    ]
    ret = main(argv)
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["property_id"] == "TEST-PROP"
    assert "base_case" in data
    assert "monte_carlo" in data


def test_cli_simulate_command(capsys):
    argv = [
        "simulate",
        "--price", "1000000",
        "--rent", "120000",
        "--iterations", "150",
    ]
    ret = main(argv)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Monte Carlo (150 trials):" in captured.out
    assert "Mean IRR:" in captured.out


def test_cli_simulate_json_command(capsys):
    argv = [
        "simulate",
        "--price", "1000000",
        "--rent", "120000",
        "--iterations", "150",
        "--json",
    ]
    ret = main(argv)
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["iterations"] == 150
    assert "value_at_risk_95" in data
