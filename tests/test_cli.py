"""The CLI surface. A planned verb must refuse loudly, not print a plausible-looking result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usis.cli.main import EXIT_NOT_IMPLEMENTED, EXIT_USIS_ERROR, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "spectroscopy" in capsys.readouterr().out


def test_no_verb_prints_help():
    assert main([]) == 0


def test_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_info_json_carries_the_provenance_fields(capsys):
    assert main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["usis_version"]
    assert "git_sha" in payload
    assert payload["environment"]["python"]


def test_licences_lists_the_gate(capsys):
    assert main(["licences"]) == 0
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out
    assert "NO" in out


def test_convert_axis(capsys):
    assert (
        main(
            [
                "convert",
                "axis",
                "--from",
                "wavelength_nm",
                "--to",
                "wavenumber_cm-1",
                "--json",
                "500.0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"][0] == pytest.approx(20000.0)


def test_convert_across_the_raman_boundary_without_excitation_fails_cleanly(capsys):
    code = main(["convert", "axis", "--from", "raman_shift_cm-1", "--to", "wavelength_nm", "1000"])
    assert code == EXIT_USIS_ERROR
    assert "excitation" in capsys.readouterr().err


def test_convert_intensity(capsys):
    assert (
        main(
            [
                "convert",
                "intensity",
                "--from",
                "transmittance",
                "--to",
                "absorbance",
                "--json",
                "0.1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"][0] == pytest.approx(1.0)


def test_data_list_shows_the_manifest_and_the_gate(capsys):
    assert main(["data", "list"]) == 0
    out = capsys.readouterr().out
    assert "rod" in out
    assert "CC0-1.0" in out


def test_data_list_json(capsys):
    assert main(["data", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rod"]["vendorable"] is True
    assert payload["rod"]["verification"] == "VERIFIED"


def test_data_with_no_subverb_prints_help(capsys):
    assert main(["data"]) == 0
    assert "verify" in capsys.readouterr().out


def test_data_verify_on_the_committed_fixtures(tmp_path, capsys):
    import shutil

    source = Path(__file__).parent / "fixtures" / "rod"
    destination = tmp_path / "rod"
    destination.mkdir()
    shutil.copy(source / "8100020.rod", destination / "8100020.rod")
    assert main(["data", "verify", "--dest", str(tmp_path)]) == 0
    assert "rod:8100020" in capsys.readouterr().out


def test_data_verify_reports_a_failure_with_a_non_zero_exit(tmp_path, capsys):
    destination = tmp_path / "rod"
    destination.mkdir()
    (destination / "broken.rod").write_text("data_x\n_rod_database.code 1\n")
    assert main(["data", "verify", "--dest", str(tmp_path)]) == EXIT_USIS_ERROR
    assert "FAIL" in capsys.readouterr().err


def test_data_verify_on_an_empty_tree_exits_non_zero(tmp_path, capsys):
    assert main(["data", "verify", "--dest", str(tmp_path)]) == EXIT_USIS_ERROR
    assert "nothing found" in capsys.readouterr().err


def test_run_context_emits_json(capsys):
    assert main(["run-context", "r1", "--seed", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "r1"
    assert payload["seed"] == 3


@pytest.mark.parametrize("verb", ["analyze", "simulate"])
def test_planned_verbs_refuse_and_name_their_milestone(verb, capsys):
    assert main([verb]) == EXIT_NOT_IMPLEMENTED
    err = capsys.readouterr().err
    assert "not implemented at this milestone" in err
    assert "M" in err


def test_methods_lists_the_registry_with_the_surfacing_column(capsys):
    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    assert "audit.saturation@1.0.0" in out
    assert "shape.broadening_origin@0.2.0" in out
    assert "surfaces" in out
    hypothesized_row = next(
        line for line in out.splitlines() if line.startswith("shape.broadening_origin@0.2.0 ")
    )
    assert "NO" in hypothesized_row


def test_methods_filters_by_maturity_and_emits_json(capsys):
    assert main(["methods", "--maturity", "established", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    keys = [row["key"] for row in rows]
    assert "audit.saturation@1.0.0" in keys
    assert "shape.broadening_origin@0.2.0" not in keys
    assert all(row["maturity"] == "established" for row in rows)
    # Every established row must carry the reproduction reference its maturity claims (D14).
    assert all(row["evidence"] for row in rows)


def test_methods_reports_an_empty_filter_without_pretending(capsys):
    assert main(["methods", "--maturity", "refuted"]) == 0
    assert "no methods registered at this filter" in capsys.readouterr().out
