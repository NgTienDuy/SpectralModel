"""Corpus verification. Offline: nothing here calls fetch."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from usis.core.errors import UsisError
from usis.data.pull import (
    _chunk_spans,
    _whole_lines,
    dataset_dir,
    pull_bulk,
    pull_records,
    verify_directory,
)
from usis.data.sources.qm9s import Qm9sSource
from usis.data.sources.rod import RodSource

FIXTURES = Path(__file__).parent / "fixtures" / "rod"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    destination = dataset_dir(tmp_path, "rod")
    destination.mkdir(parents=True)
    for name in ("8100020.rod", "1000001.rod"):
        shutil.copy(FIXTURES / name, destination / name)
    return tmp_path


def test_verify_reparses_every_record(corpus):
    results = verify_directory(corpus)
    assert len(results) == 2
    assert all(r.ok for r in results)
    assert {r.first_spectrum_id for r in results} == {"rod:8100020", "rod:1000001"}
    assert all(r.n_spectra == 1 for r in results)
    assert all(r.n_points > 1000 for r in results)


def test_verify_derives_the_checksum_from_the_bytes_on_disk(corpus):
    results = verify_directory(corpus)
    by_id = {r.first_spectrum_id: r for r in results}
    assert by_id["rod:8100020"].sha256 == (
        "a9dd4f94c6162dab6099a4ab5bf59996d7c21f75056cf95af696aaf9e26bd32a"
    )


def test_verify_can_be_limited_to_one_dataset(corpus):
    assert len(verify_directory(corpus, dataset_id="rod")) == 2
    assert verify_directory(corpus, dataset_id="nope") == []


def test_a_multi_spectrum_payload_is_counted_not_collapsed(tmp_path):
    # A matrix CSV holds thousands of spectra in one file; verify must report the count rather
    # than the first row.
    import shutil

    destination = dataset_dir(tmp_path, "scidata-api-raman")
    destination.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent / "fixtures" / "scidata_api" / "api_compounds_subset.csv.gz",
        destination / "subset.csv.gz",
    )
    results = verify_directory(tmp_path)
    assert len(results) == 1
    assert results[0].ok
    assert results[0].n_spectra == 6
    assert results[0].first_spectrum_id == "scidata-api:00000"


def test_a_corrupt_record_fails_rather_than_being_skipped(corpus):
    (dataset_dir(corpus, "rod") / "broken.rod").write_text("data_x\n_rod_database.code 1\n")
    results = verify_directory(corpus)
    failures = [r for r in results if not r.ok]
    assert len(failures) == 1
    assert "no _raman_spectrum loop" in failures[0].error
    # The checksum is still reported, so a failure can be attributed to specific bytes.
    assert len(failures[0].sha256) == 64


def test_an_unknown_dataset_directory_is_reported_not_ignored(corpus):
    (corpus / "mystery-dataset").mkdir()
    results = verify_directory(corpus)
    bad = [r for r in results if r.dataset_id == "mystery-dataset"]
    assert len(bad) == 1 and not bad[0].ok
    assert "no entry" in bad[0].error


def test_verify_on_a_missing_directory_explains_itself(tmp_path):
    with pytest.raises(UsisError, match="run 'usis data pull'"):
        verify_directory(tmp_path / "absent")


def test_pull_with_no_records_is_refused_before_any_io(tmp_path):
    # Guards the empty-list path without touching the network.
    with pytest.raises(UsisError, match="no record ids"):
        pull_records("rod", [], root=tmp_path)


# --- bulk fetch: the byte-selection logic, offline ------------------------------------------------


def test_a_budget_at_or_above_the_file_size_is_the_whole_file():
    assert _chunk_spans(1000, 1000, 8) == [(0, 1000)]
    assert _chunk_spans(1000, 5000, 8) == [(0, 1000)]


def test_chunks_start_at_zero_so_a_csv_keeps_its_header():
    spans = _chunk_spans(1_000_000, 8_000, 8)
    assert spans[0][0] == 0


def test_chunks_span_the_file_rather_than_its_first_page():
    """The reason striding exists: both published CSVs are ordered, so a prefix is one corner."""
    total = 1_000_000
    spans = _chunk_spans(total, 8_000, 8)
    assert len(spans) == 8
    assert spans[-1][1] <= total
    # The last chunk must reach the far end, not sit just past the first.
    assert spans[-1][0] > 0.9 * total


def test_one_chunk_is_exactly_a_prefix():
    assert _chunk_spans(1_000_000, 8_000, 1) == [(0, 8_000)]


def test_a_middle_chunk_drops_its_leading_partial_line():
    """A byte offset lands mid-row; keeping that row would hand the loader a truncated record."""
    assert _whole_lines(b"lf-of-a-row\nfull\nrow\n", first=False) == b"full\nrow\n"


def test_the_first_chunk_keeps_its_first_line_because_that_is_the_header():
    assert _whole_lines(b"header\nrow\n", first=True) == b"header\nrow\n"


def test_a_trailing_partial_line_is_always_dropped():
    assert _whole_lines(b"header\nrow\nhalf-a-r", first=True) == b"header\nrow\n"


def test_a_chunk_with_no_complete_line_yields_nothing_rather_than_a_broken_row():
    assert _whole_lines(b"no newline here", first=False) == b""
    assert _whole_lines(b"no newline here", first=True) == b""


def test_bulk_fetch_is_refused_for_a_per_record_source(tmp_path):
    """ROD publishes per record; pretending it has a bulk file would mirror nothing."""
    with pytest.raises(UsisError, match="not published as bulk files"):
        pull_bulk("rod", root=tmp_path)


def test_the_rod_index_is_committed_and_holds_real_codes():
    """``pull --all`` has to fetch the same corpus on two different days."""
    ids = RodSource().list_record_ids()
    assert len(ids) > 1000
    assert all(code.isdigit() for code in ids)
    assert len(set(ids)) == len(ids)


def test_a_source_without_an_index_refuses_instead_of_mirroring_nothing():
    with pytest.raises(UsisError, match="does not publish an enumerable record index"):
        Qm9sSource("raman").list_record_ids()
