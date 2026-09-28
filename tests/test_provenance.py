"""Run provenance and the replay log. Without these, model revision is anecdote."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from usis.core.provenance import (
    RunContext,
    capture_environment,
    content_hash,
    git_state,
    hash_payload,
    set_global_seeds,
    utc_now,
)
from usis.core.runlog import RunLog, read_events


def test_utc_now_is_iso_and_utc():
    stamp = utc_now()
    assert stamp.endswith("+00:00")
    assert "T" in stamp


def test_content_hash_is_stable_and_sensitive(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"spectrum")
    first = content_hash(path)
    assert len(first) == 64
    assert content_hash(path) == first
    path.write_bytes(b"spectrun")
    assert content_hash(path) != first


def test_hash_payload_ignores_key_order():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
    assert hash_payload({"a": 1}) != hash_payload({"a": 2})


def test_git_state_reports_a_sha_or_says_it_is_unavailable():
    state = git_state()
    assert set(state) >= {"git_sha", "git_dirty"}
    assert state["git_sha"] == "unavailable" or len(state["git_sha"]) == 40


def test_capture_environment_names_the_interpreter_and_numpy():
    env = capture_environment()
    assert env["python"].startswith("3.")
    assert env["numpy"] != "missing"


def test_seeding_is_reported_and_actually_deterministic():
    seeded = set_global_seeds(1234)
    assert seeded["python"] == 1234
    first = (random.random(), float(np.random.rand()))
    set_global_seeds(1234)
    assert (random.random(), float(np.random.rand())) == first


def test_seeded_dict_only_claims_what_was_seeded():
    seeded = set_global_seeds(0)
    assert "numpy" in seeded
    if "torch" in seeded:
        pytest.importorskip("torch")


def test_run_context_round_trips_through_disk(tmp_path):
    ctx = RunContext.create("run-1", seed=7, config={"lr": 0.1}, notes="smoke")
    path = ctx.write(tmp_path / "run")
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "run-1"
    assert payload["seed"] == 7
    assert payload["config_hash"] == hash_payload({"lr": 0.1})
    assert payload["notes"] == "smoke"
    assert payload["environment"]["python"]


def test_run_log_writes_and_replays_in_order(tmp_path):
    path = tmp_path / "run.jsonl"
    with RunLog(path) as log:
        log.event("start", milestone="M0")
        log.event("observe", spectrum_id="s1", snr=12.5)
        log.event("abstain", reason="library_gap")
    events = list(read_events(path))
    assert [e.kind for e in events] == ["start", "observe", "abstain"]
    assert events[1].payload["snr"] == 12.5
    assert all(e.timestamp for e in events)


def test_run_log_survives_a_process_that_never_closes_it(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path)
    log.event("start")
    # No close(): a killed run must still leave its trace on disk.
    assert len(list(read_events(path))) == 1
    log.close()


def test_malformed_run_log_line_raises_rather_than_being_skipped(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"kind": "ok"}\nnot json\n')
    with pytest.raises(ValueError, match="malformed"):
        list(read_events(path))
