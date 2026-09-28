"""Config loading and result cards: the two halves of "reproducible from a committed config"."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from usis.core.config import ConfigError, ExperimentConfig, load_config
from usis.core.provenance import RunContext
from usis.core.types import FailureCode
from usis.eval.metrics.statistics import MetricResult
from usis.eval.mrep import ResultCard, ResultCardError, failure_histogram

EXAMPLE = Path(__file__).parent.parent / "configs" / "example_baseline.yaml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- config ---------------------------------------------------------------------------------
def test_the_committed_example_config_loads():
    config = load_config(EXAMPLE)
    assert config.name == "example_baseline"
    assert config.split_kind == "sample_disjoint"
    assert config.require("metrics.bootstrap_samples") == 10000
    assert config.get("dataset.id") == "rod"


def test_split_kind_and_seed_are_required_with_no_default(tmp_path):
    for missing, text in (
        ("split_kind", "name: x\nseed: 0\n"),
        ("seed", "name: x\nsplit_kind: random\n"),
        ("name", "seed: 0\nsplit_kind: random\n"),
    ):
        with pytest.raises(ConfigError, match="missing required key"):
            load_config(_write(tmp_path, text))
        assert missing  # each case is distinct


def test_the_missing_key_message_explains_why_there_is_no_default(tmp_path):
    with pytest.raises(ConfigError, match="silent norm"):
        load_config(_write(tmp_path, "name: x\nseed: 0\n"))


def test_overrides_change_the_hash_and_mark_the_run_not_file_reproducible():
    plain = load_config(EXAMPLE)
    overridden = load_config(EXAMPLE, ["seed=7"])
    assert plain.is_reproducible_from_file is True
    assert overridden.is_reproducible_from_file is False
    assert overridden.seed == 7
    assert overridden.config_hash != plain.config_hash
    assert overridden.overrides == ("seed=7",)


def test_the_hash_covers_resolved_interpolations(tmp_path):
    # A hash of the unresolved template would be identical for two runs that differ.
    base = "name: x\nseed: 0\nsplit_kind: random\nroot: /a\npath: ${root}/data\n"
    first = load_config(_write(tmp_path, base))
    second = load_config(_write(tmp_path, base.replace("/a", "/b")))
    assert first.get("path") == "/a/data"
    assert second.get("path") == "/b/data"
    assert first.config_hash != second.config_hash


def test_a_missing_or_malformed_config_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no config at"):
        load_config(tmp_path / "absent.yaml")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))
    with pytest.raises(ConfigError, match="seed must be an integer"):
        load_config(_write(tmp_path, "name: x\nseed: not-a-number\nsplit_kind: random\n"))


def test_require_names_the_full_path_when_a_key_is_absent():
    with pytest.raises(ConfigError, match=r"metrics\.nope"):
        load_config(EXAMPLE).require("metrics.nope")


# --- result cards ---------------------------------------------------------------------------
@pytest.fixture
def context() -> RunContext:
    return RunContext.create("test-run", seed=0)


@pytest.fixture
def metric() -> MetricResult:
    return MetricResult.from_scores("accuracy", np.random.default_rng(0).beta(9, 1, 100), seed=0)


def _card(context, metric, **overrides) -> ResultCard:
    kwargs = {
        "title": "Test card",
        "run_context": context,
        "config": load_config(EXAMPLE),
        "metrics": (metric,),
        "shows": ("that the writer runs",),
        "does_not_show": ("anything about method quality",),
    }
    kwargs.update(overrides)
    return ResultCard(**kwargs)


def test_a_card_without_a_limitations_section_is_refused(context, metric):
    with pytest.raises(ResultCardError, match="does NOT show"):
        _card(context, metric, does_not_show=())


def test_a_card_without_a_shows_section_is_refused(context, metric):
    with pytest.raises(ResultCardError, match="does show"):
        _card(context, metric, shows=())


def test_a_card_recording_nothing_is_refused(context, metric):
    with pytest.raises(ResultCardError, match="records nothing"):
        _card(context, metric, metrics=(), failures={})


def test_a_card_with_only_a_failure_histogram_is_allowed(context, metric):
    card = _card(context, metric, metrics=(), failures={FailureCode.F04_FLUORESCENCE: 3})
    assert "F04_fluorescence" in card.render()


def test_the_rendered_card_carries_the_version_lock(context, metric):
    text = _card(context, metric).render()
    assert "## Version lock" in text
    assert context.git_sha[:12] in text or "unavailable" in text
    assert "hash" in text


def test_an_override_makes_the_card_say_it_is_not_attributable(context, metric):
    card = _card(context, metric, config=load_config(EXAMPLE, ["seed=3"]))
    text = card.render()
    assert card.is_attributable is False
    assert "not fully attributable" in text
    assert "seed=3" in text


def test_a_degenerate_interval_is_called_out_in_the_card(context):
    flat = MetricResult.from_scores("acc", [0.5] * 8, seed=0)
    text = _card(context, flat).render()
    assert "collapsed to a point" in text


def test_a_tail_carried_average_is_called_out(context):
    bimodal = np.concatenate([np.full(90, 0.95), np.zeros(10)])
    text = _card(context, MetricResult.from_scores("acc", bimodal, seed=0)).render()
    assert "carried by the tails" in text


def test_an_empty_failure_histogram_says_it_may_mean_the_classifier_did_not_run(context, metric):
    text = _card(context, metric).render()
    assert "not that nothing failed" in text


def test_the_card_writes_itself_and_the_run_context(tmp_path, context, metric):
    path = _card(context, metric).write(tmp_path / "run")
    assert path.name == "result_card.md"
    assert (tmp_path / "run" / "run_context.json").is_file()
    assert "# Test card" in path.read_text()


def test_split_caveats_travel_into_the_card(tmp_path, context, metric):
    from usis.data.splits import Split, SplitKind

    split = Split(
        kind=SplitKind.RANDOM,
        train=(0, 1),
        test=(2,),
        group_key="row index",
        train_groups=frozenset({"0", "1"}),
        test_groups=frozenset({"2"}),
        seed=0,
        notes=("random is a smoke test only.",),
    )
    text = _card(context, metric, split=split).render()
    assert "Caveats carried by this split" in text
    assert "smoke test only" in text


def test_failure_histogram_counts():
    codes = [FailureCode.F01_UNIT_AXIS, FailureCode.F01_UNIT_AXIS, FailureCode.F15_LEAKAGE]
    assert failure_histogram(codes) == {
        FailureCode.F01_UNIT_AXIS: 2,
        FailureCode.F15_LEAKAGE: 1,
    }


def test_a_card_needs_a_title(context, metric):
    with pytest.raises(ResultCardError, match="title"):
        _card(context, metric, title="")


def test_experiment_config_can_be_built_inline():
    config = ExperimentConfig(name="inline", seed=1, split_kind="random", values={"a": 1})
    assert config.is_reproducible_from_file is True
    assert config.config_hash
