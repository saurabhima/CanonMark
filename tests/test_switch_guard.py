"""Fingerprint guard: evaluation must refuse a switch that changed after generation."""
import warnings

import pytest

torch = pytest.importorskip("torch")

from canonmark.evaluation import make_detector  # noqa: E402
from canonmark.pipeline import wm_config  # noqa: E402
from canonmark.switch import (LearnedSwitch, SwitchMismatchError, SwitchVocab, build_switch_net,  # noqa: E402
                              switch_fingerprint)


def _save_switch(path, seed):
    torch.manual_seed(seed)
    vocab = SwitchVocab(["<s>", "for", "in", "v0", "return"])
    LearnedSwitch(build_switch_net(len(vocab.itos), 3), vocab).save(path)


def test_fingerprint_recorded_and_verified(tmp_path):
    sw = tmp_path / "switch"
    _save_switch(sw, 0)
    wm = wm_config("canonmark", "m", switch=str(sw))
    assert wm["switch_sha256"] == switch_fingerprint(str(sw)) and len(wm["switch_sha256"]) == 64
    make_detector(wm)                                   # unchanged switch: fine

    _save_switch(sw, 1)                                 # retrained into the same folder
    with pytest.raises(SwitchMismatchError, match="has changed"):
        make_detector(wm, source="runs/x.jsonl")


def test_missing_switch_is_an_error(tmp_path):
    sw = tmp_path / "switch"
    _save_switch(sw, 0)
    wm = wm_config("canonmark", "m", switch=str(sw))
    for f in sw.iterdir():
        f.unlink()
    with pytest.raises(SwitchMismatchError, match="missing"):
        make_detector(wm)


def test_legacy_runs_warn_and_always_on_needs_nothing(tmp_path):
    sw = tmp_path / "switch"
    _save_switch(sw, 0)
    legacy = {"method": "canonmark", "key": "k", "gamma": 0.5, "context_width": 3, "switch": str(sw), "model": "m"}
    with pytest.warns(UserWarning, match="no switch fingerprint"):
        make_detector(legacy)
    wm = wm_config("canonmark", "m", switch="always")
    assert wm["switch_sha256"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_detector(wm)                               # nothing to verify, no warning
