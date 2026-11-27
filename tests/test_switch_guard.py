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


def test_weighted_detector_null_is_calibrated(tmp_path):
    """Weights depend only on the canonical context, so z_w stays N(0,1) on human code."""
    import statistics
    from canonmark.data import load_humaneval, load_mbpp
    from canonmark.detect import detect
    from canonmark.keyed import KeyedGreen
    torch.manual_seed(0)
    vocab = SwitchVocab.build([["for", "in", "if", "return", "v0", "v1", "len", "==", "+"]] * 3)
    sw = LearnedSwitch(build_switch_net(len(vocab.itos), 3), vocab, threshold=0.3)
    progs = [p["reference"] for p in load_humaneval()] + [p["reference"] for p in load_mbpp("test")]
    zs = [detect(c, KeyedGreen(f"w{k}", 0.5, 3), sw, weighted=True).z for k in range(6) for c in progs]
    m, s = statistics.mean(zs), statistics.pstdev(zs)
    assert abs(m) < 0.1 and 0.85 < s < 1.15, (m, s)


def test_weighted_equals_unweighted_for_always_on():
    from canonmark.data import load_humaneval
    from canonmark.detect import detect
    from canonmark.keyed import KeyedGreen
    g = KeyedGreen("k", 0.5, 3)
    for p in load_humaneval()[:30]:
        assert abs(detect(p["reference"], g).z - detect(p["reference"], g, weighted=True).z) < 1e-9
