"""Detector statistics: calibrated null and exact invariance of z."""
import statistics

from canonmark.attacks import ATTACKS
from canonmark.data import load_humaneval, load_mbpp
from canonmark.detect import detect, stouffer
from canonmark.keyed import KeyedGreen

PROGS = [p["reference"] for p in load_humaneval()] + [p["reference"] for p in load_mbpp("test")]


def test_null_is_calibrated():
    """Human code, random keys: z should look like N(0, 1).

    This is the property CodeTracer's learned green lists cannot guarantee."""
    zs = []
    for k in range(40):
        g = KeyedGreen(f"key-{k}", gamma=0.5, context_width=3)
        zs += [detect(c, g).z for c in PROGS[k::40]]       # each program once, varied keys
    zs_all = []
    for k in range(10):
        g = KeyedGreen(f"other-{k}", 0.5, 3)
        zs_all += [detect(c, g).z for c in PROGS]
    m, s = statistics.mean(zs_all), statistics.pstdev(zs_all)
    assert abs(m) < 0.1, m
    assert 0.85 < s < 1.15, s
    fpr = sum(z > 4 for z in zs_all) / len(zs_all)
    assert fpr < 0.001


def test_z_unchanged_by_attacks():
    g = KeyedGreen("secret", 0.5, 3)
    for c in PROGS[:150]:
        z0 = detect(c, g).z
        for name in ("black", "strip_comments", "rename", "all_three"):
            att = ATTACKS[name](c)
            if att is not None:
                assert abs(detect(att, g).z - z0) < 1e-9, (name, c[:60])


def test_stouffer():
    assert abs(stouffer([2.0, 2.0]) - 2 * 2 ** 0.5) < 1e-9
