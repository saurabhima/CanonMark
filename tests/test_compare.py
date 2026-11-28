"""Statistics of scripts/compare.py on synthetic sweeps with known ground truth."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("compare", Path(__file__).resolve().parent.parent / "scripts" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

N = 164


def _runs(shift, difficulty, rng, keys=3, noise=1.0):
    """Per-key z-scores: watermarked positives = difficulty + shift + noise; negatives = noise."""
    out = []
    for _ in range(keys):
        out.append({"pos": {f"t{i}": difficulty[i] + shift + rng.normal(0, noise) for i in range(N)},
                    "neg_model": {f"t{i}": rng.normal(0, 1) for i in range(N)}})
    return out


def test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 1.0, N)                       # shared per-problem difficulty
    r = compare.compare_auroc(_runs(1.0, diff, rng), _runs(0.4, diff, rng), "neg_model", 1000, rng)
    assert r["d_auroc"] > 0.08 and r["ci95"][0] > 0 and r["p"] < 0.01


def test_no_false_positives_for_identical_configs():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(40):
        diff = rng.normal(0, 1.0, N)
        r = compare.compare_auroc(_runs(0.6, diff, rng), _runs(0.6, diff, rng), "neg_model", 300, rng)
        ps.append(r["p"])
    assert np.mean(np.array(ps) < 0.05) <= 0.15        # ~5% expected; generous bound for 40 trials


def test_pass_comparison():
    rng = np.random.default_rng(2)
    easy = rng.random(N) < 0.65
    a = [{f"t{i}": bool(easy[i]) for i in range(N)} for _ in range(3)]
    b = [{f"t{i}": bool(easy[i] and rng.random() > 0.25) for i in range(N)} for _ in range(3)]  # B breaks ~25%
    r = compare.compare_pass(a, b, 1000, rng)
    assert r["d_pass"] > 0.1 and r["ci95"][0] > 0 and r["p"] < 0.01


def test_holm():
    adj = compare.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]              # 4*0.01, max(3*0.03, ..), monotone, 1*0.5


def test_default_pairs_match_closest_pass():
    groups = {("canonmark", 3.0): [{"pass@1": "0.64"}], ("canonmark", 5.0): [{"pass@1": "0.54"}],
              ("kgw", 1.0): [{"pass@1": "0.70"}], ("kgw", 2.0): [{"pass@1": "0.61"}], ("kgw", 3.0): [{"pass@1": "0.49"}]}
    assert compare.default_pairs(groups) == [(("canonmark", 3.0), ("kgw", 2.0)), (("canonmark", 5.0), ("kgw", 3.0))]


def test_pairing_is_tighter_than_separate_intervals():
    from canonmark.evaluation import key_averaged_auroc_ci
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 1.5, N)                       # strong shared difficulty -> pairing should help
    za, zb = _runs(0.9, diff, rng, noise=0.5), _runs(0.6, diff, rng, noise=0.5)
    paired = compare.compare_auroc(za, zb, "neg_model", 1000, rng)
    _, ca = key_averaged_auroc_ci(za, n_boot=1000)
    _, cb = key_averaged_auroc_ci(zb, n_boot=1000)
    unpaired_width = np.hypot(ca[1] - ca[0], cb[1] - cb[0])      # width if A and B were independent
    assert paired["ci95"][1] - paired["ci95"][0] < 0.8 * unpaired_width


def _write_sweep(d, method, delta, shift, rng, keys=("key-a", "key-b"), none_pass=None):
    import csv, json
    d.mkdir(parents=True)
    diff = np.random.default_rng(42).normal(0, 1.0, N)          # same problems in every sweep
    z, passes, rows = {}, {"none": none_pass or {f"t{i}": True for i in range(N)}}, []
    for k in keys:
        run = f"{method}_d{delta}_{k}"
        zr = _runs(shift, diff, rng, keys=1)[0]
        z[run] = {"clean": zr, "all_three": zr}
        passes[run] = {f"t{i}": bool(rng.random() < 0.6) for i in range(N)}
        rows.append({"run": run, "method": method, "delta": delta, "key": k, "pass@1": 0.6})
    with open(d / "sweep_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (d / "sweep_z.json").write_text(json.dumps(z))
    (d / "sweep_pass.json").write_text(json.dumps(passes))


def test_cross_sweep_comparison(tmp_path):
    import json, subprocess, sys
    rng = np.random.default_rng(5)
    _write_sweep(tmp_path / "sweep_good", "canonmark", 5, 1.2, rng)
    _write_sweep(tmp_path / "sweep_base", "canonmark", 5, 0.4, rng)
    script = Path(__file__).resolve().parent.parent / "scripts" / "compare.py"
    out = subprocess.run([sys.executable, str(script), "--sweep", str(tmp_path / "sweep_good"),
                          "--with", str(tmp_path / "sweep_base"), "--pair", "canonmark:5", "canonmark:5@sweep_base",
                          "--n_boot", "500"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "sweep_good" / "compare_cross.json").read_text())[0]
    assert res["a"].endswith("@sweep_good") and res["b"].endswith("@sweep_base")
    assert res["auroc"]["clean"]["d_auroc"] > 0.1 and res["auroc"]["clean"]["ci95"][0] > 0
    assert "pass" in res and "WARNING" not in out.stdout


def test_cross_sweep_warns_on_different_negatives(tmp_path):
    import subprocess, sys
    rng = np.random.default_rng(6)
    _write_sweep(tmp_path / "s1", "canonmark", 5, 1.0, rng)
    _write_sweep(tmp_path / "s2", "canonmark", 5, 1.0, rng, none_pass={f"t{i}": i % 2 == 0 for i in range(N)})
    script = Path(__file__).resolve().parent.parent / "scripts" / "compare.py"
    out = subprocess.run([sys.executable, str(script), "--sweep", str(tmp_path / "s1"), "--with", str(tmp_path / "s2"),
                          "--pair", "canonmark:5", "canonmark:5@s2", "--n_boot", "200"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "different unwatermarked runs" in out.stdout
    bad = subprocess.run([sys.executable, str(script), "--sweep", str(tmp_path / "s1"), "--pair", "canonmark:9", "canonmark:5"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "not found" in bad.stderr
