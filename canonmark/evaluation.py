"""Metrics shared by scripts/evaluate.py and scripts/sweep.py.

* functionality: Pass@1 and syntax-error rate, with per-task pass results
* mcnemar: exact paired test of Pass@1 against the unwatermarked run
* robustness: AUROC / TPR@5%FPR / mean z before and after each attack,
  against unwatermarked model output and against human references
"""
from __future__ import annotations

import json
import math
import statistics
from functools import lru_cache

from .attacks import ATTACKS
from .detect import detect
from .execution import is_syntax_ok, run_many
from .keyed import KeyedGreen
from .switch import load_switch, verify_switch


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in open(p)]


@lru_cache(maxsize=8)
def _tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def make_detector(wm: dict, source: str | None = None, weighted: bool = False):
    """code -> z, using the run's own method, key and settings.

    weighted: use the switch-confidence-weighted z for the canonical channel
    (canonmark and hybrid; no effect on KGW).
    Raises SwitchMismatchError if the run's switch file changed after generation."""
    verify_switch(wm, source)
    m = wm["method"]
    if m == "canonmark":
        green = KeyedGreen(wm["key"], wm["gamma"], wm["context_width"])
        sw = load_switch(wm.get("switch"))
        return lambda code: detect(code, green, sw, weighted=weighted).z
    from .generate import KGWScheme, hybrid_detect, kgw_detect
    tok = _tokenizer(wm["model"])
    if m == "kgw":
        scheme = KGWScheme(wm["key"], len(tok), wm["gamma"])
        return lambda code: kgw_detect(code, tok, scheme)["z"]
    if m == "hybrid":
        green = KeyedGreen(wm["key"], wm["gamma"], wm["context_width"])
        scheme = KGWScheme(wm["key"] + "/surface", len(tok), wm["gamma"])
        sw = load_switch(wm.get("switch"))
        return lambda code: hybrid_detect(code, tok, green, scheme, sw, weighted=weighted)["z"]
    raise ValueError(m)


# --------------------------------------------------------------------------
# functionality
# --------------------------------------------------------------------------
def functionality(rows: list[dict], problems: dict, execute: bool = True) -> tuple[dict, dict]:
    """Return (summary, {task_id: passed}). `passed` is empty when execute=False."""
    codes = [r["code"] for r in rows]
    summary = {"n": len(rows), "syntax_error_rate": round(1 - sum(map(is_syntax_ok, codes)) / max(1, len(codes)), 4)}
    passed = {}
    if execute:
        ran = run_many(codes, [problems[r["task_id"]] for r in rows])
        passed = {r["task_id"]: ok for r, (ok, _) in zip(rows, ran)}
        summary["pass@1"] = round(sum(passed.values()) / max(1, len(passed)), 4)
    return summary, passed


def mcnemar(baseline: dict, treated: dict) -> dict:
    """Exact (binomial) McNemar test on tasks present in both runs.

    b = passed without watermark but failed with it, c = the reverse.
    Under H0 (watermark does not change pass rate) b ~ Binomial(b + c, 1/2).
    Pairing on the same problems makes this far more sensitive than comparing
    two independent Pass@1 numbers.
    """
    common = sorted(set(baseline) & set(treated))
    b = sum(baseline[t] and not treated[t] for t in common)
    c = sum(treated[t] and not baseline[t] for t in common)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    k = max(1, len(common))
    return {"n_pairs": len(common), "lost": b, "gained": c,
            "delta_pass@1": round((c - b) / k, 4), "p_value": round(p, 6)}


# --------------------------------------------------------------------------
# detection / robustness
# --------------------------------------------------------------------------
def _win_matrix(pos_z, neg_z):
    """M[i, j] = 1 if pos i outranks neg j, 0.5 on ties. AUROC = M.mean()."""
    import numpy as np
    p = np.asarray(pos_z, dtype=float)[:, None]
    n = np.asarray(neg_z, dtype=float)[None, :]
    return (p > n) + 0.5 * (p == n)


def auroc_ci(pos_z, neg_z, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> list[float]:
    """Percentile bootstrap CI for AUROC, resampling positives and negatives independently."""
    import numpy as np
    if n_boot <= 0 or not pos_z or not neg_z:
        return []
    m = _win_matrix(pos_z, neg_z)
    rng = np.random.default_rng(seed)
    ip = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    ineg = rng.integers(0, m.shape[1], size=(n_boot, m.shape[1]))
    # row means of the resampled positives, then averaged over resampled negatives
    stats = np.array([m[ip[b]][:, ineg[b]].mean() for b in range(n_boot)])
    a = (1 - level) / 2
    return [round(float(np.quantile(stats, a)), 4), round(float(np.quantile(stats, 1 - a)), 4)]


def det_metrics(pos_z: list[float], neg_z: list[float], n_boot: int = 1000) -> dict:
    if not pos_z or not neg_z:
        return {}
    auc = float(_win_matrix(pos_z, neg_z).mean())
    thr = sorted(neg_z)[int(0.95 * (len(neg_z) - 1))]           # empirical 5% FPR threshold
    return {"auroc": round(auc, 4), "auroc_ci95": auroc_ci(pos_z, neg_z, n_boot),
            "tpr@5%fpr": round(sum(z > thr for z in pos_z) / len(pos_z), 4),
            "tpr@z>4": round(sum(z > 4 for z in pos_z) / len(pos_z), 4),
            "fpr@z>4": round(sum(z > 4 for z in neg_z) / len(neg_z), 4),
            "mean_z_pos": round(statistics.mean(pos_z), 3),
            "mean_z_neg": round(statistics.mean(neg_z), 3)}


def robustness(rows: list[dict], neg_rows: list[dict], problems: dict, score,
               attacks: list[str] | None = None, n_boot: int = 1000, keep_z: bool = False) -> dict:
    """Detection metrics on clean code and after each attack (applied to positives and negatives).

    keep_z=True also stores the raw per-task z-scores (used by the sweep's
    key-averaged bootstrap)."""
    attacks = list(ATTACKS) if attacks is None else attacks
    sets = {"pos": [r["code"] for r in rows], "neg_model": [r["code"] for r in neg_rows],
            "neg_human": [problems[r["task_id"]]["reference"] for r in rows]}
    out = {}
    for attack in ["clean"] + attacks:
        fn = (lambda c: c) if attack == "clean" else ATTACKS[attack]
        z = {}
        for k, codes in sets.items():
            z[k] = []
            for c in codes:
                c2 = fn(c) if c.strip() else None
                z[k].append(score(c2 if c2 is not None else c))   # attack not applicable -> unchanged
        entry = {}
        if z["neg_model"]:
            entry["vs_model"] = det_metrics(z["pos"], z["neg_model"], n_boot)
        entry["vs_human"] = det_metrics(z["pos"], z["neg_human"], n_boot)
        if keep_z:
            entry["z"] = {"pos": dict(zip((r["task_id"] for r in rows), z["pos"])),
                          "neg_model": dict(zip((r["task_id"] for r in neg_rows), z["neg_model"])),
                          "neg_human": dict(zip((r["task_id"] for r in rows), z["neg_human"]))}
        out[attack] = entry
    return out


def key_averaged_auroc_ci(z_runs: list[dict], neg: str = "neg_model", n_boot: int = 1000, seed: int = 0,
                          level: float = 0.95) -> tuple[float, list[float]]:
    """AUROC averaged over several runs (e.g. secret keys) with a problem-level bootstrap CI.

    z_runs: one {"pos": {task: z}, "neg_model": {task: z}, ...} per run. Each bootstrap
    round resamples problems once and applies the same sample to every run, so the
    interval reflects uncertainty from the benchmark's finite set of problems."""
    import numpy as np
    tasks = sorted(set.intersection(*(set(zr["pos"]) & set(zr[neg]) for zr in z_runs)))
    if not tasks:
        return float("nan"), []
    mats = [_win_matrix([zr["pos"][t] for t in tasks], [zr[neg][t] for t in tasks]) for zr in z_runs]
    point = float(np.mean([m.mean() for m in mats]))
    rng = np.random.default_rng(seed)
    n = len(tasks)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ip, ineg = rng.integers(0, n, n), rng.integers(0, n, n)
        stats[b] = np.mean([m[ip][:, ineg].mean() for m in mats])
    a = (1 - level) / 2
    return round(point, 4), [round(float(np.quantile(stats, a)), 4), round(float(np.quantile(stats, 1 - a)), 4)]
