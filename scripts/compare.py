"""Paired comparison of two sweep configurations on the same problems.

For a pair A = (method, delta) and B = (method, delta) it reports, per condition
(clean code, after each attack):
  * AUROC_A, AUROC_B: averaged over each configuration's keys
  * dAUROC = AUROC_A - AUROC_B with a 95% paired-bootstrap CI and a two-sided p-value
and once per pair:
  * dPass@1 (averaged over keys) with a paired-bootstrap CI and p-value
    (needs sweep_pass.json; rerun the sweep with --eval_only to create it)

Paired: every bootstrap round resamples the problems once and applies the same
sample to both configurations and all their keys, so problem-to-problem
difficulty cancels out. That is much tighter than checking whether two separate
confidence intervals overlap. Holm's correction is applied across all AUROC tests
(and separately across Pass@1 tests) of one invocation.

Examples:
  # default: each canonmark / hybrid delta vs the KGW delta with the closest Pass@1
  python scripts/compare.py --sweep outputs/sweep_he
  # explicit pairs
  python scripts/compare.py --sweep outputs/sweep_he --pair canonmark:3 kgw:2 --pair canonmark:5 kgw:3 \
      --attacks clean all_three
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonmark.evaluation import _win_matrix  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep output directory")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="method:delta method:delta, e.g. canonmark:3 kgw:2 (repeatable)")
    ap.add_argument("--attacks", nargs="*", help="conditions to compare (default: all in the sweep)")
    ap.add_argument("--negatives", default="neg_model", choices=["neg_model", "neg_human"],
                    help="negative set: unwatermarked model output (default) or human references")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output prefix (default: <sweep>/compare)")
    return ap.parse_args()


def load_sweep(d: Path):
    runs = list(csv.DictReader(open(d / "sweep_runs.csv")))
    z = json.loads((d / "sweep_z.json").read_text())
    pf = d / "sweep_pass.json"
    passes = json.loads(pf.read_text()) if pf.is_file() else None
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in runs:
        if r["run"] in z:
            groups.setdefault((r["method"], float(r["delta"])), []).append(r)
    return groups, z, passes


def parse_cfg(s: str) -> tuple[str, float]:
    m, _, d = s.partition(":")
    if not d:
        raise SystemExit(f"bad configuration {s!r}; use method:delta, e.g. kgw:2")
    return m, float(d)


def default_pairs(groups):
    """Each non-KGW configuration vs the KGW delta with the closest mean Pass@1 (matched functionality)."""
    def mean_pass(rs):
        v = [float(r["pass@1"]) for r in rs if r["pass@1"] not in ("", "None")]
        return float(np.mean(v)) if v else None
    kgw = {k: mean_pass(v) for k, v in groups.items() if k[0] == "kgw"}
    pairs = []
    for k, rs in sorted(groups.items()):
        if k[0] == "kgw" or not kgw:
            continue
        pa = mean_pass(rs)
        if pa is None:
            continue
        best = min((kk for kk in kgw if kgw[kk] is not None), key=lambda kk: abs(kgw[kk] - pa), default=None)
        if best is not None:
            pairs.append((k, best))
    return pairs


def p_two_sided(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0."""
    n = len(boot)
    lo = (np.sum(boot <= 0) + 1) / (n + 1)
    hi = (np.sum(boot >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def compare_auroc(za: list[dict], zb: list[dict], neg: str, n_boot: int, rng) -> dict:
    tasks = sorted(set.intersection(*(set(r["pos"]) & set(r[neg]) for r in za + zb)))
    mats_a = [_win_matrix([r["pos"][t] for t in tasks], [r[neg][t] for t in tasks]) for r in za]
    mats_b = [_win_matrix([r["pos"][t] for t in tasks], [r[neg][t] for t in tasks]) for r in zb]
    a = float(np.mean([m.mean() for m in mats_a]))
    b = float(np.mean([m.mean() for m in mats_b]))
    n = len(tasks)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        ip, ineg = rng.integers(0, n, n), rng.integers(0, n, n)   # shared by A, B and all keys
        boot[i] = (np.mean([m[ip][:, ineg].mean() for m in mats_a])
                   - np.mean([m[ip][:, ineg].mean() for m in mats_b]))
    return {"auroc_a": round(a, 4), "auroc_b": round(b, 4), "d_auroc": round(a - b, 4),
            "ci95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
            "p": p_two_sided(boot), "n_problems": n}


def compare_pass(pa: list[dict], pb: list[dict], n_boot: int, rng) -> dict:
    tasks = sorted(set.intersection(*(set(p) for p in pa + pb)))
    xa = np.mean([[float(p[t]) for t in tasks] for p in pa], axis=0)    # per-problem pass rate over keys
    xb = np.mean([[float(p[t]) for t in tasks] for p in pb], axis=0)
    d = xa - xb
    n = len(tasks)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"pass_a": round(float(xa.mean()), 4), "pass_b": round(float(xb.mean()), 4),
            "d_pass": round(float(d.mean()), 4),
            "ci95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
            "p": p_two_sided(boot), "n_problems": n}


def name(cfg):
    return f"{cfg[0]} δ={cfg[1]:g}"


def main():
    a = parse()
    d = Path(a.sweep)
    groups, z, passes = load_sweep(d)
    pairs = [(parse_cfg(x), parse_cfg(y)) for x, y in a.pair] if a.pair else default_pairs(groups)
    if not pairs:
        raise SystemExit("no pairs to compare (give --pair method:delta method:delta)")
    for cfg in {c for p in pairs for c in p}:
        if cfg not in groups:
            raise SystemExit(f"{name(cfg)} not in this sweep; available: {', '.join(name(k) for k in sorted(groups))}")
    any_run = next(iter(z.values()))
    attacks = a.attacks or list(any_run)
    rng = np.random.default_rng(a.seed)

    results = []
    for A, B in pairs:
        ra, rb = groups[A], groups[B]
        entry = {"a": name(A), "b": name(B), "keys_a": len(ra), "keys_b": len(rb), "auroc": {}}
        for t in attacks:
            za = [z[r["run"]][t] for r in ra]
            zb = [z[r["run"]][t] for r in rb]
            entry["auroc"][t] = compare_auroc(za, zb, a.negatives, a.n_boot, rng)
        if passes is not None and all(r["run"] in passes and passes[r["run"]] for r in ra + rb):
            entry["pass"] = compare_pass([passes[r["run"]] for r in ra], [passes[r["run"]] for r in rb], a.n_boot, rng)
        results.append(entry)

    auc_tests = [(i, t) for i, e in enumerate(results) for t in e["auroc"]]
    for (i, t), padj in zip(auc_tests, holm([results[i]["auroc"][t]["p"] for i, t in auc_tests])):
        results[i]["auroc"][t]["p_holm"] = padj
    pass_tests = [i for i, e in enumerate(results) if "pass" in e]
    for i, padj in zip(pass_tests, holm([results[i]["pass"]["p"] for i in pass_tests])):
        results[i]["pass"]["p_holm"] = padj

    lines = [f"Paired bootstrap ({a.n_boot} rounds, problems resampled jointly for both configurations and all keys). "
             f"Negatives: {'unwatermarked model output' if a.negatives == 'neg_model' else 'human references'}. "
             "p_holm: Holm-corrected across this table's AUROC tests (Pass@1 tests corrected separately).", ""]
    if not a.pair:
        lines += ["Pairs chosen automatically: each configuration vs the KGW δ with the closest mean Pass@1.", ""]
    lines += ["| A | B | condition | AUROC A | AUROC B | ΔAUROC [95% CI] | p | p_holm |", "|---|---|---|---|---|---|---|---|"]
    for e in results:
        for t, r in e["auroc"].items():
            lines.append(f"| {e['a']} | {e['b']} | {t} | {r['auroc_a']:.3f} | {r['auroc_b']:.3f} | "
                         f"{r['d_auroc']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.4f} | {r['p_holm']:.4f} |")
    if pass_tests:
        lines += ["", "| A | B | Pass@1 A | Pass@1 B | ΔPass@1 [95% CI] | p | p_holm |", "|---|---|---|---|---|---|---|"]
        for i in pass_tests:
            e, r = results[i], results[i]["pass"]
            lines.append(f"| {e['a']} | {e['b']} | {r['pass_a']:.3f} | {r['pass_b']:.3f} | "
                         f"{r['d_pass']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.4f} | {r['p_holm']:.4f} |")
    elif passes is None:
        lines += ["", "(No sweep_pass.json: rerun the sweep with --eval_only to add paired Pass@1 comparisons.)"]
    out = Path(a.out) if a.out else d / "compare"
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
