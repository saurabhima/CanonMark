"""Evaluate generated runs: functionality, detectability and robustness.

Example:
  python scripts/evaluate.py --dataset humaneval \
      --runs outputs/he_canon.jsonl outputs/he_kgw.jsonl \
      --negatives outputs/he_none.jsonl --out outputs/he_results.json

For each watermarked run it reports
  * Pass@1 and syntax-error rate (functionality)
  * McNemar's exact test of Pass@1 against the unwatermarked run, paired by problem
  * mean z, AUROC and TPR@5%FPR against (a) unwatermarked model output and
    (b) human reference solutions (detectability)
  * the same detection metrics after each attack in canonmark.attacks,
    applied to both positives and negatives (robustness)
Negatives are scored with the run's own detector and key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonmark.attacks import ATTACKS  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.evaluation import functionality, make_detector, mcnemar, read_jsonl, robustness  # noqa: E402


def check_inputs(paths, dataset):
    missing = [p for p in paths if not Path(p).is_file()]
    if not missing:
        return
    hints = {"none": "--method none", "canon": "--method canonmark", "hybrid": "--method hybrid",
             "kgw": "--method kgw --delta 2"}
    msg = ["Missing input file(s). Create them with scripts/generate.py first:"]
    for p in missing:
        flag = next((v for k, v in hints.items() if k in Path(p).stem), "--method <none|canonmark|hybrid|kgw>")
        msg.append(f"  python scripts/generate.py --model $M --dataset {dataset} {flag} --out {p}")
    raise SystemExit("\n".join(msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="watermarked run jsonl files")
    ap.add_argument("--negatives", help="unwatermarked run jsonl (same model)")
    ap.add_argument("--no_exec", action="store_true", help="skip running tests")
    ap.add_argument("--attacks", nargs="*", default=list(ATTACKS))
    ap.add_argument("--weighted_detection", action="store_true",
                    help="weight canonical carriers by switch confidence (canonmark / hybrid)")
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap rounds for AUROC 95%% CIs (0 = off)")
    ap.add_argument("--out")
    a = ap.parse_args()
    check_inputs(a.runs + ([a.negatives] if a.negatives else []), a.dataset)

    probs = {p["task_id"]: p for p in load(a.dataset)}
    neg_rows = read_jsonl(a.negatives) if a.negatives else []
    results = {}
    base_pass = {}
    if neg_rows:
        results["unwatermarked"], base_pass = functionality(neg_rows, probs, not a.no_exec)

    for path in a.runs:
        rows = read_jsonl(path)
        wm = rows[0]["wm"]
        name = Path(path).stem
        print(f"== {name} ({wm['method']})", flush=True)
        res, passed = functionality(rows, probs, not a.no_exec)
        if base_pass and passed:
            res["mcnemar_vs_unwatermarked"] = mcnemar(base_pass, passed)
        res.update(robustness(rows, neg_rows, probs, make_detector(wm, source=path, weighted=a.weighted_detection), a.attacks, n_boot=a.n_boot))
        results[name] = res

    print(json.dumps(results, indent=1))
    print_table(results, a.attacks)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


def print_table(results, attacks):
    cols = ["clean"] + attacks
    print("\n" + "method".ljust(26) + "pass@1   Δpass  McNemar p  syntax  " + "  ".join(f"{t:>14s}" for t in cols))
    print(" " * 59 + "(AUROC vs unwatermarked model output; vs human if no negatives)")
    for name, r in results.items():
        p1 = r.get("pass@1", float("nan"))
        if name == "unwatermarked":
            print(f"{name:25s} {p1:6.3f}  {'':6s}  {'':9s}  {r['syntax_error_rate']:6.3f}")
            continue
        mc = r.get("mcnemar_vs_unwatermarked", {})
        dp = f"{mc['delta_pass@1']:+6.3f}" if mc else " " * 6
        pv = f"{mc['p_value']:9.4f}" if mc else " " * 9
        cells = []
        for t in cols:
            m = r[t].get("vs_model") or r[t]["vs_human"]
            cells.append(f"{m.get('auroc', float('nan')):14.3f}")
        print(f"{name:25s} {p1:6.3f}  {dp}  {pv}  {r['syntax_error_rate']:6.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
