"""Evaluate generated runs: functionality, detectability and robustness.

Example:
  python scripts/evaluate.py --dataset humaneval \
      --runs outputs/he_canon.jsonl outputs/he_kgw.jsonl \
      --negatives outputs/he_plain.jsonl --out outputs/he_results.json

For each watermarked run it reports
  * Pass@1 and syntax-error rate (functionality)
  * mean z, AUROC and TPR@5%FPR against (a) unwatermarked model output and
    (b) human reference solutions (detectability)
  * the same detection metrics after each attack in canonmark.attacks,
    applied to both positives and negatives (robustness)
Negatives are scored with the run's own detector and key.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_auc_score  # noqa: E402

from canonmark.attacks import ATTACKS  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.detect import detect  # noqa: E402
from canonmark.execution import is_syntax_ok, run_many  # noqa: E402
from canonmark.keyed import KeyedGreen  # noqa: E402
from canonmark.switch import load_switch  # noqa: E402


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def make_detector(wm: dict):
    if wm["method"] == "canonmark":
        green = KeyedGreen(wm["key"], wm["gamma"], wm["context_width"])
        sw = load_switch(wm.get("switch"))
        return lambda code: detect(code, green, sw).z
    if wm["method"] == "kgw":
        from transformers import AutoTokenizer
        from canonmark.generate import KGWScheme, kgw_detect
        tok = AutoTokenizer.from_pretrained(wm["model"])
        scheme = KGWScheme(wm["key"], len(tok), wm["gamma"])
        return lambda code: kgw_detect(code, tok, scheme)["z"]
    if wm["method"] == "hybrid":
        from transformers import AutoTokenizer
        from canonmark.generate import KGWScheme, hybrid_detect
        tok = AutoTokenizer.from_pretrained(wm["model"])
        green = KeyedGreen(wm["key"], wm["gamma"], wm["context_width"])
        scheme = KGWScheme(wm["key"] + "/surface", len(tok), wm["gamma"])
        sw = load_switch(wm.get("switch"))
        return lambda code: hybrid_detect(code, tok, green, scheme, sw)["z"]
    raise ValueError(wm["method"])


def det_metrics(pos_z, neg_z):
    if not pos_z or not neg_z:
        return {}
    y = [1] * len(pos_z) + [0] * len(neg_z)
    auc = roc_auc_score(y, pos_z + neg_z)
    thr = sorted(neg_z)[int(0.95 * (len(neg_z) - 1))]          # empirical 5% FPR threshold
    return {"auroc": round(auc, 4),
            "tpr@5%fpr": round(sum(z > thr for z in pos_z) / len(pos_z), 4),
            "tpr@z>4": round(sum(z > 4 for z in pos_z) / len(pos_z), 4),
            "fpr@z>4": round(sum(z > 4 for z in neg_z) / len(neg_z), 4),
            "mean_z_pos": round(statistics.mean(pos_z), 3),
            "mean_z_neg": round(statistics.mean(neg_z), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="watermarked run jsonl files")
    ap.add_argument("--negatives", help="unwatermarked run jsonl (same model)")
    ap.add_argument("--no_exec", action="store_true", help="skip running tests")
    ap.add_argument("--attacks", nargs="*", default=list(ATTACKS))
    ap.add_argument("--out")
    a = ap.parse_args()

    probs = {p["task_id"]: p for p in load(a.dataset)}
    neg_rows = read_jsonl(a.negatives) if a.negatives else []
    results = {}

    def functionality(rows):
        ps = [probs[r["task_id"]] for r in rows]
        codes = [r["code"] for r in rows]
        res = {"n": len(rows), "syntax_error_rate": round(1 - sum(map(is_syntax_ok, codes)) / len(codes), 4)}
        if not a.no_exec:
            ran = run_many(codes, ps)
            res["pass@1"] = round(sum(ok for ok, _ in ran) / len(ran), 4)
        return res

    if neg_rows:
        results["unwatermarked"] = functionality(neg_rows)

    for path in a.runs:
        rows = read_jsonl(path)
        wm = rows[0]["wm"]
        name = Path(path).stem
        print(f"== {name} ({wm['method']})", flush=True)
        score = make_detector(wm)
        res = functionality(rows)
        human = [probs[r["task_id"]]["reference"] for r in rows]
        sets = {"pos": [r["code"] for r in rows], "neg_model": [r["code"] for r in neg_rows], "neg_human": human}
        for attack in ["clean"] + a.attacks:
            fn = (lambda c: c) if attack == "clean" else ATTACKS[attack]
            z = {}
            for k, codes in sets.items():
                z[k] = []
                for c in codes:
                    c2 = fn(c) if c.strip() else None
                    z[k].append(score(c2 if c2 is not None else c))   # attack not applicable -> unchanged
            entry = {}
            if z["neg_model"]:
                entry["vs_model"] = det_metrics(z["pos"], z["neg_model"])
            entry["vs_human"] = det_metrics(z["pos"], z["neg_human"])
            res[attack] = entry
        results[name] = res

    print(json.dumps(results, indent=1))
    # compact table
    print("\nmethod".ljust(22) + "pass@1  syntax  " + "  ".join(f"{t:>11s}" for t in ["clean"] + a.attacks)
          + "   (AUROC vs unwatermarked model output; vs human if none)")
    for name, r in results.items():
        if name == "unwatermarked":
            print(f"{name:21s} {r.get('pass@1', float('nan')):6.3f}  {r['syntax_error_rate']:6.3f}")
            continue
        cells = []
        for t in ["clean"] + a.attacks:
            m = r[t].get("vs_model") or r[t]["vs_human"]
            cells.append(f"{m.get('auroc', float('nan')):11.3f}")
        print(f"{name:21s} {r.get('pass@1', float('nan')):6.3f}  {r['syntax_error_rate']:6.3f}  " + "  ".join(cells))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
