"""Sweep watermark strength (delta) x secret key for each method, then evaluate everything.

The model is loaded once. One unwatermarked run serves as the negative set and
the McNemar baseline for every configuration. Run files that already exist are
reused, so an interrupted sweep resumes where it stopped (incomplete files are
written as *.partial and never reused).

Example (GPU):
  python scripts/sweep.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset humaneval \
      --switch outputs/switch_k3 --keys k1 k2 k3 \
      --canon_deltas 3 4 5 --hybrid_deltas 5 --kgw_deltas 1 2 3 --outdir outputs/sweep_he

Outputs in --outdir:
  runs/*.jsonl          one run file per configuration
  sweep_runs.csv        one row per (method, delta, key)
  sweep_summary.csv/.md mean ± sd over keys per (method, delta); use it to plot
                        Pass@1 against AUROC after attacks (the trade-off curve)
  sweep_results.json    everything, including per-run detection details
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonmark.attacks import ATTACKS  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.evaluation import (functionality, key_averaged_auroc_ci, make_detector, mcnemar,  # noqa: E402
                                  read_jsonl, robustness)
from canonmark.pipeline import wm_config  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="humaneval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", nargs="+", default=["key-a", "key-b", "key-c"])
    ap.add_argument("--canon_deltas", nargs="*", type=float, default=[3, 4, 5])
    ap.add_argument("--hybrid_deltas", nargs="*", type=float, default=[5])
    ap.add_argument("--delta_surface", type=float, default=2.0, help="hybrid surface-channel delta")
    ap.add_argument("--kgw_deltas", nargs="*", type=float, default=[1, 2, 3])
    ap.add_argument("--switch", default="always", help="switch for canonmark/hybrid ('always' or a path)")
    ap.add_argument("--carrier_gate", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--context_width", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attacks", nargs="*", default=list(ATTACKS))
    ap.add_argument("--no_exec", action="store_true", help="skip running tests (no Pass@1 / McNemar)")
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap rounds for AUROC CIs (0 = off)")
    ap.add_argument("--eval_only", action="store_true", help="only evaluate run files that already exist")
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def fmt_delta(d: float) -> str:
    return f"{d:g}".replace(".", "p")


def configs(a) -> list[tuple[str, dict]]:
    common = dict(gamma=a.gamma, context_width=a.context_width, top_k=a.top_k)
    out = [("none", wm_config("none", a.model, **common))]
    for method, deltas in (("canonmark", a.canon_deltas), ("hybrid", a.hybrid_deltas), ("kgw", a.kgw_deltas)):
        for d in deltas:
            for key in a.keys:
                wm = wm_config(method, a.model, key=key, delta=d, **common,
                               delta_surface=a.delta_surface if method == "hybrid" else None,
                               switch=a.switch if method != "kgw" else "always",
                               carrier_gate=a.carrier_gate if method != "kgw" else False)
                out.append((f"{method}_d{fmt_delta(d)}_{key}", wm))
    return out


def main():
    a = parse()
    outdir = Path(a.outdir)
    rundir = outdir / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    probs_list = load(a.dataset)
    if a.limit:
        probs_list = probs_list[:a.limit]
    probs = {p["task_id"]: p for p in probs_list}
    plan = configs(a)
    (outdir / "sweep_config.json").write_text(json.dumps(vars(a), indent=1))

    # ---- 1. generation (model loaded lazily, once) ----------------------------
    todo = [(name, wm) for name, wm in plan if not (rundir / f"{name}.jsonl").is_file()]
    if todo and not a.eval_only:
        from canonmark.pipeline import generate_file, load_model
        model, tok, device = load_model(a.model, a.dtype)
        print(f"generating {len(todo)} of {len(plan)} runs on {device}", flush=True)
        for i, (name, wm) in enumerate(todo, 1):
            print(f"--- [{i}/{len(todo)}] {name}", flush=True)
            last = {}
            generate_file(model, tok, probs_list, wm, rundir / f"{name}.jsonl", a.batch_size, a.temperature,
                          a.top_p, a.max_new_tokens, a.seed, log=lambda s: last.update(msg=s))
            print(f"    {last.get('msg', '')}", flush=True)
        del model

    # ---- 2. evaluation ----------------------------------------------------------
    none_path = rundir / "none.jsonl"
    if not none_path.is_file():
        raise SystemExit(f"{none_path} is missing; run without --eval_only first")
    neg_rows = read_jsonl(none_path)
    base, base_pass = functionality(neg_rows, probs, not a.no_exec)
    print(f"unwatermarked: {base}", flush=True)
    results = {"unwatermarked": base}
    rows_out = []
    z_store = {}
    for name, wm in plan[1:]:
        path = rundir / f"{name}.jsonl"
        if not path.is_file():
            print(f"skip {name} (no run file)")
            continue
        rows = read_jsonl(path)
        res, passed = functionality(rows, probs, not a.no_exec)
        if base_pass and passed:
            res["mcnemar_vs_unwatermarked"] = mcnemar(base_pass, passed)
        res.update(robustness(rows, neg_rows, probs, make_detector(rows[0]["wm"]), a.attacks,
                              n_boot=a.n_boot, keep_z=True))
        z_store[name] = {t: res[t].pop("z") for t in ["clean"] + a.attacks}
        results[name] = res
        mc = res.get("mcnemar_vs_unwatermarked", {})
        rec = {"run": name, "method": wm["method"], "delta": wm["delta"], "key": wm["key"],
               "pass@1": res.get("pass@1"), "delta_pass@1": mc.get("delta_pass@1"),
               "mcnemar_lost": mc.get("lost"), "mcnemar_gained": mc.get("gained"), "mcnemar_p": mc.get("p_value"),
               "syntax_error_rate": res["syntax_error_rate"]}
        for t in ["clean"] + a.attacks:
            m = res[t].get("vs_model") or res[t]["vs_human"]
            rec[f"auroc_{t}"] = m.get("auroc")
            ci = m.get("auroc_ci95") or [None, None]
            rec[f"auroc_{t}_lo"], rec[f"auroc_{t}_hi"] = ci
            rec[f"tpr5_{t}"] = m.get("tpr@5%fpr")
        rec["mean_z_pos"] = (res["clean"].get("vs_model") or res["clean"]["vs_human"]).get("mean_z_pos")
        rows_out.append(rec)
        print(f"{name:32s} pass@1 {rec['pass@1']}  Δ {rec['delta_pass@1']}  p {rec['mcnemar_p']}  "
              f"AUROC clean {rec['auroc_clean']}  all-attacks {rec.get('auroc_all_three')}", flush=True)

    (outdir / "sweep_results.json").write_text(json.dumps(results, indent=1))
    (outdir / "sweep_z.json").write_text(json.dumps(z_store))     # raw per-task z, for re-analysis
    if not rows_out:
        return
    with open(outdir / "sweep_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    write_summary(rows_out, base, a.attacks, outdir, z_store, a.n_boot)


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def write_summary(rows, base, attacks, outdir: Path, z_store=None, n_boot: int = 1000):
    metric_cols = ["pass@1", "delta_pass@1", "syntax_error_rate", "mean_z_pos"] + [f"auroc_{t}" for t in ["clean"] + attacks]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["method"], r["delta"]), []).append(r)
    summary = []
    for (method, delta), rs in sorted(groups.items()):
        s = {"method": method, "delta": delta, "n_keys": len(rs),
             "keys_sig_p<0.05": sum(1 for r in rs if r["mcnemar_p"] is not None and r["mcnemar_p"] < 0.05)}
        for c in metric_cols:
            m, sd = _ms([r[c] for r in rs])
            s[f"{c}_mean"], s[f"{c}_sd"] = (round(m, 4) if m is not None else None), (round(sd, 4) if sd is not None else None)
        for t in ["clean"] + attacks:              # problem-level bootstrap CI of the key-averaged AUROC
            lo = hi = None
            if z_store and n_boot > 0:
                zr = [z_store[r["run"]][t] for r in rs if r["run"] in z_store]
                if zr and all(z["neg_model"] for z in zr):
                    _, ci = key_averaged_auroc_ci(zr, "neg_model", n_boot)
                    lo, hi = (ci or [None, None])
            s[f"auroc_{t}_ci_lo"], s[f"auroc_{t}_ci_hi"] = lo, hi
        summary.append(s)
    with open(outdir / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    def cell(s, c, signed=False):
        m, sd = s[f"{c}_mean"], s[f"{c}_sd"]
        if m is None:
            return "–"
        txt = (f"{m:+.3f}" if signed else f"{m:.3f}") + (f" ± {sd:.3f}" if s["n_keys"] > 1 else "")
        lo, hi = s.get(f"{c}_ci_lo".replace("_mean", "")), s.get(f"{c}_ci_hi")
        if c.startswith("auroc_") and lo is not None:
            txt += f" [{lo:.2f}, {hi:.2f}]"
        return txt

    cols = ["clean"] + attacks
    lines = [f"Unwatermarked: Pass@1 {base.get('pass@1', '–')}, syntax errors {base['syntax_error_rate']}", "",
             "| method | δ | keys | Pass@1 | ΔPass@1 | McNemar p<0.05 | mean z | "
             + " | ".join(f"AUROC {c}" for c in cols) + " |",
             "|" + "---|" * (7 + len(cols))]
    for s in summary:
        lines.append(f"| {s['method']} | {s['delta']:g} | {s['n_keys']} | {cell(s, 'pass@1')} | "
                     f"{cell(s, 'delta_pass@1', True)} | {s['keys_sig_p<0.05']}/{s['n_keys']} | {cell(s, 'mean_z_pos')} | "
                     + " | ".join(cell(s, f"auroc_{c}") for c in cols) + " |")
    lines += ["", "Mean ± sd over keys; [..] = 95% problem-level bootstrap CI of the key-averaged AUROC.",
              "AUROC against the same model's unwatermarked output.",
              "McNemar: exact paired test of Pass@1 per key against the unwatermarked run."]
    (outdir / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
