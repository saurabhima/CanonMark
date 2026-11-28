"""Why does one secret key hurt functionality more than the others?

For one configuration (method:delta) of a sweep and one suspect key, this finds the
problems the key breaks: they pass unwatermarked and with most other keys, but fail
with this key. For every problem it locates the first point where the watermarked
code departs from the unwatermarked code in canonical form, and records which
carrier the model would have written, what the watermark wrote instead, and the
context. It then ranks the substitutions that most often precede a break, and checks
whether each one is green (and switched on) under the suspect key but not under the
others, which would show the damage is specific to the key.

Example:
  python scripts/diagnose_key.py --sweep outputs/sweep_he_own --config canonmark:5 --key key-c
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonmark.canon import CanonRules, canonical_stream, texts  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.evaluation import functionality, read_jsonl  # noqa: E402
from canonmark.execution import run_tests  # noqa: E402
from canonmark.keyed import KeyedGreen  # noqa: E402
from canonmark.switch import AlwaysOn, load_switch, verify_switch  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--config", required=True, help="method:delta, e.g. canonmark:5")
    ap.add_argument("--key", required=True, help="the suspect key, e.g. key-c")
    ap.add_argument("--ref_keys", nargs="*", help="keys to compare against (default: all other keys of the config)")
    ap.add_argument("--dataset", default="humaneval")
    ap.add_argument("--top", type=int, default=15, help="substitutions to list")
    ap.add_argument("--examples", type=int, default=8, help="broken problems to show in detail")
    ap.add_argument("--out", help="report path (default: <sweep>/diagnose_<method>_d<delta>_<key>.md)")
    return ap.parse_args()


def line_at(code: str, pos: int) -> str:
    s = code.rfind("\n", 0, pos) + 1
    e = code.find("\n", pos)
    return code[s:e if e >= 0 else len(code)].rstrip()


def first_divergence(ref_code: str, wm_code: str, rules=None):
    """Index of the first differing canonical lexeme and the two streams (None if identical)."""
    a, b = canonical_stream(ref_code, rules), canonical_stream(wm_code, rules)
    ta, tb = texts(a), texts(b)
    i = 0
    while i < min(len(ta), len(tb)) and ta[i] == tb[i]:
        i += 1
    if i == len(ta) == len(tb):
        return None
    return i, a, b


def scheme_of(wm: dict, source: str):
    verify_switch(wm, source)
    green = KeyedGreen(wm["key"], wm["gamma"], wm["context_width"])
    sw = load_switch(wm.get("switch")) if wm["method"] in ("canonmark", "hybrid") else AlwaysOn()
    return green, sw


def main():
    a = parse()
    d = Path(a.sweep)
    method, _, delta = a.config.partition(":")
    delta = float(delta)
    runs = [r for r in csv.DictReader(open(d / "sweep_runs.csv"))
            if r["method"] == method and float(r["delta"]) == delta]
    if not runs:
        raise SystemExit(f"{a.config} not found in {d / 'sweep_runs.csv'}")
    by_key = {r["key"]: r["run"] for r in runs}
    if a.key not in by_key:
        raise SystemExit(f"key {a.key!r} not in this configuration; keys: {', '.join(by_key)}")
    refs = a.ref_keys or [k for k in by_key if k != a.key]
    if not refs:
        raise SystemExit("need at least one reference key")

    probs = {p["task_id"]: p for p in load(a.dataset)}
    none_rows = {r["task_id"]: r for r in read_jsonl(d / "runs" / "none.jsonl")}
    rows = {k: {r["task_id"]: r for r in read_jsonl(d / "runs" / f"{by_key[k]}.jsonl")} for k in [a.key] + refs}

    pf = d / "sweep_pass.json"
    passes = json.loads(pf.read_text()) if pf.is_file() else {}
    def passed(name, rws):
        if passes.get(name):
            return passes[name]
        return functionality(list(rws.values()), probs)[1]
    p_none = passed("none", none_rows)
    p = {k: passed(by_key[k], rows[k]) for k in [a.key] + refs}

    schemes = {k: scheme_of(next(iter(rows[k].values()))["wm"], str(d / "runs" / f"{by_key[k]}.jsonl"))
               for k in [a.key] + refs}
    tasks = sorted(set(p_none) & set.intersection(*(set(v) for v in p.values())))
    maj = (len(refs) + 1) // 2 if len(refs) > 1 else 1

    broken_by_key = [t for t in tasks if p_none[t] and not p[a.key][t] and sum(p[k][t] for k in refs) >= maj]
    broken_by_ref = {k: [t for t in tasks if p_none[t] and not p[k][t] and p[a.key][t]] for k in refs}
    broken_all = [t for t in tasks if p_none[t] and not p[a.key][t] and sum(p[k][t] for k in refs) < maj]

    # first-divergence substitutions for the suspect key, across all problems
    green_k, sw_k = schemes[a.key]
    rules_k = CanonRules.from_dict(next(iter(rows[a.key].values()))["wm"].get("canon_rules"))
    pattern_tasks = defaultdict(list)          # (ctx, from, to) -> tasks where the key's code diverged this way
    pattern_scored = {}                         # is the watermarked lexeme a carrier?
    divergence = {}
    for t in tasks:
        res = first_divergence(none_rows[t]["code"], rows[a.key][t]["code"], rules_k)
        if res is None:
            continue
        i, ref_lex, wm_lex = res
        tb = texts(wm_lex)
        ctx = green_k.context(tb, i)
        frm = ref_lex[i].text if i < len(ref_lex) else "<end>"
        to = wm_lex[i].text if i < len(wm_lex) else "<end>"
        scored = i < len(wm_lex) and wm_lex[i].scored
        pattern_tasks[(ctx, frm, to)].append(t)
        pattern_scored[(ctx, frm, to)] = scored
        divergence[t] = (i, ref_lex, wm_lex, ctx, frm, to, scored)

    def colour(k, ctx, lexeme, scored=True):
        green, sw = schemes[k]
        if lexeme.startswith("<") or not scored:
            return "n/a"
        on = sw.on(ctx)
        return ("G" if green.is_green(ctx, lexeme) else "r") + ("" if on else "(off)")

    ranked = sorted(pattern_tasks.items(),
                    key=lambda kv: (-sum(t in broken_by_key for t in kv[1]), -len(kv[1])))
    lines = [f"# Key diagnostic: {a.config}, key {a.key} (sweep {d.name})", "",
             f"Pass@1: unwatermarked {sum(p_none[t] for t in tasks) / len(tasks):.3f}; "
             + "; ".join(f"{k} {sum(p[k][t] for t in tasks) / len(tasks):.3f}" for k in [a.key] + refs), "",
             f"- Broken **only by {a.key}** (pass unwatermarked and with ≥{maj}/{len(refs)} other keys, fail with {a.key}): "
             f"**{len(broken_by_key)}** problems",
             f"- Broken by {a.key} and most other keys too (general watermark damage): {len(broken_all)}",
             "- For contrast, broken by each other key but not by " + a.key + ": "
             + ", ".join(f"{k} {len(v)}" for k, v in broken_by_ref.items()), "",
             "## First-divergence substitutions, ranked by how many key-specific breaks they precede", "",
             f"Colour of the watermark's choice under each key: G = green, r = red, (off) = switch off there, "
             f"n/a = not a carrier (never coloured). "
             f"If the substitution is green under {a.key} but red under the others, the key's colouring caused it.", "",
             f"| # breaks ({a.key} only) | # problems diverging this way | context (k={green_k.k}) | model wrote → watermarked wrote | "
             + " | ".join(f"{k}" for k in [a.key] + refs) + " |",
             "|---|---|---|---|" + "---|" * (1 + len(refs))]
    for (ctx, frm, to), ts in ranked[:a.top]:
        nb = sum(t in broken_by_key for t in ts)
        cols = " | ".join(colour(k, ctx, to, pattern_scored[(ctx, frm, to)]) for k in [a.key] + refs)
        lines.append(f"| {nb} | {len(ts)} | `{' '.join(ctx)}` | `{frm}` → `{to}` | {cols} |")

    # aggregate over lexeme substitution regardless of context
    sub = Counter()
    for t in broken_by_key:
        if t in divergence:
            _, _, _, _, frm, to, _ = divergence[t]
            sub[(frm, to)] += 1
    if sub:
        lines += ["", f"## Substitutions behind the {len(broken_by_key)} key-specific breaks (any context)", "",
                  "| count | model wrote → watermarked wrote |", "|---|---|"]
        lines += [f"| {n} | `{f}` → `{t}` |" for (f, t), n in sub.most_common(a.top)]
    unscored = sum(1 for t in broken_by_key if t in divergence and not divergence[t][6])
    if broken_by_key:
        lines += ["", f"{unscored}/{len(broken_by_key)} of these breaks first diverge at a non-carrier lexeme "
                  "(a name, number or string), i.e. after a token-level difference the canonical form does not show "
                  "directly; these are the hardest to attribute."]

    # worked examples
    lines += ["", "## Examples", ""]
    for t in broken_by_key[:a.examples]:
        if t not in divergence:
            continue
        i, ref_lex, wm_lex, ctx, frm, to, _ = divergence[t]
        ok, err = run_tests(rows[a.key][t]["code"], probs[t])
        ref_line = line_at(none_rows[t]["code"], ref_lex[i].start) if i < len(ref_lex) else "<end>"
        wm_line = line_at(rows[a.key][t]["code"], wm_lex[i].start) if i < len(wm_lex) else "<end>"
        lines += [f"**{t}**: `{frm}` → `{to}` after `{' '.join(ctx)}`  (test error: {err or 'none'})", "",
                  "```", f"unwatermarked: {ref_line.strip()}", f"{a.key:>13s}: {wm_line.strip()}", "```", ""]

    out = Path(a.out) if a.out else d / f"diagnose_{method}_d{delta:g}_{a.key}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
