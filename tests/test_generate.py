"""Generation/detection agreement, by teacher-forcing real code through the
processor with the real LLM tokenizer (no model weights needed)."""
import os

import pytest
import torch

from canonmark.canon import canonical_stream, texts
from canonmark.data import load_humaneval
from canonmark.generate import CanonLogitsProcessor, KGWScheme, kgw_detect
from canonmark.keyed import KeyedGreen

MODEL = os.environ.get("CANONMARK_TEST_TOKENIZER", "Qwen/Qwen2.5-Coder-0.5B-Instruct")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


def test_generation_decisions_match_detection(tok):
    green = KeyedGreen("k", 0.5, 3)
    proc = CanonLogitsProcessor(tok, green, top_k=5)
    V = len(tok)
    total = matched = 0
    for p in load_humaneval()[:40]:
        code = p["reference"]
        resp = "Here:\n```python\n" + code + "```"
        ids = tok(resp, add_special_tokens=False)["input_ids"]
        proc.trace = []
        for t in range(1, len(ids)):
            scores = torch.zeros(V)
            scores[ids[t]] = 10.0
            proc.trace_token = ids[t]
            proc._bias_row(torch.tensor(ids[:t]), scores)
        lex = canonical_stream(code)
        tx = texts(lex)
        det = {(green.context(tx, j), lx.text) for j, lx in enumerate(lex) if lx.scored}
        for ctx, lexeme, on, g in proc.trace:
            total += 1
            matched += (ctx, lexeme) in det
    assert total > 300
    assert matched / total > 0.95, (matched, total)


def test_kgw_detector_null(tok):
    """Averaged over keys the null mean is ~0. (A single key shifts all programs
    together because common token pairs share one colour; see README.)"""
    progs = [p["reference"] for p in load_humaneval()]
    means = []
    for k in range(20):
        scheme = KGWScheme(f"k{k}", len(tok), 0.5)
        zs = [kgw_detect(c, tok, scheme)["z"] for c in progs]
        means.append(sum(zs) / len(zs))
    # per-key shifts of up to ~0.8 are real for raw-token KGW on 164 programs
    assert abs(sum(means) / len(means)) < 0.25, means
    assert max(abs(m) for m in means) < 1.2, means


def test_hybrid_detector_null(tok):
    from canonmark.generate import hybrid_detect
    progs = [p["reference"] for p in load_humaneval()]
    zs = []
    for k in range(20):  # surface channel has KGW-like per-key drift; average many keys
        green = KeyedGreen(f"h{k}", 0.5, 3)
        scheme = KGWScheme(f"h{k}/surface", len(tok), 0.5)
        zs += [hybrid_detect(c, tok, green, scheme)["z"] for c in progs]
    m = sum(zs) / len(zs)
    sd = (sum((z - m) ** 2 for z in zs) / len(zs)) ** 0.5
    assert abs(m) < 0.2 and 0.8 < sd < 1.25, (m, sd)


def test_decisions_report_carrier_lexemes(tok):
    """The candidate analysis used for the 'flippable' label names the carrier each token produces."""
    proc = CanonLogitsProcessor(tok, KeyedGreen("k", 0.5, 3), top_k=len(tok))
    code = "def f(xs):\n    "
    scores = torch.full((len(tok),), -100.0)
    want = {}
    for word in ("for", "while", "return", "x"):
        tid = tok(word, add_special_tokens=False)["input_ids"][0]
        scores[tid] = 0.0
        want[tid] = word
    proc.top_k = 4
    dec = {tid: (carrier, lexeme) for tid, carrier, _, lexeme in proc._decisions(code, scores, proc.switch)}
    for tid, word in want.items():
        carrier, lexeme = dec[tid]
        if word == "x":
            assert not carrier and lexeme is None          # a new local name is not a carrier
        else:
            assert carrier and lexeme == word


def test_generation_matches_detection_with_body_start_rule(tok):
    from canonmark.canon import CanonRules
    rules = CanonRules(skip_body_start=True)
    green = KeyedGreen("k", 0.5, 3)
    proc = CanonLogitsProcessor(tok, green, top_k=5, rules=rules)
    V = len(tok)
    total = matched = 0
    for p in load_humaneval()[:25]:
        code = p["reference"]
        ids = tok("```python\n" + code + "```", add_special_tokens=False)["input_ids"]
        proc.trace = []
        for t in range(1, len(ids)):
            scores = torch.zeros(V)
            scores[ids[t]] = 10.0
            proc.trace_token = ids[t]
            proc._bias_row(torch.tensor(ids[:t]), scores)
        lex = canonical_stream(code, rules)
        det = {(green.context(texts(lex), j), lx.text) for j, lx in enumerate(lex) if lx.scored}
        for ctx, lexeme, on, g in proc.trace:
            total += 1
            matched += (ctx, lexeme) in det
    # ~5% of decisions are on a partial name (e.g. `max` that becomes `max_depth`); the same rate as without
    # the rule on these problems (94.3%). Such cases only dilute the signal, never fake it.
    assert total > 150 and matched / total > 0.93, (matched, total)


def test_same_run_settings():
    from canonmark.pipeline import same_run_settings, wm_config
    new = wm_config("canonmark", "m", key="k", skip_body_start=True)
    old = {k: v for k, v in wm_config("canonmark", "m", key="k").items() if k not in ("canon_rules", "seeding")}
    assert set(same_run_settings(old, new)) == {"canon_rules", "seeding"}
    assert same_run_settings(new, dict(new)) == []
    assert same_run_settings(old, wm_config("canonmark", "m", key="k", seeding="global")) == []


def test_per_problem_seeding_makes_outputs_independent_of_order(tok, tmp_path):
    """With per-problem seeding a problem's sample does not depend on what was generated before it."""
    import json
    from transformers import AutoModelForCausalLM
    from canonmark.pipeline import generate_file, wm_config
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    probs = load_humaneval()
    target = probs[2]
    outs = {}
    for seeding in ("per_problem", "global"):
        for order, plist in (("alone", [target]), ("after", [probs[0], target])):
            f = tmp_path / f"{seeding}_{order}.jsonl"
            generate_file(model, tok, plist, wm_config("none", MODEL, seeding=seeding), f, temperature=1.0,
                          max_new_tokens=24, seed=7, log=lambda s: None)
            outs[(seeding, order)] = [json.loads(l) for l in open(f)][-1]["response"]
    assert outs[("per_problem", "alone")] == outs[("per_problem", "after")]
