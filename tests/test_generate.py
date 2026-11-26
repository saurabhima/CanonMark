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
