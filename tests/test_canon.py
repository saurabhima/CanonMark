"""Properties the whole design relies on."""
import random

import pytest

from canonmark.attacks import ATTACKS, format_black
from canonmark.canon import canonical_stream, canonicalize, extract_code, find_code_region, texts
from canonmark.data import load_humaneval, load_mbpp

HE = load_humaneval()
MBPP = load_mbpp("train")


def _sig(code):
    return [(l.text, l.scored) for l in canonical_stream(code)]


@pytest.mark.parametrize("attack", ["black", "strip_comments", "rename", "all_three"])
@pytest.mark.parametrize("corpus", ["humaneval", "mbpp-train"])
def test_exact_invariance(attack, corpus):
    """The canonical stream is identical after each attack."""
    progs = [p["reference"] for p in (HE if corpus == "humaneval" else MBPP)]
    tried = same = 0
    bad = []
    for code in progs:
        att = ATTACKS[attack](code)
        if att is None:
            continue
        tried += 1
        if _sig(att) == _sig(code):
            same += 1
        else:
            bad.append(code[:80])
    assert tried > 0.9 * len(progs)
    assert same / tried >= 0.99, bad[:3]


def test_black_narrow_lines():
    for p in HE:
        att = format_black(p["reference"], 30)
        if att is not None:
            assert _sig(att) == _sig(p["reference"]), p["task_id"]


def test_prefix_stability():
    """stream(prefix) agrees with stream(full) on all but the last lexeme."""
    rng = random.Random(0)
    bad = total = 0
    for p in HE[:60]:
        code = p["reference"]
        full = texts(canonical_stream(code))
        for cut in sorted(rng.sample(range(1, len(code)), min(40, len(code) - 1))):
            pre = texts(canonical_stream(code[:cut]))
            k = max(0, len(pre) - 1)
            total += 1
            bad += pre[:k] != full[:k]
    assert bad / total < 0.005


def test_resume_from_checkpoint_matches_full():
    """Re-lexing from the last checkpoint (what generation does) equals a full lex."""
    rng = random.Random(1)
    for p in HE[:80]:
        code = p["reference"]
        for cut in rng.sample(range(1, len(code)), min(20, len(code) - 1)):
            pre = code[:cut]
            full, cps = canonicalize(pre)
            if not cps:
                continue
            cp = cps[-1]
            tail, _ = canonicalize(pre[cp.offset:], cp.state, cp.offset, cp.n_lexemes)
            resumed = full[:cp.n_lexemes] + tail
            assert texts(resumed) == texts(full), (p["task_id"], pre[-60:])


def test_signature_is_unscored():
    lex = canonical_stream("def f(x: float, y: int = 0) -> bool:\n    return x > y\n")
    scored = [l.text for l in lex if l.scored]
    assert scored == ["return", ">"]


def test_yield_from_is_not_import():
    lex = canonical_stream("def g(xs):\n    yield from xs\n    return len(xs)\n")
    assert "len" in [l.text for l in lex if l.scored]


def test_local_renaming_and_builtins():
    a = texts(canonical_stream("def f(a):\n    b = len(a)\n    return sorted(a)[b]\n"))
    b = texts(canonical_stream("def f(q):\n  zz = len( q )\n  return sorted(q)[zz]  # c\n"))
    assert a == b
    assert "len" in a and "sorted" in a


def test_code_region():
    t = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    s, e = find_code_region(t)
    assert t[s:e] == "def f():\n    return 1\n"
    assert extract_code(t) == "def f():\n    return 1\n"
    assert find_code_region("Sure! ```pyth") is None
