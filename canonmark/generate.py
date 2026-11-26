"""Generation-time watermarking as Hugging Face logits processors.

CanonLogitsProcessor (ours)
    At each step, for the top-k candidate tokens, work out which canonical
    lexeme the candidate would produce. If that lexeme is scored, the switch
    is on for its canonical context, and it is green under the key, add
    delta to the candidate's logit. Only the current line is re-lexed per
    candidate (resumable canonicaliser), so the cost is small.

KGWLogitsProcessor (baseline)
    Standard Kirchenbauer et al. watermark on raw LLM tokens: the previous
    token seeds a keyed pseudo-random green half of the vocabulary.

Both watermark only inside the code region of the response (the first
```fenced block, or everything when fenced=False), so they are compared fairly.
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
import torch
from transformers import LogitsProcessor

from .canon import canonicalize, find_code_region
from .keyed import KeyedGreen
from .switch import AlwaysOn, SwitchPolicy


class _CodeRegionProcessor(LogitsProcessor):
    def __init__(self, tokenizer, fenced: bool = True):
        self.tok = tokenizer
        self.fenced = fenced
        self.prompt_len = 0
        self._tokstr: dict[int, str] = {}
        self.stats = {"steps": 0, "code_steps": 0, "biased_steps": 0}

    def set_prompt_len(self, n: int):
        self.prompt_len = n

    def tokstr(self, tid: int) -> str:
        s = self._tokstr.get(tid)
        if s is None:
            s = self.tok.decode([tid], skip_special_tokens=True)
            self._tokstr[tid] = s
        return s

    def open_code(self, gen_ids) -> str | None:
        """The code written so far if we are inside an open code region, else None."""
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        reg = find_code_region(text, self.fenced)
        if reg is None:
            return None
        s, e = reg
        if self.fenced and e < len(text):        # block already closed
            return None
        return text[s:]


class CanonLogitsProcessor(_CodeRegionProcessor):
    def __init__(self, tokenizer, green: KeyedGreen, switch: SwitchPolicy | None = None,
                 delta: float = 2.0, top_k: int = 40, fenced: bool = True, carrier_gate: bool = False):
        super().__init__(tokenizer, fenced)
        self.green = green
        self.switch = switch if switch is not None else AlwaysOn()
        self.delta = delta
        self.top_k = top_k
        self.carrier_gate = carrier_gate
        self.trace_token: int | None = None   # testing hook: record decisions for this token id
        self.trace: list[tuple] = []

    def _switch_for(self, row: int) -> SwitchPolicy:
        # a list gives each batch row its own policy (used by RL: one sampled policy per rollout)
        return self.switch[row] if isinstance(self.switch, (list, tuple)) else self.switch

    def _decisions(self, code: str, scores_row, switch) -> list[tuple[int, bool, bool]]:
        """For each top-k candidate: (token id, produces a new carrier?, green and switched on?)."""
        lex, cps = canonicalize(code)
        if cps:
            cp = cps[-1]
            base_n, tail_src, st, off = cp.n_lexemes, code[cp.offset:], cp.state, cp.offset
        else:
            base_n, tail_src, st, off = 0, code, None, 0
        base_texts = [l.text for l in lex[max(0, base_n - self.green.k):base_n]]
        old_tail = [(l.text, l.scored) for l in lex[base_n:]]
        cand = torch.topk(scores_row, min(self.top_k, scores_row.shape[-1])).indices.tolist()
        out = []
        for tid in cand:
            ts = self.tokstr(tid)
            if not ts or "`" in ts or "�" in ts:
                out.append((tid, False, False))
                continue
            new_tail, _ = canonicalize(tail_src + ts, st, off, base_n)
            i, m = 0, min(len(new_tail), len(old_tail))
            while i < m and (new_tail[i].text, new_tail[i].scored) == old_tail[i]:
                i += 1
            j = next((j for j in range(i, len(new_tail)) if new_tail[j].scored), None)
            if j is None:
                out.append((tid, False, False))
                continue
            seq = base_texts + [x.text for x in new_tail[:j]]      # only the first new carrier counts
            ctx = self.green.context(seq, len(seq))
            on = switch.on(ctx)
            is_green = on and self.green.is_green(ctx, new_tail[j].text)
            if tid == self.trace_token:
                self.trace.append((ctx, new_tail[j].text, on, is_green))
            out.append((tid, True, is_green))
        return out

    def _bias_row(self, gen_ids, scores_row, row: int = 0):
        code = self.open_code(gen_ids)
        if code is None:
            return False
        self.stats["code_steps"] += 1
        dec = self._decisions(code, scores_row, self._switch_for(row))
        # carrier gate: only choose *between* carriers. If the model's top choice is not a
        # carrier (closing a bracket, a newline, an identifier), pushing a carrier in would
        # distort the program, so leave this step alone.
        if self.carrier_gate and not (dec and dec[0][1]):
            return False
        biased = False
        for tid, carrier, green in dec:
            if carrier and green:
                scores_row[tid] += self.delta
                biased = True
        return biased

    def __call__(self, input_ids, scores):
        self.stats["steps"] += 1
        for b in range(input_ids.shape[0]):
            if self._bias_row(input_ids[b, self.prompt_len:], scores[b], b):
                self.stats["biased_steps"] += 1
        return scores


# ---------------------------------------------------------------------------
# KGW baseline (raw-token Green-Red, context width 1)
# ---------------------------------------------------------------------------
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return x ^ (x >> np.uint64(31))


class KGWScheme:
    """green(prev, tok) is a keyed PRF; equivalent in distribution to KGW's
    per-context random partition of the vocabulary."""

    def __init__(self, key: str | bytes, vocab_size: int, gamma: float = 0.5):
        self.key = key.encode() if isinstance(key, str) else key
        self.V = vocab_size
        self.gamma = gamma
        self._thr = np.uint64(int(gamma * (1 << 64)) - 1)
        self._ids = np.arange(vocab_size, dtype=np.uint64)

    def _seed(self, prev: int) -> np.uint64:
        d = hmac.new(self.key, int(prev).to_bytes(8, "big"), hashlib.sha256).digest()
        return np.uint64(int.from_bytes(d[:8], "big"))

    def green_mask(self, prev: int) -> np.ndarray:
        return _splitmix64(self._ids ^ self._seed(prev)) <= self._thr

    def is_green(self, prev: int, tok: int) -> bool:
        return bool(_splitmix64(np.array([tok], dtype=np.uint64) ^ self._seed(prev))[0] <= self._thr)


class KGWLogitsProcessor(_CodeRegionProcessor):
    def __init__(self, tokenizer, scheme: KGWScheme, delta: float = 2.0, fenced: bool = True):
        super().__init__(tokenizer, fenced)
        self.scheme = scheme
        self.delta = delta

    def __call__(self, input_ids, scores):
        self.stats["steps"] += 1
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:]
            if len(gen) == 0 or self.open_code(gen) is None:
                continue
            self.stats["code_steps"] += 1
            mask = torch.from_numpy(self.scheme.green_mask(int(gen[-1])))
            V = min(scores.shape[-1], mask.shape[0])
            scores[b, :V] += self.delta * mask[:V].to(scores.dtype)
            self.stats["biased_steps"] += 1
        return scores


def kgw_detect(code: str, tokenizer, scheme: KGWScheme, ignore_repeated: bool = True) -> dict:
    from .detect import p_value, z_score
    ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    seen, n, g = set(), 0, 0
    for prev, tok in zip(ids, ids[1:]):
        if ignore_repeated:
            if (prev, tok) in seen:
                continue
            seen.add((prev, tok))
        n += 1
        g += scheme.is_green(prev, tok)
    z = z_score(g, n, scheme.gamma)
    return {"z": z, "p_value": p_value(z), "n_scored": n, "n_green": g}


# ---------------------------------------------------------------------------
# Hybrid: canonical channel on carriers + surface (KGW) channel on everything else
# ---------------------------------------------------------------------------
class HybridLogitsProcessor(CanonLogitsProcessor):
    """Two channels on disjoint steps. The model's top candidate decides which
    channel owns the step: if it is a carrier, green carriers get the canonical
    (edit-invariant) bias; otherwise, non-carrier tokens (identifiers, literals,
    layout) get a raw-token KGW bias (strong on clean code, fragile to edits).
    Neither channel ever pushes one kind of token into the other's step."""

    def __init__(self, tokenizer, green: KeyedGreen, scheme: KGWScheme, switch: SwitchPolicy | None = None,
                 delta: float = 2.0, delta_surface: float | None = None, top_k: int = 40, fenced: bool = True,
                 carrier_gate: bool = False):
        super().__init__(tokenizer, green, switch, delta, top_k, fenced, carrier_gate)
        self.scheme = scheme
        self.delta_surface = delta if delta_surface is None else delta_surface

    def _bias_row(self, gen_ids, scores_row, row: int = 0):
        code = self.open_code(gen_ids)
        if code is None or len(gen_ids) == 0:
            return False
        self.stats["code_steps"] += 1
        dec = self._decisions(code, scores_row, self._switch_for(row))
        prev = int(gen_ids[-1])
        carrier_step = bool(dec and dec[0][1])
        biased = False
        for tid, carrier, green in dec:
            if carrier and green and (carrier_step or not self.carrier_gate):
                scores_row[tid] += self.delta
                biased = True
            elif not carrier and (not carrier_step or not self.carrier_gate) and self.scheme.is_green(prev, tid):
                scores_row[tid] += self.delta_surface
                biased = True
        return biased


def hybrid_detect(code: str, tokenizer, green: KeyedGreen, scheme: KGWScheme, switch: SwitchPolicy | None = None,
                  ignore_repeated: bool = True) -> dict:
    """z of each channel on disjoint positions, combined with Stouffer.
    Surface positions = LLM tokens that do not start a scored canonical lexeme."""
    from .canon import canonical_stream
    from .detect import detect, p_value, stouffer, z_score
    canon = detect(code, green, switch, ignore_repeated)
    starts = sorted(l.start for l in canonical_stream(code) if l.scored)
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    seen, n, g, si = set(), 0, 0, 0
    for t in range(1, len(ids)):
        a, b = offs[t]
        while si < len(starts) and starts[si] < a:
            si += 1
        if si < len(starts) and a <= starts[si] < b:     # this token owns a carrier
            continue
        pair = (ids[t - 1], ids[t])
        if ignore_repeated:
            if pair in seen:
                continue
            seen.add(pair)
        n += 1
        g += scheme.is_green(*pair)
    zs = z_score(g, n, scheme.gamma)
    z = stouffer([canon.z, zs])
    return {"z": z, "p_value": p_value(z), "z_canon": canon.z, "z_surface": zs,
            "n_canon": canon.n_scored, "n_surface": n}
