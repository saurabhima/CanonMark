"""Watermark detection from code alone (no LLM, no prompt).

1. canonicalise the code
2. for every scored lexeme whose context switches the watermark on,
   check whether it is green under the key
3. one-proportion z-test against gamma
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .canon import CanonRules, canonical_stream, texts
from .keyed import KeyedGreen
from .switch import AlwaysOn, SwitchPolicy


@dataclass
class Detection:
    z: float
    p_value: float
    n_scored: int
    n_green: int
    watermarked: bool
    marks: list = field(default_factory=list)   # (lexeme, start, end, green, ctx) per counted position

    def as_dict(self):
        return {"z": self.z, "p_value": self.p_value, "n_scored": self.n_scored,
                "n_green": self.n_green, "watermarked": self.watermarked}


def z_score(n_green: int, n: int, gamma: float) -> float:
    if n == 0:
        return 0.0
    return (n_green - gamma * n) / math.sqrt(n * gamma * (1 - gamma))


def p_value(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def detect(code: str, green: KeyedGreen, switch: SwitchPolicy | None = None,
           ignore_repeated: bool = True, z_threshold: float = 4.0,
           return_marks: bool = False, weighted: bool = False, rules: CanonRules | None = None) -> Detection:
    """Score a piece of code. `code` should be just the code (see canon.extract_code).

    ignore_repeated: count each (context, lexeme) pair once. Repeats are perfectly
    correlated under the null hypothesis and would otherwise inflate z.
    weighted: weight each switched-on carrier by the switch's confidence w (its
    probability; 1 for AlwaysOn) and use z_w = sum w(g - gamma) / sqrt(gamma(1-gamma) sum w^2).
    The weights depend only on the canonical context, not on the key, so under the
    null hypothesis z_w is still exactly standardised; it gains power when
    confident positions really carry more signal.
    """
    switch = switch or AlwaysOn()
    lex = canonical_stream(code, rules)
    tx = texts(lex)
    seen = set()
    n = g = 0
    sw_num = sw_den = 0.0
    marks = []
    for j, lx in enumerate(lex):
        if not lx.scored:
            continue
        ctx = green.context(tx, j)
        if not switch.on(ctx):
            continue
        key = (ctx, lx.text)
        if ignore_repeated:
            if key in seen:
                continue
            seen.add(key)
        is_g = green.is_green(ctx, lx.text)
        n += 1
        g += is_g
        if weighted:
            w = switch.weight(ctx)
            sw_num += w * (is_g - green.gamma)
            sw_den += w * w
        if return_marks:
            marks.append((lx.text, lx.start, lx.end, is_g, ctx))
    if weighted:
        z = sw_num / math.sqrt(green.gamma * (1 - green.gamma) * sw_den) if sw_den > 0 else 0.0
    else:
        z = z_score(g, n, green.gamma)
    return Detection(z, p_value(z), n, g, z > z_threshold, marks)


def stouffer(zs, weights=None) -> float:
    """Combine independent z-scores, e.g. the canonical and a surface channel,
    or several functions from the same file / repository."""
    zs = list(zs)
    if not zs:
        return 0.0
    w = list(weights) if weights is not None else [1.0] * len(zs)
    return sum(wi * zi for wi, zi in zip(w, zs)) / math.sqrt(sum(wi * wi for wi in w))
