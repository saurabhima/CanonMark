"""Keyed green/red decision on canonical lexemes.

green(ctx, lexeme) = HMAC-SHA256(key, ctx || lexeme) mapped to [0,1) < gamma

Each (context, lexeme) pair is independently green with probability gamma
under a random key. For code written without the key (human code, other
models), every scored position is therefore green with probability exactly
gamma, which is what the z-test assumes. This holds no matter how predictable
the code is, unlike CodeTracer's learned green lists. The alphabet is open:
any API name gets a well-defined colour without a fixed vocabulary.
"""
from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache

BOS = "<s>"
_SEP_CTX = "\x1f"
_SEP_LEX = "\x1e"


class KeyedGreen:
    def __init__(self, key: str | bytes, gamma: float = 0.5, context_width: int = 3):
        if isinstance(key, str):
            key = key.encode()
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        self.key = key
        self.gamma = gamma
        self.k = context_width
        self._threshold = int(gamma * (1 << 64))
        self._is_green = lru_cache(maxsize=1 << 20)(self._is_green_uncached)

    def context(self, lexeme_texts, j: int) -> tuple[str, ...]:
        """The k canonical lexemes before position j, left-padded with <s>."""
        lo = max(0, j - self.k)
        ctx = tuple(lexeme_texts[lo:j])
        if len(ctx) < self.k:
            ctx = (BOS,) * (self.k - len(ctx)) + ctx
        return ctx

    def _is_green_uncached(self, ctx: tuple[str, ...], lexeme: str) -> bool:
        msg = (_SEP_CTX.join(ctx) + _SEP_LEX + lexeme).encode()
        digest = hmac.new(self.key, msg, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big") < self._threshold

    def is_green(self, ctx: tuple[str, ...], lexeme: str) -> bool:
        return self._is_green(ctx, lexeme)
