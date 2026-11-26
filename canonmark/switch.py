"""Switch policies: at which canonical positions is the watermark active?

The switch sees only the canonical context (the k previous lexemes), so the
detector can recompute it from the code alone, with no LLM and no prompt.

* AlwaysOn       - every scored position is a carrier (no training needed).
* SwitchNet      - a tiny learned network, first trained by supervised learning
                   to predict where the LLM is uncertain (scripts/train_switch.py),
                   and later by RL with execution + detection rewards.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class SwitchPolicy:
    def on(self, ctx: tuple[str, ...]) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class AlwaysOn(SwitchPolicy):
    def on(self, ctx):
        return True


# --------------------------------------------------------------------------
# Learned switch
# --------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"^v(\d+)$")
PAD, UNK = "<pad>", "<unk>"


def bucket(lexeme: str) -> str:
    """Coarsen open-ended lexemes so the switch vocabulary stays small."""
    m = _PLACEHOLDER.match(lexeme)
    if m:
        return lexeme if int(m.group(1)) < 16 else "v*"
    if lexeme.startswith("NUM:"):
        return lexeme if lexeme in ("NUM:0", "NUM:1", "NUM:2", "NUM:-1") else "NUM"
    if lexeme.startswith("MOD:"):
        return "MOD"
    return lexeme


class SwitchVocab:
    def __init__(self, tokens: list[str]):
        self.itos = [PAD, UNK] + [t for t in tokens if t not in (PAD, UNK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, ctx) -> list[int]:
        return [self.stoi.get(bucket(t), 1) for t in ctx]

    @classmethod
    def build(cls, lexeme_lists, min_count: int = 2) -> "SwitchVocab":
        from collections import Counter
        c = Counter(bucket(t) for lst in lexeme_lists for t in lst)
        return cls(["<s>"] + sorted(t for t, n in c.items() if n >= min_count))


def _torch():
    import torch  # imported lazily: detection with AlwaysOn needs no torch
    return torch


def build_switch_net(vocab_size: int, k: int, dim: int = 64, hidden: int = 256):
    torch = _torch()
    nn = torch.nn

    class SwitchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.pos = nn.Parameter(torch.zeros(k, dim))
            self.mlp = nn.Sequential(nn.Linear(k * dim, hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, ids):            # ids: (B, k) -> logits (B,)
            x = self.emb(ids) + self.pos
            return self.mlp(x.flatten(1)).squeeze(-1)

    return SwitchNet()


class LearnedSwitch(SwitchPolicy):
    """Wraps a trained SwitchNet; decisions are cached per context."""

    def __init__(self, net, vocab: SwitchVocab, threshold: float = 0.5):
        self.net = net.eval()
        self.vocab = vocab
        self.threshold = threshold
        self._cache: dict[tuple, bool] = {}

    def on(self, ctx):
        hit = self._cache.get(ctx)
        if hit is not None:
            return hit
        torch = _torch()
        with torch.no_grad():
            ids = torch.tensor([self.vocab.encode(ctx)])
            p = torch.sigmoid(self.net(ids)).item()
        res = p > self.threshold
        self._cache[ctx] = res
        return res

    def save(self, path: str | Path):
        torch = _torch()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "switch.pt")
        (path / "switch_config.json").write_text(json.dumps(
            {"vocab": self.vocab.itos, "k": self.net.k, "threshold": self.threshold}, indent=1))

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "LearnedSwitch":
        torch = _torch()
        path = Path(path)
        cfg = json.loads((path / "switch_config.json").read_text())
        vocab = SwitchVocab(cfg["vocab"][2:])
        net = build_switch_net(len(vocab.itos), cfg["k"])
        net.load_state_dict(torch.load(path / "switch.pt", map_location="cpu"))
        return cls(net, vocab, cfg["threshold"] if threshold is None else threshold)


def load_switch(spec: str | None) -> SwitchPolicy:
    """'always' (default) or a path to a saved LearnedSwitch directory."""
    if spec in (None, "", "always"):
        return AlwaysOn()
    return LearnedSwitch.load(spec)
