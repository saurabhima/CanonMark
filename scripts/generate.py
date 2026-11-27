"""Generate solutions for a benchmark, with or without a watermark.

Example:
  python scripts/generate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset humaneval \
      --method canonmark --switch outputs/switch_k3 --key my-secret --out outputs/he_canon.jsonl
  python scripts/generate.py ... --method none --out outputs/he_none.jsonl
  python scripts/generate.py ... --method kgw --delta 2 --out outputs/he_kgw.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonmark.data import load  # noqa: E402
from canonmark.pipeline import METHODS, generate_file, load_model, wm_config  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="humaneval", help="humaneval | mbpp-test | mbpp-train | humanevalpack-java ...")
    ap.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--method", choices=METHODS, default="canonmark")
    ap.add_argument("--key", default="canonmark-demo-key")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=5.0, help="5 with a learned switch worked best in our smoke runs")
    ap.add_argument("--delta_surface", type=float, default=None, help="hybrid: surface-channel delta (default = --delta)")
    ap.add_argument("--context_width", type=int, default=3, help="canonical context length k")
    ap.add_argument("--top_k", type=int, default=40, help="candidates examined per step")
    ap.add_argument("--switch", default="always", help="'always' or path to a trained switch")
    ap.add_argument("--carrier_gate", action="store_true", help="bias carriers only when the model's top choice is a carrier")
    ap.add_argument("--temperature", type=float, default=0.2, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", help="auto | bfloat16 | float32")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    model, tok, _ = load_model(a.model, a.dtype)
    probs = load(a.dataset)[a.offset:]
    if a.limit:
        probs = probs[:a.limit]
    wm = wm_config(a.method, a.model, a.key, a.gamma, a.delta, a.delta_surface, a.context_width, a.top_k,
                   a.switch, a.carrier_gate)
    generate_file(model, tok, probs, wm, a.out, a.batch_size, a.temperature, a.top_p, a.max_new_tokens, a.seed,
                  log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
