"""Generate solutions for a benchmark, with or without a watermark.

Example:
  python scripts/generate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset humaneval \
      --method canonmark --key my-secret --out outputs/he_canon.jsonl
  python scripts/generate.py ... --method none --out outputs/he_plain.jsonl
  python scripts/generate.py ... --method kgw  --out outputs/he_kgw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: E402

from canonmark.canon import extract_code  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.generate import (CanonLogitsProcessor, HybridLogitsProcessor, KGWLogitsProcessor,  # noqa: E402
                                KGWScheme)
from canonmark.keyed import KeyedGreen  # noqa: E402
from canonmark.switch import load_switch  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="humaneval", help="humaneval | mbpp-test | mbpp-train | humanevalpack-java ...")
    ap.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--method", choices=["none", "canonmark", "kgw", "hybrid"], default="canonmark")
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
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": torch.bfloat16 if device == "cuda" else torch.float32,
             "bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device).eval()

    probs = load(a.dataset)[a.offset:]
    if a.limit:
        probs = probs[:a.limit]

    if a.method == "canonmark":
        proc = CanonLogitsProcessor(tok, KeyedGreen(a.key, a.gamma, a.context_width), load_switch(a.switch),
                                    delta=a.delta, top_k=a.top_k, carrier_gate=a.carrier_gate)
    elif a.method == "kgw":
        proc = KGWLogitsProcessor(tok, KGWScheme(a.key, len(tok), a.gamma), delta=a.delta)
    elif a.method == "hybrid":
        proc = HybridLogitsProcessor(tok, KeyedGreen(a.key, a.gamma, a.context_width),
                                     KGWScheme(a.key + "/surface", len(tok), a.gamma), load_switch(a.switch),
                                     delta=a.delta, delta_surface=a.delta_surface, top_k=a.top_k,
                                     carrier_gate=a.carrier_gate)
    else:
        proc = None
    wm = {"method": a.method, "key": a.key, "gamma": a.gamma, "delta": a.delta, "delta_surface": a.delta_surface,
          "context_width": a.context_width, "switch": a.switch, "carrier_gate": a.carrier_gate, "model": a.model}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gen_kwargs = dict(max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
    if a.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=a.temperature, top_p=a.top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with open(out, "w") as f:
        for i in range(0, len(probs), a.batch_size):
            batch = probs[i:i + a.batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                             add_generation_prompt=True) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            plen = enc["input_ids"].shape[1]
            kwargs = dict(gen_kwargs)
            if proc is not None:
                proc.set_prompt_len(plen)
                kwargs["logits_processor"] = LogitsProcessorList([proc])
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, **kwargs)
            dt = time.time() - t0
            for p, row in zip(batch, gen[:, plen:]):
                resp = tok.decode(row, skip_special_tokens=True)
                rec = {"task_id": p["task_id"], "dataset": p["dataset"], "response": resp,
                       "code": extract_code(resp), "n_tokens": int((row != tok.pad_token_id).sum()),
                       "seconds": dt / len(batch), "wm": wm}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            done = min(i + a.batch_size, len(probs))
            extra = f" | biased steps {proc.stats['biased_steps']}/{proc.stats['code_steps']}" if proc else ""
            print(f"[{done}/{len(probs)}] {batch[-1]['task_id']} {dt:.1f}s{extra}", flush=True)


if __name__ == "__main__":
    main()
