"""Generation pipeline shared by scripts/generate.py and scripts/sweep.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from .canon import extract_code
from .generate import CanonLogitsProcessor, HybridLogitsProcessor, KGWLogitsProcessor, KGWScheme
from .keyed import KeyedGreen
from .canon import CanonRules
from .switch import load_switch, switch_fingerprint

METHODS = ("none", "canonmark", "kgw", "hybrid")
SEEDING = ("per_problem", "global")

# settings that determine what a run file contains; a sweep must not reuse a run file whose
# settings differ from the ones requested
RUN_IDENTITY = ("method", "key", "gamma", "delta", "delta_surface", "context_width", "switch", "switch_sha256",
                "carrier_gate", "canon_rules", "seeding", "model")


def problem_seed(seed: int, task_id: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"{seed}:{task_id}".encode()).digest()[:4], "big")


def same_run_settings(recorded: dict, requested: dict) -> list[str]:
    """Names of identity settings that differ (older runs: missing canon_rules / seeding mean defaults
    {skip_body_start: False} / "global")."""
    defaults = {"canon_rules": CanonRules().as_dict(), "seeding": "global", "switch_sha256": None}
    diff = []
    for k in RUN_IDENTITY:
        a = recorded.get(k, defaults.get(k))
        b = requested.get(k, defaults.get(k))
        if k == "switch_sha256" and (a is None or b is None):
            continue                     # unverifiable for old runs; the fingerprint guard handles it
        if a != b:
            diff.append(k)
    return diff


def load_model(name: str, dtype: str = "auto"):
    """Return (model, tokenizer, device). Tokenizer pads on the left for batched generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    td = {"auto": torch.bfloat16 if device == "cuda" else torch.float32,
          "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=td).to(device).eval()
    return model, tok, device


def wm_config(method: str, model: str, key: str = "canonmark-demo-key", gamma: float = 0.5, delta: float = 5.0,
              delta_surface: float | None = None, context_width: int = 3, top_k: int = 40,
              switch: str = "always", carrier_gate: bool = False, skip_body_start: bool = False,
              seeding: str = "per_problem") -> dict:
    """Everything needed to rebuild the processor (generation) and the detector (evaluation).

    canon_rules: canonical scoring rules (CanonRules); detection must use the same ones.
    seeding: "per_problem" re-seeds the sampler from (seed, task_id) for every problem, so
    different runs draw the same random numbers on the same problem and differ only through
    the watermark; "global" seeds once per run (the behaviour of runs made before this option)."""
    if seeding not in SEEDING:
        raise ValueError(f"seeding must be one of {SEEDING}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    uses_switch = method in ("canonmark", "hybrid")
    return {"method": method, "key": key, "gamma": gamma, "delta": delta, "delta_surface": delta_surface,
            "context_width": context_width, "top_k": top_k, "switch": switch,
            "switch_sha256": switch_fingerprint(switch) if uses_switch else None,   # checked at evaluation
            "carrier_gate": carrier_gate, "canon_rules": CanonRules(skip_body_start).as_dict(),
            "seeding": seeding, "model": model}


def build_processor(wm: dict, tok):
    m = wm["method"]
    if m == "none":
        return None
    if m == "canonmark":
        return CanonLogitsProcessor(tok, KeyedGreen(wm["key"], wm["gamma"], wm["context_width"]),
                                    load_switch(wm["switch"]), delta=wm["delta"], top_k=wm["top_k"],
                                    carrier_gate=wm["carrier_gate"], rules=CanonRules.from_dict(wm.get("canon_rules")))
    if m == "kgw":
        return KGWLogitsProcessor(tok, KGWScheme(wm["key"], len(tok), wm["gamma"]), delta=wm["delta"])
    if m == "hybrid":
        return HybridLogitsProcessor(tok, KeyedGreen(wm["key"], wm["gamma"], wm["context_width"]),
                                     KGWScheme(wm["key"] + "/surface", len(tok), wm["gamma"]),
                                     load_switch(wm["switch"]), delta=wm["delta"],
                                     delta_surface=wm["delta_surface"], top_k=wm["top_k"],
                                     carrier_gate=wm["carrier_gate"], rules=CanonRules.from_dict(wm.get("canon_rules")))
    raise ValueError(m)


def generate_file(model, tok, problems: list[dict], wm: dict, out: str | Path, batch_size: int = 1,
                  temperature: float = 0.2, top_p: float = 0.95, max_new_tokens: int = 512, seed: int = 42,
                  log=print) -> Path:
    """Generate one solution per problem and write a jsonl run file."""
    per_problem = wm.get("seeding", "global") == "per_problem"
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    proc = build_processor(wm, tok)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    tmp = out.with_suffix(out.suffix + ".partial")
    with open(tmp, "w") as f:
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i + batch_size]
            if per_problem:                  # same random numbers for the same problem in every run
                torch.manual_seed(problem_seed(seed, batch[0]["task_id"]))
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
            done = min(i + batch_size, len(problems))
            extra = f" | biased steps {proc.stats['biased_steps']}/{proc.stats['code_steps']}" if proc else ""
            log(f"[{done}/{len(problems)}] {batch[-1]['task_id']} {dt:.1f}s{extra}")
    tmp.replace(out)          # only complete runs get the final name (sweeps can resume safely)
    return out
