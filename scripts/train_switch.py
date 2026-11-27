"""Supervised warm-up of the learned switch (analogous to CodeTracer's SFT stage).

Label (default "flippable"): at each scored canonical position, could the keyed
bias change which carrier is written? True when the model's top choice is a
carrier and >= 2 different carriers are within delta of the top logit.
Alternative "entropy": next-token entropy >= 1.2, as in CodeTracer/SWEET. Input: only the k previous canonical lexemes, so the
detector can recompute the decision without the LLM.

Example:
  python scripts/train_switch.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --dataset mbpp-train --context_width 3 --out outputs/switch_k3
  # or train on the model's own generations:
  python scripts/train_switch.py ... --programs outputs/mbpp_train_plain.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from canonmark.canon import canonical_stream, texts  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.generate import CanonLogitsProcessor  # noqa: E402
from canonmark.keyed import KeyedGreen  # noqa: E402
from canonmark.switch import AlwaysOn, LearnedSwitch, SwitchVocab, build_switch_net  # noqa: E402


def collect(model, tok, problems, codes, k, device, max_len=2048, flip_delta=5.0, flip_top_k=20):
    """Return [(task_id, ctx, entropy, flippable)] for every scored lexeme.

    flippable: the model's top choice at this position is a carrier AND at least two
    different carriers are within flip_delta of the top logit, i.e. the keyed bias can
    change which carrier gets written. Uses the same candidate analysis as generation."""
    ctxer = KeyedGreen("unused", 0.5, k)          # only for its context() helper
    helper = CanonLogitsProcessor(tok, ctxer, AlwaysOn(), top_k=flip_top_k)
    out = []
    for p, code in zip(problems, codes):
        head = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                       add_generation_prompt=True) + "```python\n"
        full = head + code + "```"
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=max_len)
        ids = torch.tensor([enc["input_ids"]], device=device)
        offs = enc["offset_mapping"]
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        ent = -(logp.exp() * logp).sum(-1).cpu()          # ent[t] = uncertainty about token t+1
        lex = canonical_stream(code)
        tx = texts(lex)
        base = len(head)
        t = 0
        for j, lx in enumerate(lex):
            if not lx.scored:
                continue
            ch = base + lx.start
            while t < len(offs) and offs[t][1] <= ch:
                t += 1
            if t == 0 or t >= len(offs):
                continue
            row = logits[t - 1].cpu()
            prefix = full[base:offs[t][0]]                       # code written before this token
            dec = helper._decisions(prefix, row, helper.switch)
            top = float(row.max())
            near = {lexeme for tid, carrier, _, lexeme in dec if carrier and float(row[tid]) >= top - flip_delta}
            flippable = bool(dec and dec[0][1] and len(near) >= 2)
            out.append((p["task_id"], ctxer.context(tx, j), float(ent[t - 1]), flippable))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="mbpp-train")
    ap.add_argument("--programs", help="jsonl of generations (task_id, code); default: reference solutions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context_width", type=int, default=3)
    ap.add_argument("--label", choices=["flippable", "entropy"], default="flippable",
                    help="flippable: >=2 carriers within --flip_delta of the top logit (default); "
                         "entropy: next-token entropy >= --entropy_threshold (CodeTracer-style)")
    ap.add_argument("--flip_delta", type=float, default=5.0, help="use the delta you generate with")
    ap.add_argument("--flip_top_k", type=int, default=20)
    ap.add_argument("--entropy_threshold", type=float, default=1.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threshold", type=float, default=0.5, help="switch decision threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = {p["task_id"]: p for p in load(a.dataset)}
    if a.programs:
        rows = [json.loads(l) for l in open(a.programs)]
        pairs = [(probs[r["task_id"]], r["code"]) for r in rows if r["code"].strip()]
    else:
        pairs = [(p, p["reference"]) for p in probs.values()]
    if a.limit:
        pairs = pairs[:a.limit]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    raw = collect(model, tok, [p for p, _ in pairs], [c for _, c in pairs], a.context_width, device,
                  flip_delta=a.flip_delta, flip_top_k=a.flip_top_k)
    del model
    ent_rate = sum(e >= a.entropy_threshold for _, _, e, _ in raw) / len(raw)
    flip_rate = sum(f for _, _, _, f in raw) / len(raw)
    both = sum(f and e >= a.entropy_threshold for _, _, e, f in raw) / max(1, sum(f for *_, f in raw))
    print(f"{len(raw)} scored positions from {len(pairs)} programs; "
          f"uncertain (entropy>={a.entropy_threshold}): {ent_rate:.1%}; flippable (delta={a.flip_delta:g}): "
          f"{flip_rate:.1%}; flippable positions that are also high-entropy: {both:.1%}")
    print(f"training on label = {a.label}")
    # (task, ctx, label-as-float)
    data = [(t, c, float(f) if a.label == "flippable" else float(e >= a.entropy_threshold)) for t, c, e, f in raw]

    # split by problem so validation contexts come from unseen programs
    tids = sorted({t for t, _, _ in data})
    random.shuffle(tids)
    val_t = set(tids[: max(1, len(tids) // 5)])
    train = [d for d in data if d[0] not in val_t]
    val = [d for d in data if d[0] in val_t]
    vocab = SwitchVocab.build([ctx for _, ctx, _ in train], min_count=2)
    net = build_switch_net(len(vocab.itos), a.context_width)

    def tensors(rows):
        x = torch.tensor([vocab.encode(c) for _, c, _ in rows])
        y = torch.tensor([lab for _, _, lab in rows])
        return x, y

    xtr, ytr = tensors(train)
    xva, yva = tensors(val)
    pw = ((1 - ytr.mean()) / ytr.mean().clamp(min=1e-3)).detach()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    from sklearn.metrics import roc_auc_score
    best = (-1.0, None)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lossf(net(xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(xva))
        auc = roc_auc_score(yva.numpy(), pv.numpy()) if 0 < yva.sum() < len(yva) else float("nan")
        if auc > best[0]:
            best = (auc, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 0 or ep == a.epochs - 1:
            on = (pv > a.threshold).float()
            prec = (on * yva).sum() / on.sum().clamp(min=1)
            print(f"epoch {ep:3d}  val AUROC {auc:.3f}  switch-on rate {on.mean():.1%}  precision {prec:.2f}")
    net.load_state_dict(best[1])
    LearnedSwitch(net, vocab, a.threshold).save(a.out)
    print(f"saved switch (best val AUROC {best[0]:.3f}) to {a.out}")


if __name__ == "__main__":
    main()
