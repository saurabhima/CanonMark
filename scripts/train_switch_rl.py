"""RL training of the switch with GRPO-style group-normalised rewards.

Compared with CodeTracer, the action space is much smaller: green lists come
from the secret key, so the only learned decision is the switch (watermark
this canonical position or not). Each rollout samples a switch action per
canonical context it meets, and the policy gradient is taken on those
Bernoulli actions. The LLM stays frozen. No Gumbel-Top-k or straight-through
estimators are needed.

Rewards (as in CodeTracer):
  outcome  R = R_exec + lambda_wm * clip(z / 3, 0, 1)          (per rollout)
  process  +1 for a green carrier, -penalty for a red one       (per switched-on position)
Advantages: outcome rewards normalised within the group of G rollouts of the
same prompt; process rewards normalised over all switched-on positions in the
group. Each action gets A_outcome + beta * A_process.

Example (GPU):
  python scripts/train_switch_rl.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --init outputs/switch_k3 --dataset mbpp-train --steps 300 --group 8 \
      --out outputs/switch_k3_rl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: E402

from canonmark.canon import extract_code  # noqa: E402
from canonmark.data import load  # noqa: E402
from canonmark.detect import detect  # noqa: E402
from canonmark.execution import run_many  # noqa: E402
from canonmark.generate import CanonLogitsProcessor  # noqa: E402
from canonmark.keyed import KeyedGreen  # noqa: E402
from canonmark.switch import LearnedSwitch, SwitchPolicy, SwitchVocab, build_switch_net  # noqa: E402


class SampledSwitch(SwitchPolicy):
    """One rollout's policy: samples a Bernoulli action per context, once, and remembers it."""

    def __init__(self, net, vocab: SwitchVocab):
        self.net, self.vocab = net, vocab
        self.actions: dict[tuple, int] = {}

    def on(self, ctx):
        a = self.actions.get(ctx)
        if a is None:
            with torch.no_grad():
                p = torch.sigmoid(self.net(torch.tensor([self.vocab.encode(ctx)]))).item()
            a = int(random.random() < p)
            self.actions[ctx] = a
        return bool(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--init", help="switch from train_switch.py (recommended); else random init")
    ap.add_argument("--dataset", default="mbpp-train")
    ap.add_argument("--key", default="canonmark-train-key", help="training key; deploy with a different secret")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--context_width", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--group", type=int, default=8, help="rollouts per prompt (G)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--lambda_wm", type=float, default=1.0)
    ap.add_argument("--beta_process", type=float, default=0.5)
    ap.add_argument("--red_penalty", type=float, default=3.0)
    ap.add_argument("--entropy_bonus", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    if a.init:
        sw = LearnedSwitch.load(a.init)
        net, vocab = sw.net, sw.vocab
        if net.k != a.context_width:
            raise SystemExit(f"--init switch uses k={net.k}, not {a.context_width}")
    else:
        from canonmark.canon import canonical_stream, texts
        vocab = SwitchVocab.build([texts(canonical_stream(p["reference"])) for p in load(a.dataset)])
        net = build_switch_net(len(vocab.itos), a.context_width)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    green = KeyedGreen(a.key, a.gamma, a.context_width)
    problems = load(a.dataset)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / "train_log.jsonl", "w")

    for step in range(1, a.steps + 1):
        batch_probs = random.sample(problems, a.prompts_per_step)
        all_actions, all_adv = [], []
        stats = {"pass": [], "z": [], "on_rate": [], "reward": []}
        for p in batch_probs:
            G = a.group
            policies = [SampledSwitch(net, vocab) for _ in range(G)]
            proc = CanonLogitsProcessor(tok, green, policies, delta=a.delta)
            text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                           add_generation_prompt=True)
            enc = tok([text] * G, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            plen = enc["input_ids"].shape[1]
            proc.set_prompt_len(plen)
            with torch.no_grad():
                gen = llm.generate(**enc, do_sample=True, temperature=a.temperature, top_p=1.0,
                                   max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id,
                                   logits_processor=LogitsProcessorList([proc]))
            codes = [extract_code(tok.decode(r[plen:], skip_special_tokens=True)) for r in gen]
            passed = [ok for ok, _ in run_many(codes, [p] * G)]
            dets = [detect(c, green, pol, return_marks=True) for c, pol in zip(codes, policies)]
            R = torch.tensor([float(ok) + a.lambda_wm * min(max(d.z / 3, 0.0), 1.0)
                              for ok, d in zip(passed, dets)])
            A_out = (R - R.mean()) / (R.std() + 1e-4)
            # process reward per switched-on realised position, normalised over the group
            proc_r = [{m[4]: (1.0 if m[3] else -a.red_penalty) for m in d.marks} for d in dets]
            flat = [v for pr in proc_r for v in pr.values()]
            mu, sd = (statistics.mean(flat), statistics.pstdev(flat) + 1e-4) if len(flat) > 1 else (0.0, 1.0)
            for g in range(G):
                for ctx, act in policies[g].actions.items():
                    adv = A_out[g].item()
                    if ctx in proc_r[g]:
                        adv += a.beta_process * (proc_r[g][ctx] - mu) / sd
                    all_actions.append((ctx, act))
                    all_adv.append(adv)
                acts = list(policies[g].actions.values())
                stats["on_rate"].append(sum(acts) / max(1, len(acts)))
            stats["pass"] += passed
            stats["z"] += [d.z for d in dets]
            stats["reward"] += R.tolist()

        if not all_actions:
            continue
        x = torch.tensor([vocab.encode(c) for c, _ in all_actions], device=next(net.parameters()).device)
        act = torch.tensor([float(a_) for _, a_ in all_actions], device=x.device)
        adv = torch.tensor(all_adv, device=x.device)
        logits = net(x)
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(logits, act, reduction="none")
        p = torch.sigmoid(logits)
        ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        loss = -(adv * logp).mean() - a.entropy_bonus * ent.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        rec = {"step": step, "loss": loss.item(), "pass": statistics.mean(stats["pass"]),
               "z": statistics.mean(stats["z"]), "reward": statistics.mean(stats["reward"]),
               "switch_on_rate": statistics.mean(stats["on_rate"]), "n_actions": len(all_actions)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            net.eval()
            LearnedSwitch(net, vocab, 0.5).save(a.out)
            net.train()


if __name__ == "__main__":
    main()
