# CanonMark: edit-invariant watermarking for LLM-generated code

Course project (ELL8299 / COL8383 / COL873, Trustworthy LLMs): *"Who wrote this Code? Pushing the limits of Code Watermarking"*.

CanonMark is a generation-time watermark for code LLMs. It defines the watermark over a **canonical lexeme stream** instead of the LLM's BPE tokens, so the detection statistic is **exactly unchanged** by:

- formatting (`black` / `ruff format`): whitespace, indentation, redundant parentheses, trailing commas, quote style
- removing comments and docstrings
- consistently renaming local variables

| Component | What it does | File |
|---|---|---|
| Canonicaliser | Python lexer → drop layout/comments/docstrings/`( ) , ;`, strings→`STR`, numbers normalised, locals→`v0,v1,…`; keywords, operators, builtins and `.attr` names kept as **carriers** | `canonmark/canon.py` |
| Keyed green function | `HMAC(key, k previous canonical lexemes ‖ lexeme) < γ` | `canonmark/keyed.py` |
| Switch | Decides which canonical contexts carry the watermark. `AlwaysOn`, or a small learned network trained on the canonical context only | `canonmark/switch.py` |
| Logits processor | For each top-k candidate token, finds the canonical lexeme it would produce and adds δ if that lexeme is green and the switch is on | `canonmark/generate.py` |
| Detector | Canonicalise → z-test over switched-on carriers. Needs no LLM and no prompt | `canonmark/detect.py` |
| Baseline | KGW / Green-Red over raw BPE tokens, same code-region handling | `canonmark/generate.py` |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -q tests
```

What the tests check:
- **Exact invariance.** The canonical stream is identical before and after `black` (including line length 30), comment stripping, renaming and all three combined, for 99%+ of HumanEval and MBPP-train programs (100% in our runs). `z` is unchanged.
- **Prefix stability and resumability.** The stream of a partial program agrees with the stream of the whole program, which is what keeps generation and detection consistent. Re-lexing from a checkpoint equals a full lex.
- **Calibrated null.** On human-written code with random keys, z ~ N(0, 1) and the false-positive rate at z > 4 is 0.
- **Generation/detection agreement.** Teacher-forcing real code through the processor with the real tokenizer: over 95% of the (context, lexeme) decisions made during generation are exactly the ones the detector later scores.

## Pipeline

```bash
M=Qwen/Qwen2.5-Coder-1.5B-Instruct

# 1. learned switch: supervised warm-up on LLM entropy over MBPP-train (minutes), optionally RL (GPU)
python scripts/train_switch.py    --model $M --dataset mbpp-train --context_width 3 --out outputs/switch_k3
python scripts/train_switch_rl.py --model $M --init outputs/switch_k3 --dataset mbpp-train --out outputs/switch_k3_rl

# 2. generate: unwatermarked reference, ours, hybrid, raw-token KGW baseline
python scripts/generate.py --model $M --dataset humaneval --method none --out outputs/he_none.jsonl
python scripts/generate.py --model $M --dataset humaneval --method canonmark --switch outputs/switch_k3 --delta 5 \
    --key SECRET --out outputs/he_canon.jsonl
python scripts/generate.py --model $M --dataset humaneval --method hybrid --switch outputs/switch_k3 --delta 5 \
    --delta_surface 2 --key SECRET --out outputs/he_hybrid.jsonl
python scripts/generate.py --model $M --dataset humaneval --method kgw --delta 2 --key SECRET --out outputs/he_kgw.jsonl

# 3. evaluate: Pass@1 (+ McNemar test vs unwatermarked), syntax errors, z / AUROC / TPR@5%FPR,
#    before and after each attack
python scripts/evaluate.py --dataset humaneval --runs outputs/he_canon.jsonl outputs/he_hybrid.jsonl \
    outputs/he_kgw.jsonl --negatives outputs/he_none.jsonl --out outputs/he_results.json
```

### Sweeps: trade-off curves with error bars

`scripts/sweep.py` runs a grid of watermark strength δ × secret key for each method. It loads the model once, shares one unwatermarked run, evaluates everything, and resumes if interrupted:

```bash
python scripts/sweep.py --model $M --dataset humaneval --switch outputs/switch_k3 \
    --keys key-a key-b key-c --canon_deltas 3 4 5 --hybrid_deltas 5 --delta_surface 2 \
    --kgw_deltas 1 2 3 --outdir outputs/sweep_he
```

It writes `sweep_runs.csv` (one row per run), `sweep_summary.csv` / `sweep_summary.md` (mean ± sd over keys per method and δ) and `sweep_results.json`. Plot `pass@1_mean` against `auroc_all_three_mean` to get each method's trade-off curve after edits. Use several keys: with one fixed key, raw-token KGW's null z can shift by up to ±0.8, which moves its AUROC.

**McNemar's test.** Pass@1 is compared with the unwatermarked run problem by problem. `lost` counts problems the watermark broke and `gained` counts problems it fixed. The p-value is an exact two-sided binomial test on those two counts. This is much more sensitive than comparing two independent Pass@1 percentages, because both runs answer the same problems.

## First results (CPU smoke test, indicative only)

Qwen2.5-Coder-**0.5B**-Instruct, the first **30** HumanEval problems, greedy decoding, γ = 0.5, k = 3. AUROC is measured against the same model's unwatermarked output. With n = 30, one problem is 3.3 points of Pass@1, so treat differences under ~10 points as noise. Rerun on the full benchmarks with the 1.5B models on a GPU.

| Configuration | Pass@1 | Syntax err. | AUROC clean | black | strip comments | rename | all three |
|---|---|---|---|---|---|---|---|
| No watermark | 0.800 | 3.3% | – | – | – | – | – |
| KGW, δ=2 (raw tokens) | 0.567 | 3.3% | 0.676 | 0.667 | 0.712 | 0.630 | 0.595 |
| CanonMark, δ=2, AlwaysOn | 0.767 | 3.3% | 0.589 | 0.589 | 0.589 | 0.589 | 0.589 |
| CanonMark, δ=2, switch | 0.733 | 3.3% | 0.574 | = | = | = | 0.574 |
| CanonMark, δ=5, AlwaysOn | 0.533 | 13.3% | 0.795 | = | = | = | 0.795 |
| **CanonMark, δ=5, switch** | **0.633** | **0%** | **0.788** | = | = | = | **0.788** |
| CanonMark, δ=5, carrier gate | 0.733 | 6.7% | 0.623 | = | = | = | 0.623 |
| Hybrid, δ=2, switch | 0.833 | 3.3% | 0.683 | 0.665 | 0.658 | 0.666 | 0.612 |
| Hybrid, δ=5 / surface 1.5, switch | 0.667 | 0% | 0.683 | 0.680 | 0.628 | 0.697 | 0.631 |
| Hybrid, δ=5 / surface 2, carrier gate | 0.667 | 3.3% | 0.700 | 0.693 | 0.678 | 0.672 | 0.596 |

(`=` means identical to clean: the canonical z does not change under these edits.)

What the smoke test shows:
1. **Invariance holds on real model output.** Every CanonMark row has the same AUROC before and after black, comment stripping, renaming and all three combined.
2. **At δ=2 the canonical channel is too weak** (mean z ≈ 0.25). A confident, near-greedy model rarely lets a +2 bias flip a carrier.
3. **δ=5 makes it strong (AUROC 0.79) but breaks code when applied everywhere** (13% syntax errors). Green builtins win over closing brackets, so expressions run on.
4. **The learned switch fixes most of that.** Pass@1 goes from 0.533 to 0.633, syntax errors from 13% to 0%, at the same AUROC. **This configuration beats KGW on both functionality and detection, clean and especially after edits** (0.788 vs 0.595).
5. **The carrier gate** (bias only when the model's top choice is a carrier) protects functionality but removes most of the signal. It is opt-in (`--carrier_gate`).
6. **Hybrid** gets more clean-code strength from its surface channel at a low δ, but that part decays under edits, as expected.

All scripts run on CPU (slowly) and on a single GPU. On CPU, `Qwen/Qwen2.5-Coder-0.5B-Instruct` generates about 10 tokens/s. The watermark adds about 5% per token.

## Design notes

- **Why keyed HMAC and not a learned green list (CodeTracer)?** For code written without the key, every scored position is green with probability exactly γ, so the z-test is valid however predictable the code is. CodeTracer's green list is "the plausible half" learned from code and has no key, so human code can score above chance, and anyone holding the network can detect or forge the watermark.
- **Why does RL only learn the switch?** The green lists are fixed by the key, so the only decision left is whether to watermark a context. That is a Bernoulli policy, trained with group-normalised rewards: tests pass, clipped z, and +1/−3 per green/red carrier. This avoids CodeTracer's Gumbel-Top-k and straight-through estimators.
- **Context width k.** In raw-token schemes, a larger k makes edits more damaging, because each edit breaks k+1 n-grams. Here formatting, comment stripping and renaming don't change the canonical stream at all, so k is a free choice for those edits. A larger k makes green lists harder to steal and reduces the per-key bias of the null: measured per-key mean z is ±0.15 at k=3, vs ±0.39 at k=1 and up to ±0.8 for raw-token KGW on HumanEval.
- **The signature is never scored.** Lexemes inside `def …:` / `class …:` headers come from the prompt, so the detector ignores them.

## Known limitations (to measure, not hide)

- **Capacity.** A typical HumanEval function has only about 14 scored carriers in its body. Single-function z is therefore modest. Combine z over several functions (`detect.stouffer`) or with a surface channel.
- **LLM paraphrasing** is not covered by construction. Neither is targeted rewriting of carriers (e.g. `a < b` → `b > a`); the canonicaliser can be extended to normalise such rewrites.
- **Stealability.** At small k the canonical contexts repeat a lot, so green lists are easier to learn from many samples. Measure this as a function of k.
- **Python only** for now. Java and JavaScript need their own lexer rules.

## Related code

The evaluation set-up follows CodeTracer (Guo et al., arXiv 2508.11925; https://github.com/TimeLovercc/CodeTracer). The KGW baseline follows Kirchenbauer et al. (2023). The test executor replaces CodeTracer's E2B sandbox with local subprocesses.
