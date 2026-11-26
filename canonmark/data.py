"""Benchmark loaders and prompt builders (HumanEval, MBPP, HumanEvalPack).

Every problem is normalised to a dict:
  task_id, prompt (what the model is asked), reference (human solution code),
  tests (program text that raises on failure), entry_point, imports (prelude
  lines prepended before execution), dataset
"""
from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.environ.get("CANONMARK_DATA", Path(__file__).resolve().parent.parent / "data"))
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
HEP_URL = "https://huggingface.co/datasets/bigcode/humanevalpack/resolve/main/{lang}/test-00000-of-00001.parquet"

INSTRUCTION = ("Complete the following Python function. Return the complete function "
               "(including the signature) in a single ```python code block.")


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _prelude(prompt: str) -> str:
    return "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from "))) + "\n"


def load_humaneval() -> list[dict]:
    import human_eval
    path = Path(human_eval.__file__).parent / "data" / "HumanEval.jsonl.gz"
    out = []
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        out.append({
            "task_id": r["task_id"], "dataset": "humaneval",
            "prompt": f"{INSTRUCTION}\n\n```python\n{r['prompt']}```",
            "reference": r["prompt"] + r["canonical_solution"],
            "tests": r["test"] + f"\ncheck({r['entry_point']})\n",
            "entry_point": r["entry_point"], "imports": _prelude(r["prompt"]),
        })
    return out


def load_mbpp(split: str = "test") -> list[dict]:
    """split: 'train' (601-974, 374 problems), 'test' (11-510, 500), 'validation' (511-600)."""
    rng = {"train": (601, 974), "test": (11, 510), "validation": (511, 600), "prompt": (1, 10)}[split]
    path = _download(MBPP_URL, DATA_DIR / "mbpp.jsonl")
    out = []
    for line in open(path):
        r = json.loads(line)
        if not rng[0] <= r["task_id"] <= rng[1]:
            continue
        code = r["code"].replace("\r\n", "\n").replace("\t", "    ")
        tests = "\n".join([r.get("test_setup_code", "") or ""] + r["test_list"]) + "\n"
        out.append({
            "task_id": f"MBPP/{r['task_id']}", "dataset": f"mbpp-{split}",
            "prompt": (f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}\n"
                       "Return the solution in a single ```python code block."),
            "reference": code, "tests": tests, "entry_point": None, "imports": "",
        })
    return out


def load_humanevalpack(lang: str) -> list[dict]:
    """Java / JavaScript splits (execution needs a JDK / Node; detection does not)."""
    import pyarrow.parquet as pq
    path = _download(HEP_URL.format(lang=lang), DATA_DIR / f"humanevalpack_{lang}.parquet")
    out = []
    for r in pq.read_table(path).to_pylist():
        out.append({
            "task_id": r["task_id"], "dataset": f"humanevalpack-{lang}",
            "prompt": f"Complete the following function in a single code block.\n\n```\n{r['prompt']}```",
            "reference": r["prompt"] + r["canonical_solution"], "tests": r["test"],
            "entry_point": r["entry_point"], "imports": "",
        })
    return out


def load(name: str) -> list[dict]:
    if name == "humaneval":
        return load_humaneval()
    if name.startswith("mbpp"):
        return load_mbpp(name.split("-", 1)[1] if "-" in name else "test")
    if name.startswith("humanevalpack-"):
        return load_humanevalpack(name.split("-", 1)[1])
    raise ValueError(f"unknown dataset {name}")
