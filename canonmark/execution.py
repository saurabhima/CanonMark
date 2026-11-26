"""Run generated code against benchmark tests in a subprocess.

This replaces CodeTracer's E2B cloud sandbox with local execution. Generated
code is untrusted: each program runs in its own temp directory, in isolated
mode (-I), with CPU-time and memory limits and a wall-clock timeout. That is
fine for a disposable container or cluster job, but it is not a security
sandbox. Run it inside a container if you evaluate models you don't trust.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def is_syntax_ok(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        os.setsid()
    return apply


def run_tests(code: str, problem: dict, timeout: float = 10.0, mem_gb: float = 4.0) -> tuple[bool, str]:
    """True if `code` passes all of `problem`'s tests."""
    if not is_syntax_ok(code):
        return False, "syntax_error"
    program = problem.get("imports", "") + code + "\n\n" + problem["tests"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits(int(timeout) + 1, int(mem_gb * 2**30)),
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if r.returncode == 0:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["error"])[-1]
    return False, err[:200]


def run_many(codes: list[str], problems: list[dict], workers: int = 4, timeout: float = 10.0) -> list[tuple[bool, str]]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda cp: run_tests(cp[0], cp[1], timeout), zip(codes, problems)))
