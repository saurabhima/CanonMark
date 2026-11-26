"""Code edits used to test watermark robustness (from the project brief).

All attacks preserve behaviour. Each returns the edited code, or None if the
attack could not be applied (e.g. the input does not parse).
"""
from __future__ import annotations

import ast
import builtins
import io
import random
import tokenize

_BUILTINS = set(dir(builtins))


def format_black(code: str, line_length: int = 88) -> str | None:
    try:
        import black
        return black.format_str(code, mode=black.Mode(line_length=line_length))
    except Exception:
        return None


def strip_comments(code: str) -> str | None:
    """Remove # comments and docstrings."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                    and isinstance(first.value.value, str)):
                if len(node.body) == 1:          # keep the block non-empty
                    continue
                doc_lines.update(range(first.lineno, first.end_lineno + 1))
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
        code2 = tokenize.untokenize(toks)
    except (tokenize.TokenError, SyntaxError):
        return None
    out = "".join(ln for i, ln in enumerate(code2.splitlines(keepends=True), 1) if i not in doc_lines)
    try:
        ast.parse(out)
    except SyntaxError:
        return None
    return out


_NAME_POOL = ["alpha", "beta", "tmp", "val", "item", "acc", "res", "data", "q", "z", "foo", "bar",
              "baz", "cnt", "idx", "elem", "node", "buf", "w", "x_"]


def rename_locals(code: str, seed: int = 0, fraction: float = 1.0) -> str | None:
    """Consistently rename locally bound names (params, assignments, loop vars,
    nested functions). Top-level function names are kept so tests still call them."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in top:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                top.add((a.asname or a.name).split(".")[0])
    bound -= _BUILTINS | top
    rng = random.Random(seed)
    names = sorted(bound)
    rng.shuffle(names)
    names = names[: max(0, round(fraction * len(names)))]
    pool = _NAME_POOL[:]
    rng.shuffle(pool)
    mapping = {n: f"{pool[i % len(pool)]}{i}" for i, n in enumerate(names)}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    new, prev = [], None
    for t in toks:
        if t.type == tokenize.NAME and t.string in mapping and prev != ".":
            t = t._replace(string=mapping[t.string])
        new.append(t)
        if t.type not in (tokenize.NL, tokenize.COMMENT, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev = t.string
    out = tokenize.untokenize(new)
    try:
        ast.parse(out)
    except SyntaxError:
        return None
    return out


def all_three(code: str, seed: int = 0) -> str | None:
    c = strip_comments(code) or code
    c = rename_locals(c, seed) or c
    return format_black(c, 40)


ATTACKS = {
    "black": format_black,
    "strip_comments": strip_comments,
    "rename": rename_locals,
    "all_three": all_three,
}
