"""Canonical lexeme stream: the representation the watermark is defined on.

The stream is designed to be *exactly* invariant to
  * formatting (black / ruff format): whitespace, newlines, indentation,
    redundant parentheses, trailing commas, ';' statement separators, quote style
  * comment and docstring removal
  * consistent renaming of locally bound identifiers

How:
  * lex with Python's own tokenizer (independent of the LLM's BPE tokenizer)
  * drop comments, NL/NEWLINE/INDENT/DEDENT, docstrings, and ( ) , ; \\
  * strings (incl. f-strings) -> STR, numbers -> normalised value NUM:<v>
  * local identifiers -> placeholders v0, v1, ... in order of first occurrence
  * keywords, operators, builtins, attribute names (after '.') and imported
    module names keep their identity

Watermark carriers ("scored" lexemes): keywords, operators, builtins and
attribute/API names. Placeholders, STR, NUM and module names are context only.
Lexemes inside a `def`/`class` header (the signature, usually given by the
prompt) are never scored.

The canonicaliser is resumable: `canonicalize` returns checkpoints at logical
line ends, and lexing can restart from one with its saved state. Generation
uses this to re-lex only the current line for each candidate token, and the
prefix-stability property (stream(prefix) is a prefix of stream(full), except
for the last, still-growing lexeme) keeps generation and detection consistent.
"""
from __future__ import annotations

import builtins
import copy
import io
import keyword
import tokenize
from dataclasses import dataclass, field

BUILTINS = frozenset(dir(builtins))

# keywords that never carry the watermark (structural / given by prompt)
UNSCORED_KEYWORDS = frozenset({"def", "class", "import", "from", "as", "pass", "global", "nonlocal"})
# operators kept in the stream as context but never scored
UNSCORED_OPS = frozenset({":", ".", "[", "]", "{", "}", "=", "->", "@", "...", ":="})
# operators dropped entirely (black adds/removes these freely)
DROPPED_OPS = frozenset({"(", ")", ",", ";", "\\"})
OPEN_BRACKETS = frozenset({"(", "[", "{"})
CLOSE_BRACKETS = frozenset({")", "]", "}"})

_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)
_FSTRING_TYPES = frozenset(t for t in (_FSTRING_START, _FSTRING_MIDDLE, _FSTRING_END) if t is not None)
_LAYOUT_TYPES = frozenset({tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                           tokenize.ENCODING, tokenize.ENDMARKER})


@dataclass(frozen=True)
class Lexeme:
    text: str      # canonical form, e.g. "for", "==", "len", ".append", "v3", "STR", "NUM:1"
    scored: bool   # True if this lexeme is a watermark carrier
    start: int     # character span of the source token(s) in the input code
    end: int


@dataclass(frozen=True)
class CanonRules:
    """Optional scoring rules. They change which lexemes are carriers, so a run must be
    detected with the rules it was generated with (they are recorded in its settings).

    skip_body_start: the first lexeme of every function body (after the docstring) is
        never a carrier. The bias otherwise tends to push a green `return` / `if` where
        the model wanted to start with an assignment, producing hasty one-liners.
    """
    skip_body_start: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "CanonRules":
        return cls(**(d or {}))

    def as_dict(self) -> dict:
        return {"skip_body_start": self.skip_body_start}


@dataclass
class CanonState:
    """Everything needed to resume lexing at the start of a logical line."""
    names: dict = field(default_factory=dict)       # local identifier -> placeholder
    imported: set = field(default_factory=set)      # names bound by import statements
    after_block_open: bool = True                   # module start / right after a def or class header
    in_header: bool = False                         # inside `def ...:` / `class ...:`
    in_import: bool = False                         # inside an import statement
    skip_body_start: bool = False                   # CanonRules.skip_body_start
    pending_body: bool = False                      # next lexeme starts a function body
    header_kind: str | None = None                  # "def" / "class" of the current header

    def copy(self) -> "CanonState":
        return CanonState(dict(self.names), set(self.imported), self.after_block_open,
                          self.in_header, self.in_import, self.skip_body_start, self.pending_body,
                          self.header_kind)


@dataclass(frozen=True)
class Checkpoint:
    offset: int        # character offset where a fresh lex can start (start of a logical line)
    n_lexemes: int     # number of lexemes emitted before `offset`
    state: CanonState


def _norm_number(s: str) -> str:
    try:
        import ast
        return "NUM:" + repr(ast.literal_eval(s.replace("_", "")))
    except Exception:
        return "NUM:" + s.lower()


def _line_starts(code: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def canonicalize(code: str, state: CanonState | None = None, offset: int = 0,
                 n_before: int = 0, rules: CanonRules | None = None) -> tuple[list[Lexeme], list[Checkpoint]]:
    """Canonicalise `code` (which may be an incomplete prefix).

    Returns (lexemes, checkpoints). Spans are shifted by `offset`; checkpoint
    lexeme counts are shifted by `n_before` so that resumed runs compose.
    Lexing stops quietly at the first error (incomplete prefix).
    """
    st = state.copy() if state is not None else CanonState(skip_body_start=bool(rules and rules.skip_body_start))
    out: list[Lexeme] = []
    cps: list[Checkpoint] = []
    starts = _line_starts(code)

    def emit(text: str, scored: bool, start: int, end: int):
        if st.pending_body:              # first lexeme of a function body (skip_body_start)
            scored = False
            st.pending_body = False
        out.append(Lexeme(text, scored, start, end))

    def pos(rc):
        r, c = rc
        return offset + starts[r - 1] + c if r - 1 < len(starts) else offset + len(code)

    toks = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            toks.append(tok)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    depth = 0
    at_stmt_start = True
    prev_sig: str | None = None
    fstring_depth = 0
    fstring_start = 0
    n = len(toks)
    for i, tok in enumerate(toks):
        tt, s = tok.type, tok.string

        # ---- f-strings: collapse into one STR lexeme -------------------------
        if tt in _FSTRING_TYPES or fstring_depth:
            if tt == _FSTRING_START:
                if fstring_depth == 0:
                    fstring_start = pos(tok.start)
                fstring_depth += 1
            elif tt == _FSTRING_END:
                fstring_depth -= 1
                if fstring_depth == 0:
                    if prev_sig != "STR":
                        emit("STR", False, fstring_start, pos(tok.end))
                    prev_sig = "STR"
                    at_stmt_start = st.after_block_open = False
            continue

        # ---- logical line end -> checkpoint ---------------------------------
        if tt == tokenize.NEWLINE:
            at_stmt_start = True
            st.in_import = False
            if s != "" and depth == 0:    # a real newline (not the implicit one at EOF)
                cps.append(Checkpoint(pos(tok.end), n_before + len(out), st.copy()))
            continue
        if tt in _LAYOUT_TYPES:
            continue

        # ---- strings / docstrings -------------------------------------------
        if tt == tokenize.STRING:
            nxt = next((t for t in toks[i + 1:] if t.type not in (tokenize.COMMENT, tokenize.NL)), None)
            is_doc = at_stmt_start and st.after_block_open and (
                nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.ENDMARKER))
            at_stmt_start = False
            st.after_block_open = False
            if is_doc:
                continue
            if prev_sig != "STR":     # implicit concatenation 'a' 'b' -> one STR
                emit("STR", False, pos(tok.start), pos(tok.end))
            prev_sig = "STR"
            continue

        was_stmt_start = at_stmt_start
        at_stmt_start = False
        if not st.in_header:
            st.after_block_open = False

        # ---- operators ---------------------------------------------------------
        if tt == tokenize.OP:
            if s in OPEN_BRACKETS:
                depth += 1
            elif s in CLOSE_BRACKETS:
                depth = max(0, depth - 1)
            if s in DROPPED_OPS:
                continue
            scored = s not in UNSCORED_OPS and not st.in_header and not st.in_import
            emit(s, scored, pos(tok.start), pos(tok.end))
            if s == ":" and st.in_header and depth == 0:
                st.in_header = False
                st.after_block_open = True   # the next statement may be a docstring
                if st.skip_body_start and st.header_kind == "def":
                    st.pending_body = True
            prev_sig = s
            continue

        # ---- numbers -------------------------------------------------------------
        if tt == tokenize.NUMBER:
            emit(_norm_number(s), False, pos(tok.start), pos(tok.end))
            prev_sig = "NUM"
            continue

        # ---- names -----------------------------------------------------------------
        if tt == tokenize.NAME:
            if keyword.iskeyword(s):
                scored = s not in UNSCORED_KEYWORDS and not st.in_header and not st.in_import
                emit(s, scored, pos(tok.start), pos(tok.end))
                if s in ("def", "class"):
                    st.in_header = True
                    st.header_kind = s
                elif s == "import" or (s == "from" and was_stmt_start):  # not `yield from` / `raise .. from`
                    st.in_import = True
                prev_sig = s
                continue
            if prev_sig == ".":                               # attribute / method name
                emit("." + s, not st.in_header, pos(tok.start), pos(tok.end))
            elif st.in_import:                                # import statement names
                st.imported.add(s)
                emit("MOD:" + s, False, pos(tok.start), pos(tok.end))
            elif s in st.imported:
                emit("MOD:" + s, False, pos(tok.start), pos(tok.end))
            elif s in BUILTINS and s not in st.names:
                emit(s, not st.in_header, pos(tok.start), pos(tok.end))
            else:
                if s not in st.names:
                    st.names[s] = f"v{len(st.names)}"
                emit(st.names[s], False, pos(tok.start), pos(tok.end))
            prev_sig = s
            continue

        # ---- anything else (ERRORTOKEN etc.) -----------------------------------------
        if s.strip():
            emit(s, False, pos(tok.start), pos(tok.end))
            prev_sig = s
    return out, cps


def canonical_stream(code: str, rules: CanonRules | None = None) -> list[Lexeme]:
    """Canonical lexemes of a (possibly partial) piece of Python code."""
    return canonicalize(code, rules=rules)[0]


def texts(lexemes) -> list[str]:
    return [lx.text for lx in lexemes]


# ---------------------------------------------------------------------------
# Locating code inside an LLM response
# ---------------------------------------------------------------------------
FENCE = "```"


def find_code_region(text: str, fenced: bool = True) -> tuple[int, int] | None:
    """Return (start, end) of the code region in `text`, or None if no code has started.

    fenced=True: code lives in the first ``` block (language tag line skipped);
    `end` is the closing fence or len(text) if the block is still open.
    fenced=False: the whole text is code.
    """
    if not fenced:
        return 0, len(text)
    i = text.find(FENCE)
    if i < 0:
        return None
    nl = text.find("\n", i + len(FENCE))
    if nl < 0:
        return None                      # still writing the ```python line
    start = nl + 1
    j = text.find(FENCE, start)
    return start, (j if j >= 0 else len(text))


def extract_code(text: str, fenced: bool = True) -> str:
    """Code from an LLM response (first fenced block; whole text if none)."""
    reg = find_code_region(text, fenced)
    if reg is None:
        return text if fenced and FENCE not in text else ""
    return text[reg[0]:reg[1]]
