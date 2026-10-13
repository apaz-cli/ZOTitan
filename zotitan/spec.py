"""The shared `(NAME, positional..., KEY=VALUE...)` component grammar.

Objective specs (`--objective`) and architecture specs (`--model.model-id`) both use
it; see the README for the full grammar and quoting rules.
"""
from typing import Any


_QUOTES = ("'", '"')


def split_top_level(s: str, delim: str) -> list[str]:
    """Split `s` on `delim`, ignoring delimiters inside quoted regions (either ' or ")
    or parentheses. A doubled quote inside a quote ('' or "") is a literal quote, not a
    terminator."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None      # the open quote char, or None when unquoted
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if quote is not None:
            if c == quote and i + 1 < len(s) and s[i + 1] == quote:
                buf.append(c); buf.append(c); i += 2; continue   # escaped quote, stay in-quote
            if c == quote:
                quote = None
        elif c in _QUOTES:
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == delim and depth == 0:
            out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c)
        i += 1
    if quote is not None:
        raise ValueError(f"unterminated {quote} quote in spec: {s!r}")
    out.append("".join(buf))
    return out


def parse_value(tok: str) -> Any:
    """Parse a single field value. Quoted (' or ") -> string (a doubled quote -> a literal
    quote); otherwise it must be a bare literal (int / float / True / False / None)."""
    tok = tok.strip()
    if tok and tok[0] in _QUOTES:
        q = tok[0]
        if len(tok) < 2 or tok[-1] != q:
            raise ValueError(f"malformed quoted value in spec: {tok!r}")
        return tok[1:-1].replace(q + q, q)
    low = tok.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        raise ValueError(
            f"unquoted value {tok!r} is not a number/bool/None — wrap strings in "
            f"quotes, e.g. key='some/string'")


def split_kv(field: str) -> tuple[str | None, str]:
    """Return (key, value) if the field has a top-level '=', else (None, field)."""
    quote: str | None = None
    for i, c in enumerate(field):
        if quote is not None:
            if c == quote:
                quote = None
        elif c in _QUOTES:
            quote = c
        elif c == "=":
            return field[:i], field[i + 1:]
    return None, field


def parse_component(part: str) -> tuple[str, list[str], dict[str, Any]]:
    """Parse one `NAME` or `(NAME, positional..., KEY=VALUE...)` component into
    (name, raw positional fields, parsed kwargs). Positional fields are returned
    unparsed so each caller can type them; they must precede every KEY=VALUE."""
    part = part.strip()
    if part.startswith("(") and part.endswith(")"):
        fields = [f.strip() for f in split_top_level(part[1:-1], ",")]
    else:
        fields = [part]

    name = fields[0]
    if not name or split_kv(name)[0] is not None:
        raise ValueError(f"missing name in spec component: {part!r}")

    positional: list[str] = []
    kwargs: dict[str, Any] = {}
    for f in fields[1:]:
        key, val = split_kv(f)
        if key is None:
            if kwargs:
                raise ValueError(f"positional field after keyword field in {part!r} "
                                 f"(positional fields must come before KEY=VALUE)")
            positional.append(f)
        else:
            kwargs[key.strip()] = parse_value(val)
    return name, positional, kwargs
