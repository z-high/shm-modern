"""SH command line tokenizer, qualifier parser, and safe CALC expression evaluator."""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from typing import Dict, List, Optional, Tuple


_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

_SAFE_MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log10,
    "abs": abs,
}


def _eval_math_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTANTS:
        return _SAFE_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_math_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](_eval_math_ast(node.left), _eval_math_ast(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_MATH_FUNCS and not node.keywords:
        args = [_eval_math_ast(a) for a in node.args]
        return _SAFE_MATH_FUNCS[node.func.id](*args)
    raise ValueError(f"Unsupported expression node in CALC: {ast.dump(node)}")


def _safe_eval_math_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval_math_ast(tree)


_KNOWN_QUALIFIERS = frozenset({
    "ALL", "ALLCOMP", "APPEND", "ARRAY", "AUTOCUT", "BAZ", "BAZ_THEO",
    "BEACHBALL", "BGR", "BOATWRIGHT", "BRUNE", "CHA", "CHAN", "CLEAR",
    "COLOR", "COLOUR", "DEPTH", "DIST", "DIST_M", "DRY", "EXCLUDE",
    "F_HI", "F_LO", "FFT", "FILE", "FILTER", "FM", "FMT", "FOCAL",
    "FORMAT", "FREE", "GEIGER", "INPUT", "INV", "LAT", "LIST", "LOC",
    "LOCATE", "LOCATION", "LON", "LPC", "MAX_CHI2", "MDIR", "MEM",
    "MODE", "NET", "NETWORK", "NOAPPLY", "NOCALIB", "NOELLIPSE", "NOISE_HI",
    "NOISE_LO", "NONE", "NONORM", "NOREFINE", "NOSFD", "NOSTATIONS",
    "NOTSTAR", "NOUPDATE", "OFF", "ON", "ORIGIN", "OUT", "PDF",
    "PERIOD_HI", "PERIOD_LO", "PHASE", "PHASES", "PLANE", "POL",
    "POLARIZATION", "PREFIX", "PSD", "REC", "REL", "RHO", "RHO_KG_M3",
    "SETTLE", "SIGMA_S", "SIGMA_T", "SLO", "SLOWNESS", "STATION", "STEP",
    "SVG", "T_HI", "T_LO", "TIME", "TITLE", "TMAX", "TMIN", "URL",
    "VEL", "VEL_M_S", "VELOCITY", "WAVE", "ZERO", "ZEROPHASE", "AGENCY",
})

_IO_PATH_COMMANDS = frozenset({
    "READ", "WRITE", "READS", "READM", "READSEED", "WRITESEED",
    "READGSE", "WRITEGSE", "READA", "WRITEA", "READEVT", "WRITEEVT",
    "READSXML", "WRITEQML", "READQML", "WRITEGEOJSON", "REPORT",
    "SFD", "CALIB",
})


def split_statements(s: str) -> List[str]:
    """Split a command line on ';' outside balanced single/double quotes."""
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n and s[i + 1] == '"':
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
        else:
            if ch in ('"', "'"):
                k = i + 1
                has_close = False
                while k < n:
                    if ch == '"' and s[k] == "\\" and k + 1 < n and s[k + 1] == '"':
                        k += 2
                        continue
                    if s[k] == ch:
                        has_close = True
                        break
                    k += 1
                if has_close:
                    quote = ch
                buf.append(ch)
                i += 1
            elif ch == ";":
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 1
            else:
                buf.append(ch)
                i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def split_quoted_tokens(cmd: str) -> List[Tuple[str, bool]]:
    """Split command string into (token, started_quoted) pairs preserving spaces inside quotes."""
    result: List[Tuple[str, bool]] = []
    i = 0
    n = len(cmd)
    while i < n:
        while i < n and cmd[i].isspace():
            i += 1
        if i >= n:
            break
        buf: List[str] = []
        started_quoted = False
        token_start = i
        while i < n and not cmd[i].isspace():
            ch = cmd[i]
            if ch in ('"', "'"):
                k = i + 1
                has_close = False
                while k < n:
                    if ch == '"' and cmd[k] == "\\" and k + 1 < n and cmd[k + 1] == '"':
                        k += 2
                        continue
                    if cmd[k] == ch:
                        has_close = True
                        break
                    k += 1
                if has_close:
                    if i == token_start:
                        started_quoted = True
                    i += 1
                    while i < n:
                        if ch == '"' and cmd[i] == "\\" and i + 1 < n and cmd[i + 1] == '"':
                            buf.append('"')
                            i += 2
                            continue
                        if cmd[i] == ch:
                            i += 1
                            break
                        buf.append(cmd[i])
                        i += 1
                    continue
            buf.append(ch)
            i += 1
        result.append(("".join(buf), started_quoted))
    return result


def is_qualifier_token(p: str, was_quoted: bool, verb: Optional[str]) -> bool:
    """Disambiguate SH /QUALIFIER[=VAL] tokens from Unix filesystem paths."""
    if was_quoted or not p.startswith("/") or len(p) <= 1 or verb is None:
        return False
    if os.path.exists(p):
        return False
    parent = os.path.dirname(p)
    if parent not in ("", "/") and os.path.isdir(parent):
        return False

    body = p[1:]
    if "=" in body:
        head, _val = body.split("=", 1)
        if not head:
            return False
        if "/" not in head:
            return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", head))
        head_segs = head.split("/")
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", seg) for seg in head_segs):
            return False
        return verb not in _IO_PATH_COMMANDS or all(seg.upper() in _KNOWN_QUALIFIERS for seg in head_segs)

    segs = body.split("/")
    if any(not seg or "." in seg or "~" in seg for seg in segs):
        return False
    if all(seg.upper() in _KNOWN_QUALIFIERS for seg in segs):
        return True
    if verb not in _IO_PATH_COMMANDS and len(segs) <= 3:
        if all(re.fullmatch(r"[A-Z][A-Z0-9_]*", seg) for seg in segs):
            return True
    return False


def tokenize_command(cmd: str) -> Tuple[List[str], Dict[str, str]]:
    """Split SH command line into positional arguments and /QUALIFIER[=VALUE] dictionary."""
    raw_parts = split_quoted_tokens(cmd)
    tokens: List[str] = []
    quals: Dict[str, str] = {}

    for p, was_quoted in raw_parts:
        verb = tokens[0].upper() if tokens else None
        if is_qualifier_token(p, was_quoted, verb):
            q_body = p[1:]
            if "=" in q_body:
                head, v = q_body.split("=", 1)
                if "/" in head:
                    head_segs = head.split("/")
                    for sub_q in head_segs[:-1]:
                        if sub_q:
                            quals[sub_q.strip().upper()] = "TRUE"
                    quals[head_segs[-1].strip().upper()] = v.strip()
                else:
                    quals[head.strip().upper()] = v.strip()
            else:
                for sub_q in q_body.split("/"):
                    if sub_q:
                        quals[sub_q.strip().upper()] = "TRUE"
        else:
            tokens.append(p)
    return tokens, quals

