import re
from typing import List, Dict

ALIASES = {
    "reboot": "restart",
    "svc": "service",
}


def normalize_aliases(candidates: List[str]) -> List[str]:
    out = []
    for c in candidates:
        low = c.lower()
        out.append(ALIASES.get(low, c))
    return list(dict.fromkeys(out))


def normalize_time_windows(candidates: List[str]) -> List[str]:
    # already in PTnM / PTnH / PnD format from extractor; ensure uppercase
    out = []
    for c in candidates:
        if isinstance(c, str) and re.match(r"^p[T\d]", c, flags=re.I):
            out.append(c.upper())
        else:
            out.append(c)
    return out


def normalize_clauses(clauses: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {}
    for k, v in clauses.items():
        if not v:
            out[k] = []
            continue
        if k == "operation":
            out[k] = normalize_aliases(v)
        elif k == "time_window":
            out[k] = normalize_time_windows(v)
        else:
            # basic normalization: strip whitespace
            out[k] = [s.strip() for s in v if s is not None]
    return out
