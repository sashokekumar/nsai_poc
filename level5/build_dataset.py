"""
level5/build_dataset.py
-----------------------
Derives the Level 5 predicate dataset from level4/data/labeled_clean.csv.

Adds 11 binary predicate columns:
  Entity-type predicates (7):  one per entity_type label from level4
    is_infrastructure, is_service, is_metric, is_incident,
    is_job, is_pipeline, is_unknown
  Signal predicates (4):  keyword-derived from utterance text
    is_sre_domain, has_runbook, is_known_incident, is_metric_query

Output: level5/data/level5_labeled.csv

Run from repo root:
    python -m level5.build_dataset
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
SRC_CSV   = REPO_ROOT / "level4" / "data" / "labeled_clean.csv"
OUT_DIR   = REPO_ROOT / "level5" / "data"
OUT_CSV   = OUT_DIR / "level5_labeled.csv"

# ---------------------------------------------------------------------------
# Entity-type predicate derivation
# Each is a direct one-hot of the existing entity_type column
# ---------------------------------------------------------------------------
ENTITY_TYPES = ["infrastructure", "service", "metric",
                "incident", "job", "pipeline", "unknown"]

# ---------------------------------------------------------------------------
# Signal predicate keyword patterns
# ---------------------------------------------------------------------------
_SRE_KEYWORDS = re.compile(
    r"\b(pod|node|cluster|namespace|deployment|replica|service|mesh|gateway|"
    r"dns|latency|cpu|memory|disk|io|throughput|error.rate|sla|slo|alert|"
    r"incident|runbook|pipeline|job|kafka|queue|metric|log|trace|monitor|"
    r"observ|autoscal|health.check|circuit.break|rate.limit|tls|certificate|"
    r"backup|storage|database|cache|replica|load.balanc|proxy|api)\b",
    re.IGNORECASE,
)

_RUNBOOK_KEYWORDS = re.compile(
    r"\b(runbook|playbook|procedure|remediat|mitigat|rollback|restart|scale|"
    r"configure|set|inject|enable|disable|deploy|execute|run|trigger|apply)\b",
    re.IGNORECASE,
)

_INCIDENT_KEYWORDS = re.compile(
    r"\b(incident|outage|degradat|failure|alert|pagerduty|postmortem|"
    r"root.cause|rca|impact|down|crash|spike|anomal)\b",
    re.IGNORECASE,
)

_METRIC_KEYWORDS = re.compile(
    r"\b(metric|cpu|memory|latency|throughput|error.rate|p99|p95|percentile|"
    r"utiliz|trend|cost|sla|slo|dashboard|grafana|prometh|usage|saturat)\b",
    re.IGNORECASE,
)


def derive_predicates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # -- Entity-type one-hots (from existing column, guaranteed clean) --
    for et in ENTITY_TYPES:
        col = f"is_{et}"
        out[col] = (df["entity_type"] == et).astype(int)

    # -- Signal predicates (keyword-derived from utterance) --
    utt = df["utterance"].str.lower().fillna("")
    out["is_sre_domain"]      = utt.apply(lambda u: int(bool(_SRE_KEYWORDS.search(u))))
    out["has_runbook"]        = utt.apply(lambda u: int(bool(_RUNBOOK_KEYWORDS.search(u))))
    out["is_known_incident"]  = utt.apply(lambda u: int(bool(_INCIDENT_KEYWORDS.search(u))))
    out["is_metric_query"]    = utt.apply(lambda u: int(bool(_METRIC_KEYWORDS.search(u))))

    return out


def audit(df: pd.DataFrame):
    print(f"{'='*60}")
    print(f"  Level 5 Dataset Audit")
    print(f"{'='*60}")
    print(f"Rows : {len(df)}")
    print(f"Cols : {list(df.columns)}\n")

    print("Intent distribution:")
    print(df["intent"].value_counts().to_string())
    print()

    predicate_cols = [f"is_{et}" for et in ENTITY_TYPES] + \
                     ["is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query"]
    print("Predicate coverage (% rows = 1):")
    for col in predicate_cols:
        pct = df[col].mean() * 100
        print(f"  {col:<22}  {df[col].sum():>5} / {len(df)}  ({pct:5.1f}%)")
    print()

    print("Null check:", df.isnull().sum()[df.isnull().sum() > 0].to_dict() or "none")
    print()

    # Predicate→intent correlation for key signal predicates
    print("Signal predicates: mean intent distribution when predicate=1")
    for col in ["is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query"]:
        sub = df[df[col] == 1]["intent"].value_counts(normalize=True).round(2)
        print(f"  {col}: {sub.to_dict()}")
    print()

    print("VERDICT: level5_labeled.csv READY" if df.isnull().sum().sum() == 0
          else "WARNING: nulls found")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {SRC_CSV}")
    df = pd.read_csv(SRC_CSV)
    print(f"  Base dataset: {len(df)} rows, {len(df.columns)} cols\n")

    df_out = derive_predicates(df)

    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    audit(df_out)


if __name__ == "__main__":
    main()
