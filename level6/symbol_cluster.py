# level6/symbol_cluster.py
"""
SymbolCluster — clusters failure ReasoningStates in predicate space.

The clustering substrate is the 11-dimensional predicate_probs vector
produced by the Level 5 predicate head.  Clustering in this named symbolic
space (rather than the 256-dim trunk) means every cluster centroid is
directly interpretable: e.g. {is_service: 0.91, is_sre_domain: 0.82,
has_runbook: 0.07} requires no LLM to understand.

Algorithm
---------
HDBSCAN (sklearn ≥ 1.3) on the [N, 11] predicate_probs matrix with cosine
metric.  Cosine is preferred over euclidean here because predicate_probs are
bounded in [0, 1] and we care about the *pattern* of activation (which
predicates fire together), not their absolute magnitude.  min_cluster_size
defaults to 15 to keep clusters large enough for lifecycle validation.

Per-cluster output fields
-------------------------
cluster_id           : int  (-1 = HDBSCAN noise — included but flagged)
symbol_name          : str  auto-derived from top-2 present + top-1 absent predicate
size                 : int  number of failure samples in this cluster
dominant_confusion   : str  top "predicted → gold" pair by count
present_predicates   : list[str]  centroid ≥ 0.70
absent_predicates    : list[str]  centroid ≤ 0.30
uncertain_predicates : list[str]  0.30 < centroid < 0.70
centroid             : dict[str, float]  mean predicate_probs per predicate
cohesion             : float  mean cosine similarity of members to centroid
grounding_quality    : float  cohesion × (n_stable / 11)
                              n_stable = |present| + |absent|  (unambiguous predicates)
example_utterances   : list[str]  up to N_EXAMPLES representative samples
majority_gold_intent      : str  mode of gold_intent in cluster
majority_predicted_intent : str  mode of predicted_intent in cluster
is_noise_cluster     : bool  True when cluster_id == -1

Usage
-----
    # Default: reads failure_set.jsonl, writes clusters.json
    python -m level6.symbol_cluster

    # Custom paths / tuning
    python -m level6.symbol_cluster \\
        --failure-set level6/data/failure_set.jsonl \\
        --out        level6/data/clusters.json \\
        --min-cluster-size 15 \\
        --min-samples      5 \\
        --n-examples       5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level5.model.dataset import PREDICATE_COLS, INTENT_LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESENT_THRESHOLD  = 0.70   # centroid ≥ this → predicate is "present"
ABSENT_THRESHOLD   = 0.30   # centroid ≤ this → predicate is "absent"
N_EXAMPLES         = 5      # utterances to store per cluster

DEFAULT_FAILURE_SET = REPO_ROOT / "level6" / "data" / "failure_set.jsonl"
DEFAULT_OUT         = REPO_ROOT / "level6" / "data" / "clusters.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_failures(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mean_cosine_to_centroid(members: np.ndarray, centroid: np.ndarray) -> float:
    """Mean cosine similarity of all member vectors to the centroid."""
    sims = [_cosine_sim(m, centroid) for m in members]
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Symbol name auto-derivation
# ---------------------------------------------------------------------------

def _symbol_name(centroid_dict: dict[str, float], dominant_confusion: str = "") -> str:
    """
    Two-mode symbol naming that reflects what the cluster *actually signals*.

    Mode 1 — Present-predicate mode (at least one centroid >= PRESENT_THRESHOLD):
        SYM_<top1_present>__<top2_present>__NOT_<top1_absent>
        Used when the model fires strongly on known predicates.  The centroid
        itself is the grounding.

    Mode 2 — Uncertainty-boundary mode (all centroids < PRESENT_THRESHOLD):
        SYM_<top_uncertain>__<predicted>_vs_<gold>
        Used when *no* predicate fires confidently.  The cluster does not
        represent a known concept — it represents a MISSING discriminative
        signal.  The confusion transition (predicted->gold) names what the
        model cannot separate.  This is cognitively more meaningful than
        listing the predicates that nearly-fired.

    Examples
    --------
    Strong predicates:
        centroid = {is_service: 0.91, is_sre_domain: 0.82, has_runbook: 0.07}
        -> "SYM_is_service__is_sre_domain__NOT_has_runbook"

    Uncertain centroid (the common case for boundary failures):
        top uncertain = is_sre_domain (0.44), confusion = investigate -> summarization
        -> "SYM_sre_domain__investigate_vs_summarization"
    """
    sorted_by_val = sorted(centroid_dict.items(), key=lambda x: x[1], reverse=True)
    present  = [k for k, v in sorted_by_val if v >= PRESENT_THRESHOLD]
    absent   = [k for k, v in sorted_by_val if v <= ABSENT_THRESHOLD]
    absent_sorted = sorted(absent, key=lambda k: centroid_dict[k])

    if present:
        # Mode 1: strong present predicates drive the name
        parts     = present[:2]
        not_parts = absent_sorted[:1]
        name = "SYM_" + "__".join(parts)
        if not_parts:
            name += "__NOT_" + not_parts[0]
        return name

    # Mode 2: uncertainty-boundary — the cluster marks missing discriminative evidence
    # Top uncertain = highest centroid still below PRESENT_THRESHOLD
    top_pred = sorted_by_val[0][0] if sorted_by_val else "unknown"
    # Strip leading "is_" for readability in the confusion-based name
    top_pred_clean = top_pred[3:] if top_pred.startswith("is_") else top_pred

    if dominant_confusion:
        parts = dominant_confusion.split(" -> ")
        if len(parts) == 2:
            pred_side = parts[0].strip()
            gold_side = parts[1].strip()
            return f"SYM_{top_pred_clean}__{pred_side}_vs_{gold_side}"

    # Fallback: best-effort from centroid alone
    name = "SYM_" + "__".join([sorted_by_val[0][0]])
    if absent_sorted:
        name += "__NOT_" + absent_sorted[0]
    return name


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def cluster_failures(
    failures: list[dict],
    min_cluster_size: int = 15,
    min_samples: int = 5,
    n_examples: int = N_EXAMPLES,
) -> list[dict]:
    """
    Cluster failure ReasoningStates in predicate_probs [11] space.

    Returns a list of cluster dicts ordered by cluster size (descending),
    noise cluster (-1) last.
    """
    if not failures:
        raise ValueError("No failure states provided - cannot cluster.")

    # ------------------------------------------------------------------ #
    # 1. Build feature matrix
    # ------------------------------------------------------------------ #
    X = np.array([f["predicate_probs"] for f in failures], dtype=np.float32)
    n, d = X.shape
    if d != len(PREDICATE_COLS):
        raise ValueError(
            f"predicate_probs dimension {d} does not match "
            f"PREDICATE_COLS length {len(PREDICATE_COLS)}"
        )

    # ------------------------------------------------------------------ #
    # 2. HDBSCAN in cosine metric space
    # ------------------------------------------------------------------ #
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels = hdb.fit_predict(X)

    unique_labels = sorted(set(labels))
    print(f"[SymbolCluster] {n} failures -> {len([l for l in unique_labels if l >= 0])} clusters "
          f"+ {int(np.sum(labels == -1))} noise points")

    # ------------------------------------------------------------------ #
    # 3. Per-cluster statistics
    # ------------------------------------------------------------------ #
    clusters: list[dict] = []

    for cid in unique_labels:
        mask     = labels == cid
        members  = X[mask]                    # [k, 11]
        members_rows = [f for f, m in zip(failures, mask) if m]

        centroid_arr = members.mean(axis=0)   # [11]
        centroid_dict = {
            col: round(float(centroid_arr[i]), 4)
            for i, col in enumerate(PREDICATE_COLS)
        }

        present     = [k for k, v in centroid_dict.items() if v >= PRESENT_THRESHOLD]
        absent      = [k for k, v in centroid_dict.items() if v <= ABSENT_THRESHOLD]
        uncertain   = [
            k for k, v in centroid_dict.items()
            if ABSENT_THRESHOLD < v < PRESENT_THRESHOLD
        ]
        n_stable    = len(present) + len(absent)
        cohesion    = _mean_cosine_to_centroid(members, centroid_arr)
        grounding_quality = round(cohesion * (n_stable / len(PREDICATE_COLS)), 4)

        # Dominant confusion  (predicted → gold)
        confusion_counts: Counter = Counter()
        for row in members_rows:
            pred_i = row.get("predicted_intent", "")
            gold_i = row.get("gold_intent", "")
            if pred_i and gold_i and pred_i != gold_i:
                confusion_counts[(pred_i, gold_i)] += 1
        dominant_confusion = ""
        if confusion_counts:
            (pi, gi), _ = confusion_counts.most_common(1)[0]
            dominant_confusion = f"{pi} -> {gi}"

        # Majority intents
        gold_counts = Counter(r.get("gold_intent", "") for r in members_rows)
        pred_counts = Counter(r.get("predicted_intent", "") for r in members_rows)
        majority_gold      = gold_counts.most_common(1)[0][0] if gold_counts else ""
        majority_predicted = pred_counts.most_common(1)[0][0] if pred_counts else ""

        # Representative utterances: prefer misclassifications for clarity
        misclass = [r["utterance"] for r in members_rows if r.get("is_misclassification")]
        others   = [r["utterance"] for r in members_rows if not r.get("is_misclassification")]
        examples = (misclass + others)[:n_examples]

        clusters.append({
            "cluster_id":                int(cid),
            "symbol_name":               _symbol_name(centroid_dict, dominant_confusion),
            "size":                      int(members.shape[0]),
            "is_noise_cluster":          cid == -1,
            "dominant_confusion":        dominant_confusion,
            "present_predicates":        sorted(present, key=lambda k: -centroid_dict[k]),
            "absent_predicates":         sorted(absent,  key=lambda k:  centroid_dict[k]),
            "uncertain_predicates":      sorted(uncertain, key=lambda k: centroid_dict[k]),
            "centroid":                  centroid_dict,
            "cohesion":                  round(cohesion, 4),
            "grounding_quality":         grounding_quality,
            "example_utterances":        examples,
            "majority_gold_intent":      majority_gold,
            "majority_predicted_intent": majority_predicted,
        })

    # Sort: real clusters by descending size, noise cluster last
    real   = sorted([c for c in clusters if not c["is_noise_cluster"]], key=lambda c: -c["size"])
    noise  = [c for c in clusters if c["is_noise_cluster"]]
    return real + noise


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_clusters(clusters: list[dict]):
    print()
    print("=" * 70)
    print("  SymbolCluster Results")
    print("=" * 70)
    real   = [c for c in clusters if not c["is_noise_cluster"]]
    noise  = [c for c in clusters if c["is_noise_cluster"]]
    total_in_clusters = sum(c["size"] for c in real)
    total_noise       = noise[0]["size"] if noise else 0

    print(f"  Clusters formed  : {len(real)}")
    print(f"  Points clustered : {total_in_clusters}")
    print(f"  Noise points     : {total_noise}")
    print()

    for c in real:
        print(f"  [{c['cluster_id']}] {c['symbol_name']}")
        print(f"      size         : {c['size']}")
        print(f"      cohesion     : {c['cohesion']:.4f}  "
              f"grounding_quality: {c['grounding_quality']:.4f}")
        print(f"      majority     : gold={c['majority_gold_intent']}  "
              f"predicted={c['majority_predicted_intent']}")
        print(f"      confusion    : {c['dominant_confusion']}")
        print(f"      present      : {c['present_predicates']}")
        print(f"      absent       : {c['absent_predicates']}")
        print(f"      uncertain    : {c['uncertain_predicates']}")
        print(f"      examples     :")
        for u in c["example_utterances"]:
            print(f"        - {u}")
        print()

    if noise:
        n = noise[0]
        print(f"  [NOISE] size={n['size']}  "
              f"(below min_cluster_size - not eligible for symbol birth)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_clusters(clusters: list[dict], out_path: Path):
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(clusters, fh, indent=2, cls=_NumpyEncoder)
    print(f"\n[SymbolCluster] clusters.json -> {out_path}  ({len(clusters)} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster failure states in predicate space.")
    p.add_argument("--failure-set",      type=Path, default=DEFAULT_FAILURE_SET)
    p.add_argument("--out",              type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-cluster-size", type=int,  default=15)
    p.add_argument("--min-samples",      type=int,  default=5)
    p.add_argument("--n-examples",       type=int,  default=N_EXAMPLES)
    p.add_argument("--dry-run",          action="store_true",
                   help="Print results but do not write clusters.json")
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.failure_set.exists():
        print(f"[SymbolCluster] ERROR: failure set not found: {args.failure_set}")
        print("  Run:  python -m level6.failure_collector  first.")
        sys.exit(1)

    print(f"[SymbolCluster] Loading failures from: {args.failure_set}")
    failures = load_failures(args.failure_set)
    print(f"[SymbolCluster] {len(failures)} failure states loaded")

    clusters = cluster_failures(
        failures,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        n_examples=args.n_examples,
    )

    print_clusters(clusters)

    if not args.dry_run:
        save_clusters(clusters, args.out)


if __name__ == "__main__":
    main()
