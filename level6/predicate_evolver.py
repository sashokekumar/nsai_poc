# level6/predicate_evolver.py
"""
PredicateEvolver -- Task 20: Discriminative Predicate Evolution.

When a candidate rule passes accuracy_delta_noretrain but fails the FPR gate
the antecedent is too broad: the rule fires correctly on the target failure
cluster but also fires destructively on previously-correct non-failure rows.

This module mines WHY the rule fires too broadly and proposes corrections:

  Phase 1 -- Discriminant Analysis
    For each FPR-failing symbol:
      a. Re-run the injected model forward pass to identify:
           TP rows -- cluster members where the rule CORRECTLY fixes the prediction
                      (baseline wrong, injected right)
           FP rows -- non-failure rows where the rule INCORRECTLY breaks the prediction
                      (baseline right, injected wrong)
      b. Compute mean predicate_probs for TP vs FP groups.
      c. Rank existing predicates by net_benefit:
           net_benefit = fp_exclusion_rate - tp_exclusion_rate
         where fp_exclusion_rate is the fraction of FP rows that would be
         excluded if we added this predicate as an antecedent condition, and
         tp_exclusion_rate is the corresponding TP rows that would also be lost.

  Phase 2 -- Antecedent Refinement
    Apply the top-K discriminant conditions to estimate residual FPR.
    If residual_fpr < 0.10: the existing vocabulary is sufficient; the
    candidate rule just needs a tighter antecedent.

  Phase 3 -- New Predicate Proposal
    If residual_fpr >= 0.10 after applying all beneficial existing-predicate
    conditions: the current 11-predicate vocabulary has a gap.  Emit a
    PredicateProposal naming the missing discriminative predicate.

The FPR failures from Task 11 tell us exactly where the vocabulary gap is:
  S_001 (is_sre_domain -> investigate): fires on all SRE queries, but
    SRE queries span investigate, summarization, and execution.  We need
    a predicate that marks "this is a diagnostic/investigative SRE query"
    rather than a summary or execution request.
    -> Proposed: is_diagnostic_request

  S_003 (is_unknown -> summarization): fires on all unknown queries, but
    unknown queries include genuine out-of-scope utterances.  We need a
    predicate that marks "this is an SRE-domain summary request even though
    the utterance is not clearly categorised by existing predicates."
    -> Proposed: has_sre_summary_target

Outputs
-------
  level6/data/antecedent_refinements.json  -- per-symbol tighter antecedents
  level6/data/predicate_proposals.json     -- new predicate proposals

Usage
-----
    python -m level6.predicate_evolver

    # Analyse only specific symbols
    python -m level6.predicate_evolver --symbol S_001 S_003

    # Change number of top discriminants to apply
    python -m level6.predicate_evolver --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level5.model.dataset import INTENT_LABELS, PREDICATE_COLS   # noqa: E402
from level5.model.level5_model import Level5IntentModel           # noqa: E402
from level6.rule_validator import (                               # noqa: E402
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_CLUSTERS,
    DEFAULT_FAILURE_SET,
    DEFAULT_FULL_INFERENCE,
    DEFAULT_REGISTRY,
    DEFAULT_RULE_BASE,
    REPORT_DIR,
    _inject_rule,
    _intent_to_idx,
    _load_model_with_rule_base,
    _match_cluster_members,
    _predict_from_predicates,
)
from level6.symbol_registry import SymbolRegistry                 # noqa: E402

DATA_DIR = REPO_ROOT / "level6" / "data"

# Thresholds for antecedent condition generation.
# A predicate is added as a positive condition (AND pred >= threshold)
# when its TP mean >= CONDITION_HIGH_THRESHOLD.
# It is added as a negative condition (AND NOT pred <= threshold)
# when its FP mean >= CONDITION_LOW_THRESHOLD (predicate is prevalent in FP
# but not in TP -- adding AND NOT pred excludes many FP rows).
CONDITION_HIGH_THRESHOLD = 0.55
CONDITION_LOW_THRESHOLD  = 0.45

# FPR target after antecedent refinement.
FPR_TARGET = 0.10

# Maximum discriminant predicates to apply for residual FPR estimation.
DEFAULT_TOP_K = 4

# ---------------------------------------------------------------------------
# Predicate gap name lookup
# Key: (majority_gold_intent, majority_predicted_intent)
# Value: (proposed_predicate_name, rationale, suggested_definition)
# ---------------------------------------------------------------------------

_GAP_LOOKUP: dict[tuple[str, str], tuple[str, str, str]] = {
    # S_001 pattern: gold=investigate, predicted=investigate (low-confidence cluster)
    # Rule fires too broadly on SRE-domain rows that should be summarization
    ("investigate", "investigate"): (
        "is_diagnostic_request",
        "Utterance requests root-cause investigation, anomaly analysis, or metric diagnosis "
        "rather than a status summary. Discriminates investigate from summarization in the "
        "SRE-domain uncertainty region (cluster S_001) where 'is_sre_domain' alone fires too "
        "broadly on summarization and execution queries.",
        "Detect diagnostic intent markers: action-oriented verbs ('diagnose', 'investigate', "
        "'trace', 'drill down', 'root cause', 'identify cause', 'why is', 'what caused') "
        "combined with absence of summary verbs ('summarize', 'brief', 'overview', 'recap', "
        "'status of'). Can be computed as a logistic classifier over TF-IDF n-gram features "
        "trained on labeled investigate vs summarization utterances.",
    ),
    # S_003 pattern: gold=summarization, predicted=out_of_scope
    # Rule fires too broadly on unknown rows that are genuinely out-of-scope
    ("summarization", "out_of_scope"): (
        "has_sre_summary_target",
        "Utterance requests a summary of a specific SRE artifact (alerts, incidents, "
        "deployments, latency trends, audit findings) rather than a generic off-topic question. "
        "Discriminates true SRE summarization requests from out-of-scope queries in the "
        "is_unknown cluster (S_003) where 'is_unknown AND NOT is_incident' fires too broadly.",
        "Detect SRE summary targets: SRE nouns ('alerts', 'incidents', 'deployments', "
        "'metrics', 'latency', 'trends', 'findings', 'open incidents', 'observability', "
        "'stack health', 'service status') combined with summary verbs ('summarize', 'brief', "
        "'overview', 'recap', 'give me a summary of', 'provide a brief on'). Can be computed "
        "as entity-presence score: count of SRE domain terms per token in the utterance.",
    ),
    # Additional patterns for future use
    ("investigate", "summarization"): (
        "is_diagnostic_request",
        "Utterance is an investigative query requiring root-cause analysis, not a summary.",
        "Detect diagnostic markers: 'why', 'root cause', 'investigate', 'trace', 'diagnose'.",
    ),
    ("summarization", "investigate"): (
        "is_status_summary_request",
        "Utterance requests current state/summary, not active investigation.",
        "Detect summary markers: 'summarize', 'brief', 'status', 'recap', 'overview'.",
    ),
    ("execution", "investigate"): (
        "has_explicit_runbook_trigger",
        "Utterance specifies a concrete runbook or job to execute, not to investigate.",
        "Detect execution markers: 'run', 'trigger', 'execute' combined with runbook/job IDs.",
    ),
    ("execution", "out_of_scope"): (
        "has_executable_sre_action",
        "Utterance is a concrete SRE execution request, not off-topic.",
        "Detect SRE execution verbs ('run', 'trigger', 'deploy') + SRE object names.",
    ),
    ("out_of_scope", "summarization"): (
        "is_non_sre_query",
        "Utterance is outside the SRE knowledge domain.",
        "Detect absence of SRE terminology + generic question patterns ('what is', 'define').",
    ),
}


def _lookup_predicate_gap(
    majority_gold_intent: str,
    majority_predicted_intent: str,
) -> tuple[str, str, str]:
    """Return (proposed_name, rationale, suggested_definition) for the vocabulary gap."""
    key = (majority_gold_intent, majority_predicted_intent)
    if key in _GAP_LOOKUP:
        return _GAP_LOOKUP[key]
    # Fallback: generate a descriptive name
    name = (
        f"discriminates_{majority_gold_intent}_from_{majority_predicted_intent}"
        .replace(" ", "_").replace("-", "_")
    )
    rationale = (
        f"Discriminates {majority_gold_intent} from {majority_predicted_intent} in "
        f"the boundary region where existing 11 predicates are insufficient."
    )
    definition = "To be defined by domain experts based on utterance analysis."
    return name, rationale, definition


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiscriminantResult:
    """Per-predicate discriminant analysis result."""
    predicate: str
    mu_tp: float                # mean predicate prob in TP group
    mu_fp: float                # mean predicate prob in FP group
    delta: float                # |mu_tp - mu_fp|
    direction: str              # "high_in_tp" or "high_in_fp"
    condition_type: str         # "AND" or "AND NOT"
    fp_exclusion_rate: float    # fraction of FP rows excluded by this condition
    tp_exclusion_rate: float    # fraction of TP rows also excluded (collateral loss)
    net_benefit: float          # fp_exclusion_rate - tp_exclusion_rate


@dataclass
class AntecedentRefinement:
    """
    Tighter antecedent suggestion using only existing predicates.

    The original candidate rule's antecedent is supplemented with additional
    conditions derived from the top discriminant predicates.
    """
    symbol_id: str
    symbol_name: str
    original_antecedent_description: str
    candidate_rule_name: str
    original_fpr: float
    n_tp: int                              # TP rows (failures correctly fixed)
    n_fp: int                              # FP rows (non-failures incorrectly flipped)
    n_previously_correct: int              # denominator for FPR
    top_discriminants: list[dict]          # top-k DiscriminantResult dicts
    suggested_additional_conditions: list[str]   # e.g. ["AND is_metric >= 0.55"]
    estimated_residual_fpr: float
    can_resolve_with_existing: bool        # residual_fpr < FPR_TARGET
    tp_predicate_means: dict[str, float]
    fp_predicate_means: dict[str, float]


@dataclass
class PredicateProposal:
    """
    New discriminative predicate not in the current 11-column vocabulary.

    Emitted when no combination of existing predicates can reduce FPR below
    the 0.10 threshold for a given symbol.
    """
    proposal_id: str                       # PP_001, PP_002, ...
    source_symbol_id: str
    source_cluster_id: int
    majority_gold_intent: str
    majority_predicted_intent: str
    dominant_confusion: str
    candidate_rule_name: str
    proposed_predicate_name: str
    rationale: str
    suggested_definition: str
    top_existing_discriminants: list[dict]  # closest existing predicates for context
    example_tp_utterances: list[str]        # utterances the rule correctly fixes
    example_fp_utterances: list[str]        # utterances the rule incorrectly breaks
    estimated_minimum_coverage: float       # min fraction of FP rows new pred must exclude
    residual_fpr_after_existing: float      # FPR after best existing-pred tightening
    status: str = "proposed"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_baseline_model(
    checkpoint_path: Path,
    rule_base_path: Path,
    device: torch.device,
) -> Level5IntentModel:
    """
    Load the original 4-rule Level 5 model from checkpoint.
    Uses strict=False to tolerate extra keys from older checkpoint architectures.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level5IntentModel(
        rule_base_path=str(rule_base_path),
        hard_rules=ckpt.get("hard_rules", False),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model


def _get_cluster_for_symbol(
    clusters: list[dict],
    symbol_name: str,
) -> Optional[dict]:
    """Find the cluster matching this symbol by symbol_name."""
    for c in clusters:
        if c.get("symbol_name") == symbol_name:
            return c
    return None


# ---------------------------------------------------------------------------
# TP / FP identification
# ---------------------------------------------------------------------------

@torch.no_grad()
def _identify_tp_fp_rows(
    cluster_members: list[dict],
    non_failure_rows: list[dict],
    baseline_model: Level5IntentModel,
    injected_model: Level5IntentModel,
    device: torch.device,
) -> tuple[list[dict], list[dict], int]:
    """
    Identify true-positive and false-positive rows.

    TP rows: cluster members where baseline is WRONG and injected is CORRECT.
    FP rows: non-failure rows where baseline is CORRECT and injected is WRONG.

    Returns (tp_rows, fp_rows, n_previously_correct).
    n_previously_correct is the FPR denominator: total non-failure rows where
    baseline was correct.
    """
    # --- cluster members (potential TP) ---
    tp_rows: list[dict] = []
    if cluster_members:
        pm_c   = np.array([r["predicate_probs"] for r in cluster_members], dtype=np.float32)
        gold_c = np.array([_intent_to_idx(r["gold_intent"]) for r in cluster_members])
        base_c = _predict_from_predicates(baseline_model, pm_c, device)
        inj_c  = _predict_from_predicates(injected_model,  pm_c, device)
        for i, row in enumerate(cluster_members):
            g = gold_c[i]
            if g >= 0 and base_c[i] != g and inj_c[i] == g:
                tp_rows.append(row)

    # --- non-failure rows (potential FP) ---
    fp_rows: list[dict] = []
    n_previously_correct = 0
    if non_failure_rows:
        pm_n   = np.array([r["predicate_probs"] for r in non_failure_rows], dtype=np.float32)
        gold_n = np.array([_intent_to_idx(r["gold_intent"]) for r in non_failure_rows])
        base_n = _predict_from_predicates(baseline_model, pm_n, device)
        inj_n  = _predict_from_predicates(injected_model,  pm_n, device)

        valid = gold_n >= 0
        previously_correct_mask = valid & (base_n == gold_n)
        n_previously_correct = int(np.sum(previously_correct_mask))

        for i, row in enumerate(non_failure_rows):
            g = gold_n[i]
            if g >= 0 and base_n[i] == g and inj_n[i] != g:
                fp_rows.append(row)

    return tp_rows, fp_rows, n_previously_correct


# ---------------------------------------------------------------------------
# Discriminant analysis
# ---------------------------------------------------------------------------

def _discriminant_analysis(
    tp_rows: list[dict],
    fp_rows: list[dict],
) -> tuple[list[DiscriminantResult], dict[str, float], dict[str, float]]:
    """
    Rank existing predicates by their ability to separate TP from FP rows.

    For each predicate:
      - If mu_tp > mu_fp: adding AND predicate >= CONDITION_HIGH_THRESHOLD
        excludes FP rows where predicate is low, at the cost of also excluding
        TP rows where predicate is low.
      - If mu_fp > mu_tp: adding AND NOT predicate (i.e., predicate <= threshold)
        excludes FP rows where predicate is high, at the cost of TP rows too.

    Ranked by net_benefit = fp_exclusion_rate - tp_exclusion_rate.
    """
    if not tp_rows or not fp_rows:
        return [], {}, {}

    tp_mat = np.array([r["predicate_probs"] for r in tp_rows], dtype=np.float32)   # [Ntp, 11]
    fp_mat = np.array([r["predicate_probs"] for r in fp_rows], dtype=np.float32)   # [Nfp, 11]

    tp_means = tp_mat.mean(axis=0)
    fp_means = fp_mat.mean(axis=0)

    results: list[DiscriminantResult] = []
    for i, pred in enumerate(PREDICATE_COLS):
        mu_tp = float(tp_means[i])
        mu_fp = float(fp_means[i])
        delta = abs(mu_tp - mu_fp)

        if mu_tp >= mu_fp:
            # Predicate is higher in TP -> add AND pred condition
            direction      = "high_in_tp"
            condition_type = "AND"
            # FP rows excluded: those where predicate_prob < CONDITION_HIGH_THRESHOLD
            fp_excl = int(np.sum(fp_mat[:, i] < CONDITION_HIGH_THRESHOLD))
            tp_excl = int(np.sum(tp_mat[:, i] < CONDITION_HIGH_THRESHOLD))
        else:
            # Predicate is higher in FP -> add AND NOT pred condition
            direction      = "high_in_fp"
            condition_type = "AND NOT"
            # FP rows excluded: those where predicate_prob > CONDITION_LOW_THRESHOLD
            fp_excl = int(np.sum(fp_mat[:, i] > CONDITION_LOW_THRESHOLD))
            tp_excl = int(np.sum(tp_mat[:, i] > CONDITION_LOW_THRESHOLD))

        fp_exclusion_rate = fp_excl / len(fp_rows)
        tp_exclusion_rate = tp_excl / len(tp_rows)
        net_benefit       = fp_exclusion_rate - tp_exclusion_rate

        results.append(DiscriminantResult(
            predicate         = pred,
            mu_tp             = round(mu_tp, 4),
            mu_fp             = round(mu_fp, 4),
            delta             = round(delta, 4),
            direction         = direction,
            condition_type    = condition_type,
            fp_exclusion_rate = round(fp_exclusion_rate, 4),
            tp_exclusion_rate = round(tp_exclusion_rate, 4),
            net_benefit       = round(net_benefit, 4),
        ))

    results.sort(key=lambda r: -r.net_benefit)

    tp_means_dict = {PREDICATE_COLS[i]: round(float(tp_means[i]), 4) for i in range(len(PREDICATE_COLS))}
    fp_means_dict = {PREDICATE_COLS[i]: round(float(fp_means[i]), 4) for i in range(len(PREDICATE_COLS))}
    return results, tp_means_dict, fp_means_dict


# ---------------------------------------------------------------------------
# Residual FPR estimation
# ---------------------------------------------------------------------------

def _estimate_residual_fpr(
    fp_rows: list[dict],
    discriminants: list[DiscriminantResult],
    top_k: int,
    n_previously_correct: int,
) -> tuple[float, list[str]]:
    """
    Simulate applying the top-K discriminant conditions to the FP rows.

    Conditions are applied greedily (sequential AND logic).
    Returns (estimated_residual_fpr, list_of_condition_strings).
    """
    if not fp_rows or not discriminants or n_previously_correct == 0:
        return 0.0, []

    fp_mat  = np.array([r["predicate_probs"] for r in fp_rows], dtype=np.float32)
    active  = np.ones(len(fp_rows), dtype=bool)   # FP rows still not excluded
    conditions: list[str] = []

    for disc in discriminants[:top_k]:
        if disc.net_benefit <= 0.0:
            break   # no further improvement possible
        col = PREDICATE_COLS.index(disc.predicate)
        if disc.condition_type == "AND":
            # Exclude FP rows where predicate < threshold (they fail the AND condition)
            active &= fp_mat[:, col] >= CONDITION_HIGH_THRESHOLD
            conditions.append(f"AND {disc.predicate} >= {CONDITION_HIGH_THRESHOLD}")
        else:
            # Exclude FP rows where predicate > threshold (they fail the AND NOT condition)
            active &= fp_mat[:, col] <= CONDITION_LOW_THRESHOLD
            conditions.append(f"AND NOT {disc.predicate} > {CONDITION_LOW_THRESHOLD}")

    residual_fp  = int(np.sum(active))
    residual_fpr = residual_fp / n_previously_correct
    return round(residual_fpr, 4), conditions


# ---------------------------------------------------------------------------
# Per-symbol analysis
# ---------------------------------------------------------------------------

def analyze_symbol(
    symbol_id: str,
    registry_sym: dict,
    candidate_dir: Path,
    report_dir: Path,
    failure_rows: list[dict],
    non_failure_rows: list[dict],
    clusters: list[dict],
    existing_rule_base: dict,
    checkpoint_path: Path,
    baseline_model: Level5IntentModel,
    device: torch.device,
    top_k: int,
) -> tuple[Optional[AntecedentRefinement], Optional[PredicateProposal]]:
    """
    Full discriminant analysis for one FPR-failing symbol.

    Returns (AntecedentRefinement, PredicateProposal) where PredicateProposal
    is None if existing predicates are sufficient to tighten the antecedent.
    Returns (None, None) if the symbol is not FPR-limited or data is missing.
    """
    sym_name = registry_sym.get("name", symbol_id)

    # Load validation report
    report_path = report_dir / f"report_{symbol_id}.json"
    if not report_path.exists():
        print(f"  [{symbol_id}] No validation report -- skipping")
        return None, None

    with open(report_path) as f:
        report = json.load(f)

    fpr       = report.get("false_positive_rate", 0.0)
    acc_delta = report.get("accuracy_delta_noretrain", 0.0)

    # Only analyse symbols where FPR is the bottleneck (acc_delta OK, FPR too high)
    if acc_delta < 0.03:
        reason = f"acc_delta ({acc_delta:+.4f}) is below threshold -- rule validity issue, not antecedent breadth"
        print(f"  [{symbol_id}] SKIP: {reason}")
        return None, None

    if fpr < FPR_TARGET:
        print(f"  [{symbol_id}] FPR ({fpr:.4f}) already acceptable -- no refinement needed")
        return None, None

    print(f"\n  [{symbol_id}] {sym_name}")
    print(f"      acc_delta         : {acc_delta:+.4f}  (OK)")
    print(f"      original_fpr      : {fpr:.4f}  (FAIL -- too high)")

    # Load candidate rule JSON
    candidate_files = sorted(candidate_dir.glob(f"R_{symbol_id}*.json"))
    if not candidate_files:
        print(f"  [{symbol_id}] No candidate rule JSON in {candidate_dir} -- skipping")
        return None, None

    with open(candidate_files[0]) as f:
        candidate_rule = json.load(f)

    candidate_rule_name = candidate_rule.get("name", f"R_{symbol_id}")

    # Build injected rule_base and load injected model
    injected_rb = _inject_rule(existing_rule_base, candidate_rule)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(injected_rb, tf, indent=2)
        tmp_path = tf.name

    try:
        injected_model = _load_model_with_rule_base(checkpoint_path, tmp_path, device)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Find cluster
    cluster = _get_cluster_for_symbol(clusters, sym_name)
    if cluster is None:
        print(f"  [{symbol_id}] Could not find cluster by symbol_name -- skipping")
        return None, None

    # Find cluster members among failure rows
    cluster_members = _match_cluster_members(failure_rows, cluster)
    print(f"      cluster_members   : {len(cluster_members)}")

    # Identify TP and FP rows
    tp_rows, fp_rows, n_previously_correct = _identify_tp_fp_rows(
        cluster_members, non_failure_rows, baseline_model, injected_model, device
    )
    print(f"      TP rows (fixed)   : {len(tp_rows)}")
    print(f"      FP rows (flipped) : {len(fp_rows)}")

    if not tp_rows or not fp_rows:
        print(f"  [{symbol_id}] Insufficient TP or FP rows -- skipping (tp={len(tp_rows)}, fp={len(fp_rows)})")
        return None, None

    # Discriminant analysis
    discriminants, tp_means, fp_means = _discriminant_analysis(tp_rows, fp_rows)

    print(f"      Top {top_k} discriminants (by net_benefit):")
    for d in discriminants[:top_k]:
        sym = "+" if d.net_benefit >= 0 else ""
        print(
            f"        {d.predicate:<25s}  {d.condition_type:<7s}"
            f"  delta={d.delta:.3f}"
            f"  tp={d.mu_tp:.3f}  fp={d.mu_fp:.3f}"
            f"  net_benefit={sym}{d.net_benefit:.3f}"
        )

    # Estimate residual FPR after applying top-K conditions
    residual_fpr, conditions = _estimate_residual_fpr(
        fp_rows, discriminants, top_k, n_previously_correct
    )
    print(f"      conditions applied: {conditions}")
    print(f"      residual_fpr est. : {residual_fpr:.4f}  (target < {FPR_TARGET})")

    can_resolve = residual_fpr < FPR_TARGET

    # Build AntecedentRefinement
    refinement = AntecedentRefinement(
        symbol_id                    = symbol_id,
        symbol_name                  = sym_name,
        original_antecedent_description = candidate_rule.get("description", ""),
        candidate_rule_name          = candidate_rule_name,
        original_fpr                 = round(fpr, 4),
        n_tp                         = len(tp_rows),
        n_fp                         = len(fp_rows),
        n_previously_correct         = n_previously_correct,
        top_discriminants            = [asdict(d) for d in discriminants[:top_k]],
        suggested_additional_conditions = conditions,
        estimated_residual_fpr       = residual_fpr,
        can_resolve_with_existing    = can_resolve,
        tp_predicate_means           = tp_means,
        fp_predicate_means           = fp_means,
    )

    # Build PredicateProposal if existing predicates cannot fully resolve the gap
    proposal: Optional[PredicateProposal] = None
    if not can_resolve:
        majority_gold      = registry_sym.get("majority_gold_intent", "")
        majority_predicted = registry_sym.get("majority_predicted_intent", "")
        dominant_confusion = registry_sym.get("dominant_confusion", "")

        proposed_name, rationale, suggested_def = _lookup_predicate_gap(
            majority_gold, majority_predicted
        )

        # Minimum coverage the new predicate must achieve to bring residual FPR < target
        still_fp   = int(residual_fpr * n_previously_correct)
        target_fp  = max(0, int((FPR_TARGET - 0.001) * n_previously_correct))
        must_excl  = max(0, still_fp - target_fp)
        min_coverage = round(must_excl / max(1, still_fp), 4)

        tp_utterances = [r["utterance"] for r in tp_rows if "utterance" in r][:5]
        fp_utterances = [r["utterance"] for r in fp_rows if "utterance" in r][:5]

        print(f"      Existing predicates CANNOT fully resolve gap")
        print(f"        -> PredicateProposal: {proposed_name}")
        print(f"           min_coverage needed: {min_coverage:.1%} of {still_fp} residual FP rows")

        proposal = PredicateProposal(
            proposal_id                  = "",   # assigned by run_evolution
            source_symbol_id             = symbol_id,
            source_cluster_id            = cluster.get("cluster_id", -1),
            majority_gold_intent         = majority_gold,
            majority_predicted_intent    = majority_predicted,
            dominant_confusion           = dominant_confusion,
            candidate_rule_name          = candidate_rule_name,
            proposed_predicate_name      = proposed_name,
            rationale                    = rationale,
            suggested_definition         = suggested_def,
            top_existing_discriminants   = [asdict(d) for d in discriminants[:3]],
            example_tp_utterances        = tp_utterances,
            example_fp_utterances        = fp_utterances,
            estimated_minimum_coverage   = min_coverage,
            residual_fpr_after_existing  = residual_fpr,
        )
    else:
        print(f"      Existing predicates CAN resolve gap -- antecedent tightening sufficient")

    return refinement, proposal


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evolution(
    symbol_ids: Optional[list[str]],
    checkpoint_path: Path,
    registry_path: Path,
    rule_base_path: Path,
    candidate_dir: Path,
    failure_set_path: Path,
    full_inference_path: Path,
    clusters_path: Path,
    report_dir: Path,
    top_k: int,
) -> tuple[list[AntecedentRefinement], list[PredicateProposal]]:
    """
    Run predicate evolution analysis for all (or specified) FPR-failing symbols.
    """
    device = torch.device("cpu")

    print("[PredicateEvolver] Checkpoint      :", checkpoint_path)
    print("[PredicateEvolver] Loading data ...")

    failure_rows     = _load_jsonl(failure_set_path)
    full_rows        = _load_jsonl(full_inference_path)
    non_failure_rows = [r for r in full_rows if not r.get("is_failure", True)]

    print(f"[PredicateEvolver]   failure rows   : {len(failure_rows)}")
    print(f"[PredicateEvolver]   non-failure rows: {len(non_failure_rows)}")

    with open(clusters_path) as f:
        clusters = json.load(f)

    with open(rule_base_path) as f:
        existing_rule_base = json.load(f)

    registry = SymbolRegistry(registry_path)

    # Load baseline model once (4 rules, shared across all symbols)
    print("[PredicateEvolver] Loading baseline model ...")
    baseline_model = _load_baseline_model(checkpoint_path, rule_base_path, device)

    # Determine which symbols to analyse
    all_proposed = [
        (sid, sym) for sid, sym in registry._data["symbols"].items()
        if sym.get("status") == "proposed"
    ]
    if symbol_ids:
        all_proposed = [(sid, sym) for sid, sym in all_proposed if sid in symbol_ids]

    refinements: list[AntecedentRefinement] = []
    proposals:   list[PredicateProposal]    = []

    print(f"\n[PredicateEvolver] Analysing {len(all_proposed)} proposed symbol(s) ...")

    for sid, sym in all_proposed:
        ref, prop = analyze_symbol(
            symbol_id         = sid,
            registry_sym      = sym,
            candidate_dir     = candidate_dir,
            report_dir        = report_dir,
            failure_rows      = failure_rows,
            non_failure_rows  = non_failure_rows,
            clusters          = clusters,
            existing_rule_base= existing_rule_base,
            checkpoint_path   = checkpoint_path,
            baseline_model    = baseline_model,
            device            = device,
            top_k             = top_k,
        )
        if ref is not None:
            refinements.append(ref)
        if prop is not None:
            proposals.append(prop)

    # Assign proposal IDs
    for i, prop in enumerate(proposals, start=1):
        prop.proposal_id = f"PP_{i:03d}"

    return refinements, proposals


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    refinements: list[AntecedentRefinement],
    proposals:   list[PredicateProposal],
    out_dir:     Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path  = out_dir / "antecedent_refinements.json"
    prop_path = out_dir / "predicate_proposals.json"

    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in refinements], f, indent=2)

    with open(prop_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in proposals], f, indent=2)

    return ref_path, prop_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PredicateEvolver -- Task 20: Discriminative Predicate Evolution"
    )
    parser.add_argument("--symbol",       nargs="*", default=None,
                        help="Specific symbol IDs to analyse (default: all Proposed)")
    parser.add_argument("--checkpoint",   type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--registry",     type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--rule-base",    type=Path, default=DEFAULT_RULE_BASE)
    parser.add_argument("--candidate-dir",type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--failure-set",  type=Path, default=DEFAULT_FAILURE_SET)
    parser.add_argument("--full-inference",type=Path,default=DEFAULT_FULL_INFERENCE)
    parser.add_argument("--clusters",     type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--report-dir",   type=Path, default=REPORT_DIR)
    parser.add_argument("--top-k",        type=int,  default=DEFAULT_TOP_K,
                        help="Number of top discriminant conditions to apply")
    args = parser.parse_args()

    refinements, proposals = run_evolution(
        symbol_ids         = args.symbol,
        checkpoint_path    = args.checkpoint,
        registry_path      = args.registry,
        rule_base_path     = args.rule_base,
        candidate_dir      = args.candidate_dir,
        failure_set_path   = args.failure_set,
        full_inference_path= args.full_inference,
        clusters_path      = args.clusters,
        report_dir         = args.report_dir,
        top_k              = args.top_k,
    )

    ref_path, prop_path = save_results(refinements, proposals, DATA_DIR)

    print(f"\n{'='*70}")
    print(f"  PredicateEvolver Summary")
    print(f"{'='*70}")
    print(f"  Symbols analysed       : {len(refinements)}")
    print(f"  Antecedent refinements : {len(refinements)}")
    print(f"  Predicate proposals    : {len(proposals)}")

    if refinements:
        print(f"\n  Antecedent Refinements:")
        for r in refinements:
            resolved = "YES -- existing predicates sufficient" if r.can_resolve_with_existing else "NO  -- new predicate needed"
            print(f"    [{r.symbol_id}]  residual_fpr={r.estimated_residual_fpr:.4f}  "
                  f"can_resolve={resolved}")
            for cond in r.suggested_additional_conditions:
                print(f"              {cond}")

    if proposals:
        print(f"\n  Predicate Proposals:")
        for p in proposals:
            print(f"    [{p.proposal_id}] {p.proposed_predicate_name}")
            print(f"           source      : {p.source_symbol_id} / {p.dominant_confusion}")
            print(f"           min_coverage: {p.estimated_minimum_coverage:.1%} of residual FP rows")
            print(f"           rationale   : {p.rationale[:80]}...")

    print(f"\n  [saved] {ref_path}")
    print(f"  [saved] {prop_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
