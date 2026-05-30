# Level 6 — Self-Evolving Neuro-Symbolic Cognitive Architecture

Level 6 extends the Level 5 rule-compiled network with a **self-modifying symbolic substrate**.
The static, human-authored rule vocabulary of Level 5 is replaced by an evolving symbol registry:
failures in the Level 5 model drive discovery of new predicate-space symbols, which generate
candidate rules, which are validated and promoted into the rule base.
The neural layer (SentenceTransformer + predicate head) remains unchanged between evolution cycles.

## Kautz Typology Position

| Type | Description |
|------|-------------|
| 1 | Standard neural network — no symbolic component |
| 2 | Symbolic rules applied *after* neural inference |
| 3 | Neural network trained on symbolic knowledge |
| 4 | Neuro-symbolic — neural + differentiable symbolic layer |
| **5** | **Neural network compiled from symbolic rules** (Level 5 PoC) |
| **6** | **Neural network that dynamically modifies its own symbolic substrate** ← Level 6 |

Kautz's Type 6 describes neural systems that *dynamically use symbolic reasoning internally*.
The standard is not fully formalised at the implementation level.
This PoC implements a practical extension toward that direction: a **Type 5 architecture with
a self-modifying symbolic substrate** scoped to *Symbol Vocabulary Evolution* only.

## Evolution Loop

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Level 6 Closed Loop                           │
  │                                                                      │
  │  L5 inference (frozen encoder + predicate head)                      │
  │       │                                                              │
  │       ▼                                                              │
  │  FailureCollector  ──►  predicate_probs[N, 11]  (failure_set.jsonl)  │
  │       │                                                              │
  │       ▼                                                              │
  │  SymbolCluster  (HDBSCAN, cosine metric, 11-dim predicate space)     │
  │       │   births new SYM_ symbols with auto-derived names            │
  │       ▼                                                              │
  │  SymbolRegistry  (lifecycle state machine, JSON-backed)              │
  │       │   proposed → experimental → active → weakening → deprecated  │
  │       ▼                                                              │
  │  RuleCandidateGen  ──►  candidate_rules/*.json                       │
  │       │   product t-norm antecedent trees, RuleCompiler-compatible   │
  │       ▼                                                              │
  │  RuleValidator  ──►  accuracy_delta, FPR  (no-retrain gate)          │
  │       │                                                              │
  │  RetrainValidator  ──►  retrain_delta  (fine-tune gate)              │
  │       │                                                              │
  │       ▼                                                              │
  │  rule_base.json  updated  ──►  next L5 inference cycle               │
  └──────────────────────────────────────────────────────────────────────┘
```

## ReasoningState

`ReasoningState` is the central data structure that flows through all Level 6 components.
It wraps the four output tensors from a Level 5 forward pass:

| Field | Shape | Description |
|-------|-------|-------------|
| `trunk_repr` | `[256]` | Shared trunk activation (Linear 384→256 + ReLU) |
| `predicate_probs` | `[11]` | Sigmoid predicate head — named symbolic grounding |
| `rule_activations` | `[4]` | DifferentiableRuleLayer outputs (R1..R4) |
| `intent_dist` | `[4]` | Softmax intent probabilities |
| `max_confidence` | `float` | Max intent probability |
| `predicted_intent` | `str` | Argmax intent label |
| `is_failure` | `bool` | Misclassification OR low-confidence |
| `gold_intent` | `str` | Ground-truth label (if available) |

The **11-dimensional `predicate_probs`** is the Level 6 clustering substrate.
Because each dimension has a human-readable name, cluster centroids are directly
interpretable without an LLM.

**Predicate index map** (must stay in sync with `level5/model/dataset.py`):

| Index | Predicate |
|-------|-----------|
| 0 | `is_infrastructure` |
| 1 | `is_service` |
| 2 | `is_metric` |
| 3 | `is_incident` |
| 4 | `is_job` |
| 5 | `is_pipeline` |
| 6 | `is_unknown` |
| 7 | `is_sre_domain` |
| 8 | `has_runbook` |
| 9 | `is_known_incident` |
| 10 | `is_metric_query` |

## Symbol Lifecycle

```
  proposed ──► experimental ──► active ──► weakening ──► deprecated ──► (GC)
```

| Transition | Criteria |
|-----------|----------|
| `proposed → experimental` | coverage ≥ 10 failures AND cohesion ≥ 0.65 AND acc_delta ≥ 0.03 AND FPR < 0.10 |
| `experimental → active` | retrain_delta ≥ 0.01 (fine-tune proves neural co-evolution) |
| `active → weakening` | acc_delta < 0.01 for 2 consecutive evaluation cycles |
| `weakening → deprecated` | weakening for ≥ 3 consecutive cycles |
| `deprecated → GC` | deprecated for ≥ 5 cycles |

**Grounding quality** = `cohesion × (n_stable / 11)` where *stable* predicates have centroid ≤ 0.30 or ≥ 0.70.
Symbols with grounding_quality < 0.50 are flagged `low_confidence_grounding`.

## File Structure

```
level6/
├── __init__.py                  # Module docstring + Kautz typology summary
├── reasoning_state.py           # ReasoningState dataclass, PREDICATE_COLS, INTENT_LABELS
├── lifecycle.py                 # SymbolStatus enum + all promotion/deprecation criteria
├── failure_collector.py         # FailureCollector — runs L5 inference, flags failures
├── symbol_cluster.py            # SymbolCluster — HDBSCAN on predicate_probs[11]
├── symbol_registry.py           # SymbolRegistry — lifecycle state machine (JSON-backed)
├── rule_candidate_gen.py        # RuleCandidateGen — symbol → candidate rule JSON
├── rule_validator.py            # RuleValidator — no-retrain accuracy_delta + FPR
├── retrain_validator.py         # RetrainValidator — fine-tune retrain_delta validation
├── predicate_evolver.py         # PredicateEvolver — mines discriminative predicates from FP/TP
├── antecedent_refiner.py        # AntecedentRefiner — tightens antecedents from evolver output
├── evolution_engine.py          # EvolutionEngine — orchestrates one full 12-step cycle
├── multi_cycle_runner.py        # MultiCycleRunner — run N cycles, print lifecycle table
├── build_seed_dataset.py        # Builds level6_seed.csv (1,961 rows) from L5 dataset + hard cases
├── level6_self_evolving_nsai.ipynb  # End-to-end walkthrough notebook
├── TODO.md                      # Implementation task list (all tasks tracked)
├── data/
│   ├── level6_seed.csv          # 1,961-row labeled dataset (L5 + 300 hard seed utterances)
│   ├── failure_set.jsonl        # 1,050 failure ReasoningStates (predicate_probs as list[11])
│   ├── full_inference.jsonl     # 1,961 full inference records
│   ├── failure_summary.json     # Aggregate failure stats + predicate profile
│   ├── clusters.json            # HDBSCAN clusters (4 entries including noise)
│   ├── antecedent_refinements.json  # AntecedentRefinement records (2)
│   ├── predicate_proposals.json     # PredicateProposal records (0 — existing vocab sufficient)
│   ├── symbol_registry.json     # Persistent symbol registry (cycle=5 after demo)
│   ├── candidate_rules/         # Candidate rule JSON files + _manifest.json
│   └── validation_reports/      # Per-symbol validation reports (report_S_001/002/003.json)
└── evaluation/
    ├── __init__.py
    ├── evolution_metrics.py         # Per-cycle metrics (symbol counts, coverage, delta dist)
    ├── experiment_a_baseline.py     # Exp A — L5 failure rate + predicate diversity
    ├── experiment_b_evolution_cycle.py  # Exp B — cluster/rule structural validation
    └── experiment_c_validate_rules.py   # Exp C — accuracy_delta two-gate validation
```

## Results

### Level 5 Baseline (level6_seed.csv)

| Metric | Value |
|--------|-------|
| Total samples | 1,961 |
| Intent accuracy | 62.1% |
| Total failures | 1,050 (53.5%) |
| Misclassifications | 743 |
| Low-confidence | 935 |
| Dominant failure predicate | `is_sre_domain` (mean=0.438 in failure set) |

### Symbol Discovery

| Symbol ID | Name | Cluster Size | Cohesion | GQ | Dominant Confusion |
|-----------|------|-------------|---------|-----|-------------------|
| S_001 | `SYM_sre_domain__investigate_vs_summarization` | 654 | 0.976 | 0.799 | `investigate → summarization` |
| S_002 | `SYM_has_runbook__execution_vs_out_of_scope` | 30 | 0.997 | 0.725 | `execution → out_of_scope` |
| S_003 | `SYM_unknown__out_of_scope_vs_summarization` | 20 | 0.997 | 0.907 | `out_of_scope → summarization` |

S_002 was flagged **redundant** — its antecedent (`has_runbook`) is a subset of the existing
`R2_runbook_execution` rule, demonstrating the system correctly identifies and avoids rule duplication.

### Candidate Rule Validation

| Symbol | Δacc (no-retrain) | FPR | Δretrain | Eligible |
|--------|------------------|-----|----------|---------|
| S_001 | +0.2401 | 0.0362 | +0.0255 | ✓ |
| S_002 | −0.1310 | 0.0000 | — | ✗ (redundant) |
| S_003 | +0.2538 | 0.0000 | +0.2585 | ✓ |

Refined rules (`R_S_001_refined`, `R_S_003_refined`) were generated by `AntecedentRefiner`
after `PredicateEvolver` identified that `has_runbook` and `is_unknown` magnitude were the
key discriminants separating true-positives from false-positives.

### Level 5 vs Level 6 Comparison

| Metric | Level 5 | Level 6 (S_001 injected) |
|--------|---------|--------------------------|
| Intent accuracy | 62.1% | ~73.5% (estimated) |
| Failure coverage by active rules | 0 / 1,050 | 654 / 1,050 (62.3%) |
| Active symbols | 0 | 2 (S_001, S_003 at peak) |
| Rule base size | 4 rules | 6 rules (+2 evolved) |
| Symbolic evolution cycles | 0 (static) | 5 cycles demonstrated |
| Symbol lifecycle | N/A | Active → Weakening → Deprecated |

### 5-Cycle Lifecycle Demo

Running `python -m level6.multi_cycle_runner --cycles 5 --simulate-weakening` demonstrated
the full symbol lifecycle:

| Cycle | Event |
|-------|-------|
| 1 | S_001, S_003 promoted `proposed → active` |
| 2 | `active → weakening` (simulated acc_delta < 0.01) |
| 4 | `weakening → deprecated` (3 consecutive weak cycles) |
| 5 | dep_age=1 (approaching GC threshold of 5) |

## Reproduce Commands

```bash
# Run from repo root with venv active

# 1. Run Level 5 inference on seed data and collect failures
python -m level6.failure_collector \
    --checkpoint level5/saved_models/exp_b_l5_main/best_model.pt \
    --data level6/data/level6_seed.csv

# 2. Cluster failures in predicate space
python -m level6.symbol_cluster

# 3. Register born symbols
python -m level6.symbol_registry --register level6/data/clusters.json

# 4. Generate candidate rules
python -m level6.rule_candidate_gen

# 5. Validate candidates (no-retrain gate)
python -m level6.rule_validator

# 6. Run predicate evolution (FPR discrimination)
python -m level6.predicate_evolver

# 7. Refine antecedents and re-validate
python -m level6.antecedent_refiner

# 8. Run fine-tune validation (retrain gate)
python -m level6.retrain_validator

# 9. Run one full evolution cycle (all steps 1-8 orchestrated)
python -m level6.evolution_engine \
    --checkpoint level5/saved_models/exp_b_l5_main/best_model.pt \
    --data level6/data/level6_seed.csv

# 10. Run multi-cycle lifecycle demo (5 cycles, simulate weakening)
python -m level6.multi_cycle_runner --cycles 5 --simulate-weakening

# 11. Run evaluation suite
python -m level6.evaluation.experiment_a_baseline
python -m level6.evaluation.experiment_b_evolution_cycle
python -m level6.evaluation.experiment_c_validate_rules
python -m level6.evaluation.evolution_metrics

# 12. Open the end-to-end notebook
jupyter notebook level6/level6_self_evolving_nsai.ipynb
```

## Out of Scope (This PoC)

This PoC is scoped to **Symbol Vocabulary Evolution** only.
The following are explicitly out of scope and left as open research directions:

- **Symbol operator evolution** — evolving the logic operators (AND/OR/NOT) themselves
- **Reasoning strategy evolution** — changing which predicates are used in the shared trunk
- **Differentiable symbolic transforms** — gradient-based rule structure search
- **Ontology relationship evolution** — adding new predicate dimensions beyond the existing 11
- **Online multi-epoch evolution** — continuous learning without a discrete cycle boundary

## Open Research Problems

1. **Predicate-absence vs uncertainty** — HDBSCAN clusters in the moderate-activation zone
   (0.30–0.70 centroid band) represent cases where the predicate is *uncertain*, not absent.
   The current Mode 2 naming (dominant confusion pair) handles this pragmatically, but a
   formal treatment of fuzzy symbolic grounding is an open problem.

2. **Convergence guarantees** — the current lifecycle criteria (weakening/deprecation
   thresholds) are heuristic. There is no formal proof that the symbol vocabulary converges
   to a stable fixed point or that it avoids oscillation between active and deprecated states.

3. **Stable multi-cycle evolution with real data** — the 5-cycle demo uses
   `--simulate-weakening` to inject synthetic degradation. A full evaluation requires
   running on a stream of genuinely evolving SRE incident data across real calendar cycles.

4. **Rule interaction effects** — the current validation gates (acc_delta, FPR) measure
   each candidate rule in isolation. Rules can interact in the RuleCompiler (product t-norm
   scores are summed before blending with trunk logits), and multi-rule interaction effects
   are not measured.
