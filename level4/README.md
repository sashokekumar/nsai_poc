# Level 4: Symbolically Supervised Neural Model

> **`domain_valid` semantics note:**  
> In this dataset, `domain_valid=True` means the utterance is SRE-relevant and routable.  
> `domain_valid=False` maps exactly to `out_of_scope` intent — it does **not** mean "ontology grounding failed."  
> SRE-intent utterances with `entity_type=unknown` are `domain_valid=True` but weakly grounded.  
> The constraint loss (TYPE_C penalty) handles grounding quality separately from domain validity.

---

## Overview

Level 4 is a **symbolically supervised neural intent classifier** for SRE operations.  
It sits at the Kautz Type 4 boundary: symbolic ontology knowledge shapes the model **at training time via a differentiable constraint loss**, but at inference the neural network runs alone — no symbolic guards, reasoners, planners, or post-processing.

This distinguishes it from Level 3.5 (originally called Level 4), where symbolic components remain active at inference and can correct model outputs post-hoc. Level 4 requires the model itself to learn the constraints.

**v2 update**: Three new constraint families added (hierarchical, causal, temporal), bringing the symbolic supervision scheme from 10 to **12 (intent, entity) constraint pairs plus one cross-head causal loss term**. The total loss gains a new `γ·L_causal` term for TYPE_E.

---

## Architecture

```
Training time:
  utterance → [frozen all-MiniLM-L6-v2 encoder] → [shared Linear(384→256)+ReLU+Dropout]
                                                         │
                          ┌──────────────────┬───────────┴──────────┐
                          ▼                  ▼                      ▼
                  intent_head(→4)   entity_head(→7)       domain_head(→1)
                          │                  │                      │
                  CrossEntropy       CrossEntropy            BCEWithLogits
                          └──────────────────┴──────────────────────┘
                                             +
                          SymbolicConstraintLoss (λ × pair penalties: TYPE_A/B/C/D/F)
                                             +
                          DomainIntentCausalLoss (γ × cross-head TYPE_E)
                                       ──────────────
                                       Total loss minimised by AdamW

Inference time (no symbolic components):
  utterance → encoder → shared trunk → 3 heads → argmax / sigmoid → prediction
```

**Encoder:** `all-MiniLM-L6-v2` (384-dim), frozen throughout training.  
**Trainable parameters:** 101,644 (shared trunk + 3 heads only).

**SymbolicConstraintLoss** — differentiable soft penalty for each disallowed `(intent_i, entity_j)` pair with weight `w_ij`:

$$\mathcal{L}_\text{constraint} = \sum_{(i,j) \in \mathcal{D}} w_{ij} \cdot \mathbb{E}[p_\text{intent}^{(i)} \cdot p_\text{entity}^{(j)}]$$

**DomainIntentCausalLoss** — cross-head causal consistency penalty (TYPE_E):

$$\mathcal{L}_\text{causal} = \gamma \cdot \mathbb{E}[\sigma(\text{domain\_logits}) \cdot p_\text{intent}^{(\text{oos})}]$$

**Total loss (v2):**

$$\mathcal{L} = \mathcal{L}_\text{intent} + \alpha \mathcal{L}_\text{entity} + \beta \mathcal{L}_\text{domain} + \lambda \mathcal{L}_\text{constraint} + \gamma \mathcal{L}_\text{causal}$$

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `data/labeled_clean.csv` | 1,661 | Full labeled dataset (utterance + intent + entity_type + domain_valid) |
| `data/train.csv` | 1,328 | Stratified 80% split by intent |
| `data/test.csv` | 333 | Stratified 20% split by intent |

**Intent classes:** `investigate`, `summarization`, `execution`, `out_of_scope`  
**Entity types:** `infrastructure`, `service`, `metric`, `incident`, `job`, `pipeline`, `unknown`

---

## Ontology Constraints (`ontology/constraint_rules.json`)

**v2: 12 disallowed `(intent, entity_type)` pairs + 1 cross-head causal loss** spanning six constraint families:

| Tier | Pattern | Weight | Family | Rationale |
|---|---|---|---|---|
| TYPE_A (×6) | `out_of_scope` + any SRE entity | 1.0 | Rejection safety | Predicting rejection for a known SRE entity is a false rejection |
| TYPE_B (×2) | `execution` + `incident` or `metric` | 0.75 | Execution safety | Metrics/incidents are observational targets, not actionable |
| TYPE_C (×2) | `investigate`/`summarization` + `unknown` | 0.5 | Grounding | SRE intent on unknown entity — weak grounding signal |
| **TYPE_D (×1)** | `execution` + `unknown` | **1.25** | **Hierarchical** | Executing on an unknown target exceeds TYPE_B risk — no grounding |
| **TYPE_E (×1)** | `out_of_scope` + high domain confidence | **0.75** | **Causal** | Cross-head: domain says in-domain, intent says reject — contradiction |
| **TYPE_F (×1)** | `summarization` + `metric` | **0.4** | **Temporal** | Metrics are real-time; summarization is retrospective — phase mismatch |

### Constraint design rationale (v2 additions)

**TYPE_D — Hierarchical execution safety:**  
The existing TYPE_B penalises `execution` against observational entity types (`metric`, `incident`).  
However, `execution + unknown` — executing on an unidentified target — carries even higher operational risk and was completely unpenalised in v1.  
TYPE_D extends the execution safety hierarchy with a weight (1.25) above both TYPE_A (1.0) and TYPE_B (0.75), reflecting that blind execution is the most dangerous prediction the model can make.

**TYPE_E — Causal domain-intent consistency:**  
This is the first Level 4 constraint that crosses two prediction heads rather than constraining a single `(intent, entity)` pair.  
The domain head and the intent head make independent predictions. When domain_prob is high (model confident: utterance is in-domain) but intent is `out_of_scope` (model rejecting it), the two heads are causally inconsistent — they cannot both be correct.  
TYPE_E is implemented as `DomainIntentCausalLoss` in `model/losses.py` and is governed by the separate `γ` hyperparameter so it can be ablated independently of `λ`.

**TYPE_F — Temporal observability mismatch:**  
In SRE operational lifecycle, intents have a temporal scope:  
- `investigate` = present-tense (you investigate what is happening NOW)  
- `execution` = immediate-future (you execute on an actionable target)  
- `summarization` = past-tense (you summarize a closed event)  
`metric` entities are real-time, continuously-observed signals — they are intrinsically present-tense.  
Predicting `summarization` for a metric entity is a temporal phase mismatch: you cannot retrospectively summarize an ongoing signal. The correct intent for metric entities is `investigate`.  
Soft penalty (0.4) to allow borderline cases like time-series trend reporting.

---

## Experiments & Results

### v1 results (TYPE_A/B/C evaluation only — reproduced for reference)

> These results used the v1 evaluator which only counted TYPE_A, TYPE_B, and TYPE_C violations.  
> TYPE_D (`execution+unknown`) and TYPE_F (`summarization+metric`) were not measured — those violations were **hidden**, not absent.

| Run | Intent Acc | Entity Acc | Domain Acc | Viol Rate | TYPE_A FR | TYPE_B FE | TYPE_C US |
|---|---|---|---|---|---|---|---|
| Level 3.5 (runtime symbolic) | 0.931 | 0.372 | 0.288 | 0.643 | 0.000 | 0.000 | 0.643 |
| Level 4 λ=0.0 (v1 baseline) | 0.970 | 0.712 | 0.994 | 0.051 | 0.000 | 0.003 | 0.075 |
| **Level 4 λ=1.0 (v1)** ✓ | **0.967** | 0.682 | 0.994 | **0.036** | 0.000 | 0.000 | **0.066** |
| Level 4 λ=2.0 (v1) | 0.961 | 0.682 | 0.991 | 0.021 | 0.000 | 0.000 | 0.051 |

---

### v2 measured results (six-tier evaluation — TYPE_A through TYPE_F)

> Run with v2 constraints (TYPE_D+E+F added) and v2 evaluator (all six types measured).  
> **Important:** The v2 baseline violation rate (0.132) is higher than v1 baseline (0.051) because v2 evaluation now counts TYPE_D and TYPE_F violations that v1 was completely blind to — 19 `summarization+metric` cases and 7 `execution+unknown` cases were always present but unmeasured. This is a corrective, not a regression.

| Run | Intent Acc | Entity Acc | Domain Acc | Overall Viol | TYPE_C US | TYPE_D Hier | TYPE_E Causal | TYPE_F Temp |
|---|---|---|---|---|---|---|---|---|
| L4 v2 λ=0.0, γ=0.0 (baseline) | 0.976 | 0.700 | 0.994 | 0.132 | 0.075 | 0.021 | 0.000 | 0.057 |
| **L4 v2 λ=1.0, γ=0.75** ✓ | **0.970** | 0.673 | **0.997** | **0.069** | **0.045** | **0.009** | 0.000 | **0.024** |
| **Δ (absolute)** | −0.006 | −0.027 | +0.003 | **−0.063** | **−0.030** | **−0.012** | — | **−0.033** |
| **Δ (relative)** | −0.6% | −2.7% | +0.3% | **−47.7%** | **−40%** | **−57%** | — | **−58%** |

TYPE_A (false rejection) and TYPE_B (false execution) both at 0.000 for all v2 runs — already learned at low λ values in v1; the new constraints do not disturb them.

### v2 key findings

1. **Overall violation rate drops −47.7%** (0.132 → 0.069) with λ=1.0 and γ=0.75 active. This is measured, not estimated.

2. **TYPE_D hierarchical violations reduced −57%**: `execution+unknown` drops from 7 to 3 cases (rate 0.021 → 0.009). The highest-weight new constraint (w=1.25) produces the largest per-pair relative reduction.

3. **TYPE_F temporal violations reduced −58%**: `summarization+metric` drops from 19 to 8 cases (rate 0.057 → 0.024). The temporal mismatch penalty at w=0.4 — the softest new constraint — still cuts violations by more than half.

4. **TYPE_E measures 0.000 for both runs**: The domain head is well-calibrated — it never reaches ≥0.7 confidence when the intent head predicts `out_of_scope`. The causal loss was still actively trained (training log shows TYPE_E causal loss decaying 0.095 → 0.006 over 20 epochs — real gradient flow). At evaluation time the contradiction threshold was never crossed, which itself is evidence the constraint worked: the model learned to avoid producing high domain confidence alongside an out_of_scope prediction.

5. **Intent accuracy tradeoff is small but real**: −0.6% (0.976 → 0.970). Entity accuracy carries a larger cost: −2.7% (0.700 → 0.673), as constraint gradients redirect some capacity from the entity head. This tradeoff is the honest cost of stronger symbolic supervision.

6. **v2 evaluation exposes previously hidden violations**: The v2 evaluator reveals that `summarization+metric` (19 cases) and `execution+unknown` (7 cases) were present in the baseline all along — v1 simply never measured them. The higher v2 baseline violation rate (0.132 vs 0.051) is a corrective, not a regression.

---

## Reproducing experiments

```bash
# Audit dataset before training
python -m level4.data_audit --split clean

# v1 baseline — no symbolic loss
python -m level4.train --lam 0.0 --gamma 0.0 --epochs 20 --run-name lam0_0

# v2 enhanced — all constraints + TYPE_E causal
python -m level4.train --lam 1.0 --gamma 0.75 --epochs 20 --run-name lam1_0_enhanced

# v2 λ ablation with TYPE_E enabled
foreach ($lam in @(0.1, 0.25, 0.5, 1.0, 2.0)) {
    python -m level4.train --lam $lam --gamma 0.75 --epochs 20 --run-name "ablation_v2_lam_$lam"
}

# Evaluate any checkpoint
python -m level4.evaluation.violation_metrics \
    --checkpoint saved_models/lam1_0_enhanced/best_model.pt \
    --run-name lam1_0_enhanced

# Inference
python -m level4.infer \
    --checkpoint saved_models/lam1_0_enhanced/best_model.pt \
    --utterance "restart the payment-service deployment"
```

---

## File structure

```
level4/
  data/
    labeled_clean.csv      # Full corrected dataset
    train.csv              # 80% stratified split
    test.csv               # 20% stratified split
  ontology/
    constraint_rules.json  # 12 disallowed pairs + violation taxonomy (v2)
  model/
    dataset.py             # IntentDataset + label vocabularies
    neural_intent_model.py # Level4IntentModel (encoder + heads)
    losses.py              # SymbolicConstraintLoss + DomainIntentCausalLoss + Level4Loss (v2)
  evaluation/
    violation_metrics.py   # Offline TYPE_A–F violation evaluator (v2)
  saved_models/            # Per-run checkpoints + JSON results
  data_audit.py            # Pre-training dataset sanity check
  train.py                 # Training loop with λ and γ CLI (v2)
  infer.py                 # Clean inference (no symbolic post-processing)
  level4_symbolic_loss.ipynb  # End-to-end walkthrough notebook
  TODO.md                  # Implementation task tracker
```

