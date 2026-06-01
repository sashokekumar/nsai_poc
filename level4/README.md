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

## Experiments & Results (v1 constraints)

> **Note:** The results below reflect the v1 constraint set (10 pairs, TYPE_A/B/C only).  
> Retraining with v2 constraints (λ=1.0, γ=0.75) is expected to show further reduction  
> in overall violation rate, with measurable TYPE_D and TYPE_F rates dropping toward 0.

### Comparison table (test set, n=333)

| Run | Intent Acc | Entity Acc | Domain Acc | Viol Rate | TYPE_A FR | TYPE_B FE | TYPE_C US |
|---|---|---|---|---|---|---|---|
| Level 3.5 (runtime symbolic) | 0.931 | 0.372 | 0.288 | **0.643** | 0.000 | 0.000 | 0.643 |
| Level 4 λ=0.0 (baseline neural) | 0.970 | 0.712 | 0.994 | 0.051 | 0.000 | 0.003 | 0.075 |
| Level 4 λ=0.1 | 0.970 | 0.712 | 0.994 | 0.045 | 0.000 | 0.000 | 0.069 |
| Level 4 λ=0.25 | 0.967 | 0.613 | 0.991 | 0.054 | 0.000 | 0.000 | 0.081 |
| Level 4 λ=0.5 | 0.970 | 0.622 | 0.991 | 0.048 | 0.000 | 0.000 | 0.075 |
| **Level 4 λ=1.0** ✓ | **0.967** | 0.682 | 0.994 | **0.036** | 0.000 | 0.000 | **0.066** |
| Level 4 λ=2.0 | 0.961 | 0.682 | 0.991 | **0.021** | 0.000 | 0.000 | **0.051** |

Abbreviations: FR = false rejection, FE = false execution, US = ungrounded SRE.

### Key findings

1. **Level 3.5 comparison caveat** — the 64.3% TYPE_C rate reflects schema mismatch, not a failure of the symbolic approach itself.

2. **Baseline neural (λ=0) already far better** — 97.0% intent accuracy, 5.1% violation rate. The frozen encoder provides strong semantic features without any symbolic signal.

3. **TYPE_B false execution eliminated at λ≥0.1** — the constraint loss learned `execution + metric` and `execution + incident` are invalid with just one disallowed-pair penalty.

4. **Sweet spot: λ=1.0** — violation rate drops to 3.6% (−30% vs baseline), intent accuracy preserved at 96.7%. No accuracy–violation tradeoff at this scale.

5. **v2 expected gains** — with TYPE_D (execution+unknown, w=1.25) and TYPE_E (causal cross-head, γ=0.75) active, the most dangerous unpenalised combinations should be suppressed. TYPE_F (summarization+metric, w=0.4) addresses the investigate↔summarization confusion observed in Level 6 failure analysis.

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

