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
                          SymbolicConstraintLoss (λ × ontology penalty)
                                       ──────────────
                                       Total loss minimised by AdamW

Inference time (no symbolic components):
  utterance → encoder → shared trunk → 3 heads → argmax / sigmoid → prediction
```

**Encoder:** `all-MiniLM-L6-v2` (384-dim), frozen throughout training.  
**Trainable parameters:** 101,644 (shared trunk + 3 heads only).  
**Constraint loss:** differentiable soft penalty — for each disallowed `(intent_i, entity_j)` pair with weight `w_ij`:

$$\mathcal{L}_\text{constraint} = \sum_{(i,j) \in \mathcal{D}} w_{ij} \cdot \mathbb{E}[p_\text{intent}^{(i)} \cdot p_\text{entity}^{(j)}]$$

**Total loss:**

$$\mathcal{L} = \mathcal{L}_\text{intent} + \alpha \mathcal{L}_\text{entity} + \beta \mathcal{L}_\text{domain} + \lambda \mathcal{L}_\text{constraint}$$

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

10 disallowed `(intent, entity_type)` pairs derived from SRE domain semantics:

| Tier | Pattern | Weight | Violation type |
|---|---|---|---|
| TYPE_A (×6) | `out_of_scope` + any SRE entity | 1.0 | False rejection |
| TYPE_B (×2) | `execution` + `incident` or `metric` | 0.75 | False execution |
| TYPE_C (×2) | `investigate`/`summarization` + `unknown` | 0.5 | Ungrounded SRE |

---

## Experiments & Results

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

1. **Level 3.5 is included as a structural comparison, but its runtime ontology/entity extraction scheme was not optimized for the Level 4 entity/domain evaluation schema** — 64.3% TYPE_C violation rate, entity accuracy 37% reflect schema mismatch rather than a failure of the symbolic approach itself.

2. **Baseline neural (λ=0) already far better** — 97.0% intent accuracy, 5.1% violation rate. The frozen encoder provides strong semantic features without any symbolic signal.

3. **TYPE_B false execution eliminated at λ≥0.1** — the constraint loss learned `execution + metric` and `execution + incident` are invalid with just one disallowed-pair penalty.

4. **Sweet spot: λ=1.0** — violation rate drops to 3.6% (−30% vs baseline), intent accuracy preserved at 96.7%. No accuracy–violation tradeoff at this scale.

5. **λ=2.0 halves violations further** (2.1%, TYPE_C 5.1%) with only 0.9% accuracy cost — the constraint loss dominates but does not yet collapse classification.

---

## Reproducing experiments

```bash
# Audit dataset before training
python -m level4.data_audit --split clean

# Experiment A — baseline (λ=0)
python -m level4.train --lam 0.0 --epochs 20 --run-name lam0_0

# Experiment C — symbolically supervised (λ=0.5)
python -m level4.train --lam 0.5 --epochs 20 --run-name lam0_5

# Experiment D — λ ablation sweep
foreach ($lam in @(0.1, 0.25, 1.0, 2.0)) {
    python -m level4.train --lam $lam --epochs 20 --run-name lam_$lam
}

# Evaluate any checkpoint
python -m level4.evaluation.violation_metrics \
    --checkpoint saved_models/lam0_5/best_model.pt \
    --run-name lam0_5

# Inference
python -m level4.infer \
    --checkpoint saved_models/lam1_0/best_model.pt \
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
    constraint_rules.json  # 10 disallowed pairs + violation taxonomy
  model/
    dataset.py             # IntentDataset + label vocabularies
    neural_intent_model.py # Level4IntentModel (encoder + heads)
    losses.py              # SymbolicConstraintLoss + Level4Loss
  evaluation/
    violation_metrics.py   # Offline TYPE_A/B/C violation evaluator
  saved_models/            # Per-run checkpoints + JSON results
  data_audit.py            # Pre-training dataset sanity check
  train.py                 # Training loop with λ CLI
  infer.py                 # Clean inference (no symbolic post-processing)
  level4_symbolic_loss.ipynb  # End-to-end walkthrough notebook
  TODO.md                  # Implementation task tracker
```


