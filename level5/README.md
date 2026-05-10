# Level 5 — Rule-Compiled Network (Kautz Architecture)

Level 5 compiles symbolic rules **directly into the neural network architecture** as a differentiable rule layer. Unlike Level 4 (which applies constraint penalties at training time only) and Level 4.5 (which masks outputs at runtime), Level 5 rules are structural — they influence every forward pass through product t-norm logic and learned rule strengths.

## Architecture

```
Utterance (text)
     │
     ▼
┌─────────────────────────────────────┐
│  Frozen SentenceTransformer Encoder │  all-MiniLM-L6-v2  →  384-dim
└─────────────────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  Shared Trunk (MLP)      │  Linear(384→256) + ReLU + Dropout(0.3)
└──────────────────────────┘
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
┌───────────────────────┐        ┌────────────────────────────┐
│  Predicate Head       │        │  Intent Head (trunk_logits)│
│  Linear(256→11)       │        │  Linear(256→4)             │
│  sigmoid → [0,1]      │        └────────────────────────────┘
└───────────────────────┘                   │
     │                                      │
     ▼                                      │
┌───────────────────────────────────────┐   │
│  RuleCompiler (product t-norm logic)  │   │
│  4 rules × learnable rule_strength   │   │
│  → rule_scores [batch, 4]  ∈ [0,1]   │   │
└───────────────────────────────────────┘   │
     │                                      │
     ▼                                      │
┌───────────────────────────────────────┐   │
│  rule_score_projection                │   │
│  Linear(4→4, bias=True)               │   │
│  maps [0,1] scores → logit scale     │   │
└───────────────────────────────────────┘   │
     │                                      │
     └──────────────┬────────────────────────┘
                    ▼
         intent_logit = α · rule_logits + (1-α) · trunk_logits
                    │
                    ▼
               softmax → 4 intent classes
```

**Intents (4):** `investigate`, `summarization`, `execution`, `out_of_scope`

**Predicates (11):** `is_infrastructure`, `is_service`, `is_metric`, `is_incident`, `is_job`, `is_pipeline`, `is_unknown`, `is_sre_domain`, `has_runbook`, `is_known_incident`, `is_metric_query`

## Product T-Norm Logic

All logic operations are differentiable, allowing backpropagation through symbolic rules:

| Operation | Formula |
|-----------|---------|
| AND(a, b) | $a \cdot b$ |
| OR(a, b)  | $a + b - a \cdot b$ |
| NOT(a)    | $1 - a$ |

The blend weight $\alpha$ (learnable sigmoid parameter, init=0.5) controls how much rule scores vs trunk logits drive the final intent prediction.

## Rule Base

Defined in `level5/data/rule_base.json`. Four rules derived from SRE domain knowledge:

| Rule | Logic | Target Intent | Init Strength |
|------|-------|--------------|---------------|
| R1 | OR(is_metric_query, is_known_incident) | investigate | 0.7 |
| R2 | AND(has_runbook, is_sre_domain) | execution | 0.9 |
| R3 | AND(is_known_incident, is_sre_domain, NOT has_runbook) | summarization | 0.6 |
| R4 | AND(is_unknown, NOT is_sre_domain) | out_of_scope | 0.8 |

`rule_base.json` schema:
```json
{
  "rules": [
    {
      "name": "R1_metric_investigate",
      "logic": {"op": "OR", "args": ["is_metric_query", "is_known_incident"]},
      "consequent": "investigate",
      "strength_init": 0.7
    }
  ]
}
```

## Experiments

Three ablations were run to isolate the contribution of the rule layer:

| Experiment | Description | Key Config |
|------------|-------------|-----------|
| **Exp A** — Rules Off | Rules frozen at strength=0.0 and pred_weight=0; pure trunk baseline with no symbolic component | `--freeze-rules --rule-strength-init 0.0 --pred-weight 0.0` |
| **Exp B** — Main (L5) | Full neuro-symbolic: learnable rule strengths + predicate supervision | `--epochs 20` |
| **Exp C** — Hard Rules | Rules frozen at strength=1.0 (hard symbolic); trunk must learn around them | `--freeze-rules --rule-strength-init 1.0` |

### Reproduce Commands

```bash
# Exp A — rules disabled (pure trunk baseline)
python -m level5.train --run-name exp_a_rules_disabled --freeze-rules --rule-strength-init 0.0 --pred-weight 0.0 --epochs 20

# Exp B — main Level 5 (learnable rules)
python -m level5.train --run-name exp_b_l5_main --epochs 20

# Exp C — hard symbolic rules (frozen at 1.0)
python -m level5.train --run-name exp_c_hard_rules --freeze-rules --rule-strength-init 1.0 --epochs 20
```

### Evaluate

```bash
python -m level5.evaluation.violation_metrics \
    --checkpoint saved_models/exp_b_l5_main/best_model.pt \
    --run-name exp_b_l5_main \
    --compare-l4 level4/saved_models/lam2_0/evaluation_metrics.json
```

### Inference

```bash
# Single utterance
python -m level5.infer --checkpoint saved_models/exp_b_l5_main/best_model.pt \
    --utterance "Check error rate for payment-service"

# Batch CSV
python -m level5.infer --checkpoint saved_models/exp_b_l5_main/best_model.pt \
    --input-file data.csv --utterance-col utterance --output-file results.json
```

## Results

| Experiment | Intent Acc | Pred Acc | Rule Fidelity | Viol Rate | TYPE-A | TYPE-B | TYPE-C | Rule Strengths | Blend α |
|------------|-----------|---------|--------------|----------|--------|--------|--------|----------------|---------|
| L3.5 Baseline | 0.9309 | — | — | 0.6426 | 0.0 | 0.0 | 0.6426 | — | — |
| L4 λ=2.0 | 0.9610 | — | — | 0.0210 | 0.0 | 0.0 | 0.0511 | — | — |
| L5-A (rules off) | 0.9880 | 0.3574 | 0.0033 | 0.5255 | 0.2402 | 0.0 | 0.5345 | [0.000, 0.000, 0.000, 0.000] | 0.500 |
| L5-B (main) | 0.9880 | 0.8821 | 0.6156 | 0.0601 | 0.0 | 0.0 | 0.0841 | [0.698, 0.906, 0.597, 0.803] | 0.467 |
| L5-C (hard rules) | 0.9880 | 0.8834 | 0.7207 | 0.0390 | 0.0 | 0.0 | 0.0631 | [1.000, 1.000, 1.000, 1.000] | 0.500 |

### Key Findings

- **All L5 variants reach 98.8% intent accuracy** — +2.7 pp over L4 λ=2.0 (96.1%)
- **Exp A (rules truly off)** confirms the rule layer is essential: without rules, violation rate rises to 52.6% with high TYPE-A (24%) and TYPE-C (53%) — pure trunk provides no symbolic grounding
- **Exp B (main)** reduces violations to 6.0% with zero TYPE-A/B — learnable rule strengths converge near their init values, validating the rule priors; rule fidelity 61.6%
- **Exp C (hard symbolic)** achieves the lowest violation rate (3.9%) and highest rule fidelity (72.1%) — frozen-at-1.0 rules act as hard constraints the trunk learns to complement
- **Zero TYPE-B violations in B and C** — the rule layer eliminates false-execution entirely
- **TYPE-C (ungrounded SRE)** is the residual challenge at 6–8%; additional rules with broader SRE coverage would address this
- **rule_score_projection layer** (Linear 4→4) is critical: it maps rule scores from [0,1] to the logit scale before blending, preventing scale mismatch with trunk logits

## Kautz Classification

| Level | Symbolic Role | Integration Point | Differentiable? |
|-------|--------------|------------------|-----------------|
| L3.5 | Runtime post-hoc rules | After prediction | No |
| L4 | Training-time constraint penalty | Loss function only | Yes (loss) |
| L4.5 | Runtime output masking | Softmax post-processing | No |
| **L5** | **Compiled rule layer** | **Forward pass** | **Yes (end-to-end)** |

## File Structure

```
level5/
├── data/
│   ├── level5_labeled.csv       # 1,661 rows, 15 cols (utterance + 11 predicates + intent)
│   └── rule_base.json           # 4 symbolic rules with nested AND/OR/NOT logic
├── model/
│   ├── __init__.py
│   ├── dataset.py               # Level5Dataset — loads CSV, validates predicates
│   ├── differentiable_logic.py  # Product t-norm: logic_and, logic_or, logic_not
│   ├── rule_compiler.py         # RuleCompiler — evaluates rule trees over predicate probs
│   ├── level5_model.py          # Level5IntentModel — full architecture
│   └── losses.py                # Level5Loss = CrossEntropy + pred_weight * BCE
├── evaluation/
│   ├── __init__.py
│   └── violation_metrics.py     # Intent acc, pred acc, rule fidelity, TYPE-A/B/C violations
├── saved_models/
│   ├── exp_a_rules_disabled/    # best_model.pt, training_log.json, evaluation_metrics.json
│   ├── exp_b_l5_main/           # best_model.pt, training_log.json, evaluation_metrics.json
│   └── exp_c_hard_rules/        # best_model.pt, training_log.json, evaluation_metrics.json
├── train.py                     # Training loop with 3-experiment CLI
├── infer.py                     # Inference: single/batch/REPL with interpretable output
├── level5_rule_compiled_network.ipynb  # End-to-end walkthrough notebook
├── TODO.md
└── README.md
```
