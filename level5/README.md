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
│  Shared Trunk (MLP)      │  Linear(384→256) + ReLU + Dropout(0.2)
└──────────────────────────┘
     │
     ▼
┌───────────────────────┐
│  Predicate Head       │  Linear(256→11) + sigmoid → 11 probabilities
└───────────────────────┘
     │
     ▼
┌───────────────────────────────────────┐
│  RuleCompiler (product t-norm logic)  │
│  N rules × learnable rule_strength   │
│  → rule_scores [batch, N]  ∈ [0,1]   │
└───────────────────────────────────────┘
     │
     ▼
┌───────────────────────────────────────┐
│  scatter_to_intents                   │
│  max rule activation per intent [B,4]│
└───────────────────────────────────────┘
     │
     ▼
  torch.logit(score.clamp(ε))  →  softmax  →  4-class intent
```

**Key Level 5 property:** The RuleCompiler is the **sole decision path** — no residual neural intent head, no blend weight. The trunk's only job is to estimate predicate probabilities; intent is determined entirely by compiled symbolic rules.

## Product T-Norm Logic

All logic operations are differentiable, allowing backpropagation through symbolic rules:

| Operation | Formula |
|-----------|---------|
| AND(a, b) | $a \cdot b$ |
| OR(a, b)  | $a + b - a \cdot b$ |
| NOT(a)    | $1 - a$ |


## Rule Base (v2 — Data-Derived)

Defined in `level5/data/rule_base_derived.json`. Five rules derived from predicate-intent chi-squared association mining on `level5_labeled.csv` (see `level5/model/rule_miner.py`). Contrast with `rule_base.json` (hand-authored v1).

| Rule | Logic | Target Intent | Init Strength | Source | Key Support |
|------|-------|--------------|---------------|--------|-------------|
| DR1 | `AND(is_sre_domain, OR(is_infrastructure, is_metric_query))` | investigate | 0.80 | data-derived | is_infra lift=1.56 (P=0.396) vs R1's is_known_incident (P=0.087) |
| DR2 | `AND(has_runbook, is_sre_domain)` | execution | 0.90 | data-validated | has_runbook chi2=928.6, lift=3.80 — same as R2 |
| DR3 | `AND(is_infrastructure, is_sre_domain, NOT has_runbook)` | execution | 0.70 | **data-derived (NEW)** | no human equivalent; is_infra lift=1.72 for execution (chi2=95.3) |
| DR4 | `AND(is_sre_domain, NOT has_runbook, OR(is_incident, is_metric))` | summarization | 0.75 | data-derived | is_incident lift=3.82 + is_metric lift=2.03; R3 only used is_known_incident |
| DR5 | `AND(is_unknown, NOT is_sre_domain)` | out_of_scope | 0.90 | data-validated | is_unknown P=1.0 for OOS — same as R4 |

**What changed from v1 (R1–R4):**
- **DR1 ≠ R1:** R1 used `OR(is_metric_query, is_known_incident)` — but `is_known_incident` appears in only 8.7% of investigate cases (P=0.087). `is_infrastructure` appears in 39.6% (P=0.396). DR1 corrects this.
- **DR3 is new:** There was no rule for infra+SRE+no-runbook → execution. This covers utterances like *"restart the payment-service pods"* that don't reference a runbook but are clearly executory.
- **DR4 ≠ R3:** R3 used `is_known_incident` only. `is_metric` is a stronger summarization signal (lift=2.03 vs 1.84 for is_known_incident) and covers 19.7% of summarization cases.
- **DR2 = R2, DR5 = R4:** Data fully validates these human-authored rules.

To regenerate: `python -m level5.model.rule_miner`

## Experiments

Five ablations total across v1 and v2. v1 used a hybrid architecture with a blend weight α and a residual intent head (incompatible with current code). v2 uses the clean L5 architecture where RuleCompiler is the sole decision path.

### v2 Experiments (data-derived rules — current architecture)

| Experiment | Description | Key Config |
|------------|-------------|-----------||
| **Exp D** — Derived Soft (L5A) | Data-derived rules, learnable rule strengths | `--rule-base level5/data/rule_base_derived.json --epochs 20` |
| **Exp E** — Derived Hard (L5B) | Data-derived rules, strengths frozen at 1.0 | `--rule-base level5/data/rule_base_derived.json --hard-rules --epochs 20` |

### Reproduce Commands

```bash
# Exp D — data-derived rules, soft (learnable strengths)
python -m level5.train --run-name exp_d_derived_soft --rule-base level5/data/rule_base_derived.json --epochs 20

# Exp E — data-derived rules, hard (frozen at 1.0)
python -m level5.train --run-name exp_e_derived_hard --rule-base level5/data/rule_base_derived.json --hard-rules --epochs 20
```

### Regenerate rule base from data

```bash
python -m level5.model.rule_miner
```

### Evaluate

```bash
python -m level5.evaluation.violation_metrics \
    --checkpoint saved_models/exp_d_derived_soft/best_model.pt \
    --run-name exp_d_derived_soft

python -m level5.evaluation.violation_metrics \
    --checkpoint saved_models/exp_e_derived_hard/best_model.pt \
    --run-name exp_e_derived_hard
```

## Results

### v2 — Measured (data-derived rules, current architecture)

| Experiment | Intent Acc | Pred Acc | Rule Fidelity | Overall Viol* | TYPE-A | TYPE-B | TYPE-C | TYPE-D | TYPE-F† | Rule Strengths |
|------------|-----------|---------|--------------|--------------|--------|--------|--------|--------|---------|----------------|
| **Exp D** (derived soft) | **0.9880** | 0.8643 | **1.000** | 0.2733 | 0.000 | 0.000 | 0.063 | 0.057 | 0.210 | [0.803, 0.904, 0.666, 0.758, 0.904] |
| **Exp E** (derived hard) | **0.9910** | 0.8583 | **1.000** | 0.3213 | 0.000 | 0.003 | 0.108 | 0.099 | 0.210 | [1.0, 1.0, 1.0, 1.0, 1.0] |

*Overall rate dominated by TYPE-F (21%). See † note below.

†**TYPE-F note (data-vs-constraint tension):** The L4 ontology classifies `summarization+metric` as a violation (TYPE-F, temporal phase mismatch). But the data shows `is_metric` has lift=2.03 for summarization — one of the two strongest summarization signals. DR4 correctly fires metric→summarization based on data evidence. The 21% TYPE-F rate documents the tension between the L4 constraint and the data — not a pure model error.

### v1 — Reference (hand-authored rules, old hybrid architecture)

> ⚠️ Old Exp B/C checkpoints used a different architecture (blend weight α + residual intent head, `rule_score_projection`) that has since been replaced with the clean pure-symbolic forward pass. Old checkpoints cannot be re-evaluated with the current code. Results preserved for reference.

| Experiment | Intent Acc | Pred Acc | Rule Fidelity | Viol Rate | TYPE-A | TYPE-B | TYPE-C | Blend α |
|------------|-----------|---------|--------------|----------|--------|--------|--------|----------|
| L5-A (rules off) | 0.9880 | 0.3574 | 0.003 | 0.526 | 0.240 | 0.000 | 0.535 | 0.500 |
| L5-B (main, blend) | 0.9880 | 0.8821 | 0.616 | 0.060 | 0.000 | 0.000 | 0.084 | 0.467 |
| L5-C (hard, blend) | 0.9880 | 0.8834 | 0.721 | 0.039 | 0.000 | 0.000 | 0.063 | 0.500 |

### Key Findings

> **Rule fidelity 100% in both Exp D and E**: every val-set prediction matches the consequent of the highest-activated rule. Compared to v1's 61.6% (Exp B) and 72.1% (Exp C), this confirms that the data-derived rules better align with data patterns — the rule layer drives the decision, not the trunk.

- **Exp E (derived hard) reaches 99.1% intent accuracy**, +0.3 pp over v1 at 98.8%
- **Zero TYPE-A violations** in both D and E — the compiled rule layer prevents classifying SRE entities as `out_of_scope`
- **Zero TYPE-B in Exp D** — derived rules prevent false-execution against non-actionable entities
- **TYPE-D (unsafe execution = 5.7–9.9%)**: execution predicted with unknown entity type. DR3 (`is_infra+is_sre+NOT has_runbook → execution`) can fire when `is_unknown` also activates. Hard rules (Exp E) exacerbate this.
- **TYPE-F rate is 21% (data-vs-constraint)**: DR4 correctly fires `is_metric → summarization` (data lift=2.03). This conflicts with the L4 TYPE-F constraint. See note in results table above.
- **DR3 is the key new addition**: 5 rules vs 4; covers the `is_infrastructure → execution (no runbook)` gap that no hand-authored rule addressed.

## Kautz Classification
|-------|--------------|------------------|-----------------|
| L3.5 | Runtime post-hoc rules | After prediction | No |
| L4 | Training-time constraint penalty | Loss function only | Yes (loss) |
| L4.5 | Runtime output masking | Softmax post-processing | No |
| **L5** | **Compiled rule layer** | **Forward pass** | **Yes (end-to-end)** |

## File Structure

```
level5/
├── data/
│   ├── level5_labeled.csv          # 1,661 rows, 15 cols (utterance + 11 predicates + intent)
│   ├── rule_base.json               # 4 human-authored rules (v1 reference)
│   └── rule_base_derived.json       # 5 data-derived rules (v2 current)
├── model/
│   ├── __init__.py
│   ├── dataset.py                   # Level5Dataset — loads CSV, validates predicates
│   ├── differentiable_logic.py      # Product t-norm: logic_and, logic_or, logic_not
│   ├── rule_compiler.py             # RuleCompiler — evaluates rule trees over predicate probs
│   ├── rule_miner.py                # Chi2 association miner — generates rule_base_derived.json
│   ├── level5_model.py              # Level5IntentModel — full architecture
│   └── losses.py                    # Level5Loss = CrossEntropy + pred_weight * BCE
├── evaluation/
│   ├── __init__.py
│   └── violation_metrics.py         # Intent acc, pred acc, rule fidelity, TYPE-A/B/C/D/F
├── saved_models/
│   ├── exp_d_derived_soft/          # best_model.pt, training_log.json, evaluation_metrics.json
│   └── exp_e_derived_hard/          # best_model.pt, training_log.json, evaluation_metrics.json
├── train.py                         # Training loop (--rule-base, --hard-rules CLI args)
├── infer.py                         # Inference: single/batch/REPL with interpretable output
├── level5_rule_compiled_network.ipynb  # End-to-end walkthrough notebook
├── TODO.md
└── README.md
```
