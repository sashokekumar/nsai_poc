# Level 4.5: Logic-Gated Neuro-Symbolic Intent Classification

> **Reclassification note (Kautz typology review):**
> This folder was originally submitted as **Level 5**. Following review against the Kautz (2022) neuro-symbolic typology, it has been reclassified as **Level 4.5**.
>
> The reason: both implementations here — post-hoc masking and pre-softmax logit gating — apply symbolic rules **as runtime filters external to the neural architecture**. In the post-hoc path the constraint is non-differentiable; in the pre-softmax path gradients flow through the softmax operation, but the masking logic itself is a per-utterance external filter, not a compiled differentiable rule structure inside the model.
>
> In a strict Kautz Type 5 system, symbolic rules must be **compiled into the neural architecture as differentiable rule units** that participate in the forward pass as learned components. The corrected Level 5 implementation is in `level5/`.
>
> This folder is retained as a transition experiment between Level 4 and strict Level 5. It is useful evidence in the Level 4 → 4.5 → 5 comparison table.

## Overview

Level 4.5 is a **runtime logic-gating experiment** that demonstrates two ways of applying symbolic constraints to a neural classifier's outputs. The training data carries per-utterance logical signals — facts, constraints, allowed intents, and suppressed intents — that drive the gating logic. This is a step beyond Level 4's training-time symbolic loss, but it does not yet reach strict Level 5 because the symbolic rules remain external to the network architecture.

The PoC compares two implementations side-by-side:

- **4.5 (post-hoc masking)**: The model predicts raw probabilities first; then a logic gate zeroes suppressed intents and renormalizes. Logic is external and non-differentiable.
- **Pre-softmax gate (toward L5)**: The logic gate masks suppressed intent logits with large negative values before softmax. Logic participates in the forward pass — gradients flow through constrained outputs — but the mask is still an external per-utterance filter, not a learned rule module.

This makes Level 5 decisions **structurally enforced** rather than post-processed, meaning the model cannot produce a suppressed intent even with extreme confidence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Dataset  (data/level5_intents.csv)                                 │
│  Per utterance: gold_intent + facts + constraints +                 │
│                 allowed_intents + suppressed_intents                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TF-IDF Vectorizer  (utterance → feature vector)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  feature vector
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Linear Classifier  (features → logits, one per intent)             │
│  INTENTS = ['investigate', 'execute', 'summarize', 'ops']           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  raw logits
                               ▼
             ┌─────────────────────────────────────┐
             │  4.5 path (post-hoc)                │
             │  softmax → raw_probs                │
             │  → zero suppressed intents          │
             │  → zero non-allowed intents         │
             │  → renormalize → final_probs        │
             └─────────────────────────────────────┘
             ┌─────────────────────────────────────┐
             │  L5 path (pre-softmax gate)         │
             │  mask suppressed logits → -∞        │
             │  → softmax (constrained dist.)      │
             │  → final_probs                      │
             │  (gradients flow through gate)      │
             └─────────────────────────────────────┘
                               │
                               ▼
                    argmax → predicted_intent
```

---

## Files

| File | Description |
|---|---|
| `level5_data_prep.ipynb` | Compiles the dataset; defines `CONSTRAINT_REGISTRY`; runs logical consistency checks and readiness summary |
| `level5_logic_gating_poc.ipynb` | Trains TF-IDF + Linear classifier; implements and compares 4.5 (post-hoc) vs pre-softmax logic gating |
| `l4_a_evaluation.ipynb` | Level-3.5.5 evaluation notebook — loads the trained Level-3.5 `IntentClassifier`, runs it over this dataset, applies post-hoc constraints (4.5 path), and measures violation rates before vs after; includes a detailed markdown explanation of what Level-3.5.5 proves and what it cannot fix |
| `data/level5_intents.csv` | Compiled dataset (utterance + gold_intent + facts + constraints + allowed_intents + suppressed_intents) |

---

## Dataset — What Is Different in Level 4.5

Every previous level uses a dataset with two columns per row: `utterance` and `intent`. The model only sees the text; all symbolic logic is encoded separately in thresholds, rules, or an ontology.

**Level 4.5 changes this.** The dataset has six columns per row:

| Column | Type | Content |
|---|---|---|
| `utterance` | string | Raw user text |
| `gold_intent` | string | Ground-truth intent label |
| `facts` | dict | Per-utterance linguistic signals derived from the text |
| `constraints` | dict | Logical rules derived from the facts |
| `allowed_intents` | list | Intents the model is permitted to predict for this utterance |
| `suppressed_intents` | list | Intents that must be structurally blocked for this utterance |

This means **the constraint logic is baked into the training data**. The model is not just told what to predict — it is told, for each training example, which outputs are logically valid and which must be suppressed. Note: this is a step toward Type 5, but the constraint logic is still an external per-utterance filter — not symbolic rules compiled into the neural architecture.

### Facts (per utterance)

Three boolean signals are extracted from the utterance text:

| Key | Type | Meaning |
|---|---|---|
| `is_question` | bool | Utterance is phrased as a question |
| `has_imperative` | bool | Utterance contains an imperative verb (command phrasing) |
| `has_temporal_ref` | bool | Utterance contains a temporal reference ("last night", "yesterday", etc.) |

Example:
```python
# "why is server host123 cpu high"
facts = {'is_question': True, 'has_imperative': False, 'has_temporal_ref': False}

# "check memory usage on db-server-7"
facts = {'is_question': False, 'has_imperative': True, 'has_temporal_ref': False}

# "what caused the outage last night"
facts = {'is_question': True, 'has_imperative': False, 'has_temporal_ref': True}
```

### Constraints (per utterance — derived from facts)

Two logical constraints are derived from the facts:

| Key | Type | Meaning |
|---|---|---|
| `question_blocks_execute` | bool | A question phrasing means execution should be suppressed |
| `prefer_investigate_when_uncertain` | bool | Uncertainty markers bias toward investigation |

Example:
```python
# "why is server host123 cpu high" (is_question=True)
constraints = {'question_blocks_execute': True, 'prefer_investigate_when_uncertain': True}
# → suppressed_intents: ['execute'], allowed_intents: ['investigate', 'ops', 'summarize']

# "check memory usage on db-server-7" (has_imperative=True, is_question=False)
constraints = {'question_blocks_execute': False, 'prefer_investigate_when_uncertain': False}
# → suppressed_intents: [], allowed_intents: ['execute', 'investigate', 'ops', 'summarize']
```

### Suppression Coverage

| suppressed_intents | Count | % of dataset |
|---|---|---|
| `[]` (no suppression) | 434 | 71% |
| `['execute']` | 180 | 29% |

180 of 614 utterances (29%) have `execute` suppressed because they are phrased as questions. These are the rows where `question_blocks_execute: True`.

### Dataset Statistics

- **Source**: `data/intents_level5.csv` (repo root); compiled into `level4_5_logic_gating/data/level5_intents.csv`
- **Records**: 614 utterances
- **Intent Classes**: `out_of_scope` 169 · `execution` 150 · `investigate` 149 · `summarization` 146
- **Model canonical intents**: `['investigate', 'execute', 'summarize', 'ops']`

---

## Configuration

| Parameter | Value | Where | Purpose |
|---|---|---|---|
| `INTENTS` | `['investigate', 'execute', 'summarize', 'ops']` | `level5_logic_gating_poc.ipynb` | Canonical output classes for the classifier |
| `TfidfVectorizer` | default | notebook | Utterance vectorization |
| `np.random.seed` | 42 | notebook | Reproducibility |
| Logic gate mask value | `-∞` (large negative) | `apply_l45_masking` / L5 forward | Suppressed logits before softmax |
| Fallback (all zeroed) | uniform over allowed | `apply_l45_masking` | Prevents zero-probability edge case |

---

## Constraint Registry

Defined declaratively in `level5_data_prep.ipynb`. Importable and computation-free.

| Constraint key | Type | Description | Rationale |
|---|---|---|---|
| `block_execute` | **hard** | Prevent direct execution when present | Questions/clarifications must not trigger execution |
| `prefer_investigate` | soft | Prefer investigation over execution | Uncertainty markers bias toward investigative actions |
| `allow_execute` | soft | Allow execution when imperative language detected | Imperative phrasing indicates operator intent |
| `require_diagnostic_investigate` | **hard** | Require diagnostic investigation for system-state questions | System-state inquiries require diagnostics before action |
| `require_immediacy_flag` | soft | Mark request as immediate/urgent when temporal + imperative | Communicates urgency for downstream handling |

**Hard constraints** cannot be overridden by the model regardless of confidence. **Soft constraints** bias the prediction but do not structurally block any intent.

---

## What Makes Pre-Softmax Gating Different from Post-Hoc Masking

### Timing of Logic Application

```
4.5:  model(x) → raw_probs → apply_logic(raw_probs) → final_probs
  Logic is external and non-differentiable.

L5:   model(x, constraints) → [mask logits] → softmax → final_probs
  Logic is inside the forward pass, before softmax.
```

### Gradient Flow

```
4.5:  Gradients flow through unconstrained predictions. Logic is a post-process.
      The model can (and does) assign high probability to suppressed intents internally.

L5:   Gradients flow through constrained predictions. The model's loss only
      evaluates valid outputs. Invalid intent paths are not reinforced during learning.
```

### Structural Guarantee

```
4.5:  The raw model can produce any intent; corrections are applied after.
      If the logic layer is bypassed, violations occur.

L5:   The model cannot produce a suppressed intent by construction.
      Suppressed logits are set to -∞ before softmax → near-zero probability.
      Violation rate: exactly 0 by design.
```

---

## System Flow Diagram

```mermaid
flowchart TD
    A([User Utterance]) --> B[Facts Extraction\nis_question · has_imperative · has_temporal_ref]
    B --> C[Constraint Derivation\nquestion_blocks_execute · prefer_investigate_when_uncertain]
    C --> D[allowed_intents · suppressed_intents]

    A --> E[TF-IDF Vectorizer\nutterance → feature vector]
    E --> F[Linear Classifier\nfeatures → raw logits\none per intent in INTENTS]

    F --> G{Which approach?}

    G -- 4.5 post-hoc --> H[softmax\nraw probabilities]
    H --> I[Zero suppressed intents\nZero non-allowed intents\nRenormalize]
    I --> J[argmax → predicted_intent]

    G -- L5 pre-softmax --> K[Logic Gate\nmask suppressed logits → large negative value]
    K --> L[softmax\nconstrained probability distribution]
    L --> M[argmax → predicted_intent]

    D --> I
    D --> K

    J --> N([Output\npredicted_intent\nvalid by constraint])
    M --> N
```

---

## Why `execute` Is the Only Suppressed Intent

The suppression logic is derived directly from the `facts` column. `execute` is suppressed whenever `question_blocks_execute: True`, which fires when `is_question: True`. This mirrors the same safety asymmetry found in Levels 1–3:

```
execute     →  mutates state  →  irreversible  →  suppressed for any question phrasing
investigate →  read-only      →  reversible    →  always in allowed_intents
summarize   →  read-only      →  safe          →  always in allowed_intents
ops         →  operational    →  context-dep.  →  always in allowed_intents
```

In 29% of utterances in the dataset (180 of 614), `execute` is structurally suppressed because the utterance is phrased as a question. The model is never trained to predict `execute` on those examples — not just penalized, but **structurally prevented** in L5, and corrected after the fact in 4.5.

---

## Notebook Details

### `level5_data_prep.ipynb`

Prepares and validates the Level 5 dataset. Run all cells to compile and check the dataset before running the main PoC notebook.

| Cell | Purpose |
|---|---|
| 1 | Imports — pandas, numpy |
| 2 | Locate repo root; load `level5_intents.csv`; compile and save to `level4_5_logic_gating/data/` |
| 3 | Define `CONSTRAINT_REGISTRY` — all five constraint keys with type, description, and rationale |
| 4 | Dataset sanity checks — load compiled CSV, parse list-like columns (`allowed_intents`, `suppressed_intents`), validate types |
| 5 | Diagnostics — check for missing values, malformed rows, and constraint/intent consistency |
| 6 | Readiness summary — compute `% with hard constraints`, `% with suppressed_intents`, `logical_violations_count`; print `VERDICT` |

### `level5_logic_gating_poc.ipynb`

Main PoC notebook. Trains the classifier and compares 4.5 vs L5 logic gating.

| Cell | Purpose |
|---|---|
| 1 | Imports — numpy, pandas, sklearn, INTENTS definition |
| 2 | Load dataset from `level4_5_logic_gating/data/`; parse `allowed_intents` and `suppressed_intents` |
| 3 | TF-IDF vectorize utterances; train/test split |
| 4 | Train base LogReg classifier (no logic gating); evaluate baseline accuracy |
| 5 | **4.5 post-hoc masking** — `apply_l45_masking(probs, allowed, suppressed)`: zero suppressed intents, zero non-allowed, renormalize; compute 4.5 accuracy |
| 6 | **L5 pre-softmax gate** — mask suppressed logits with large negative before softmax; compute L5 accuracy and violation rate |
| 7 | Side-by-side comparison — baseline vs 4.5 vs L5 accuracy and violation counts |
| 8 | Save results to `level4_5_logic_gating/data/` |

### `l4_a_evaluation.ipynb`

Level-3.5.5 evaluation notebook. Runs the trained Level-3.5 `IntentClassifier` over the Level-5 dataset and applies post-hoc constraint masking (the 4.5 path). This notebook establishes the baseline for understanding what inference-time constraint enforcement can and cannot achieve — motivating the need for embedded logic in Level-5.

| Cell | Purpose |
|---|---|
| 1 | Load Level-3.5 `IntentClassifier` (from `level3_5/intent_model.joblib`); load Level-5 dataset; normalize shorthand intent names (`execute` → `execution`, `summarize` → `summarization`, `ops` → `out_of_scope`) to match model class names |
| 2 | Baseline Level-3.5 inference (unconstrained) — run `predict_proba` over all 614 utterances; store raw probability distributions |
| 3 | Apply Level-3.5.5 post-hoc constraints — zero suppressed intents, zero non-allowed intents, renormalize; store constrained predictions |
| 4 | Comparative metrics — violation rate before vs after, intent flip rate, mean score delta, intent distribution before vs after |
| 5 | Concrete example review — list utterances where the Level-3.5 raw prediction was invalid and Level-3.5.5 corrected it |
| 6 | Level-3.5.5 verdict — print `Level-3.5.5 SHOWS CLEAR VALUE` or `LIMITED VALUE` based on violation reduction rate |
| Markdown | Detailed explanation covering: what errors Level-3.5 makes; hard vs soft constraint violations; how Level-3.5.5 fixes them without retraining; what Level-3.5.5 cannot fix; and why these limitations motivate Level-5 |

**Quick start:**
```bash
cd level4_5_logic_gating
jupyter notebook level5_data_prep.ipynb         # prepare and validate dataset
jupyter notebook l4_a_evaluation.ipynb           # Level-3.5.5 post-hoc evaluation
jupyter notebook level5_logic_gating_poc.ipynb   # Level-4.5 pre-softmax gating PoC
```

---

## Worked Inference Examples

### Example 1 — Execute Suppressed (Question): `"why is server host123 cpu high"`

#### Step 1 — Facts Extracted

| Signal | Value | Source |
|---|---|---|
| `is_question` | `True` | question word "why" detected |
| `has_imperative` | `False` | no command verb |
| `has_temporal_ref` | `False` | no time reference |

#### Step 2 — Constraints Derived

| Constraint | Value |
|---|---|
| `question_blocks_execute` | `True` |
| `prefer_investigate_when_uncertain` | `True` |

```
allowed_intents:    ['investigate', 'ops', 'summarize']
suppressed_intents: ['execute']
```

#### Step 3 — 4.5 Path

```
TF-IDF → LogReg → raw_probs: [investigate: 0.55, execute: 0.30, summarize: 0.10, ops: 0.05]
apply_l45_masking → zero execute → [investigate: 0.55, execute: 0.00, summarize: 0.10, ops: 0.05]
renormalize        → [investigate: 0.786, execute: 0.000, summarize: 0.143, ops: 0.071]
argmax             → investigate ✅
```

#### Step 4 — L5 Path

```
TF-IDF → Linear → logits: [investigate: 2.1, execute: 1.2, summarize: 0.4, ops: 0.2]
logic gate → logits: [investigate: 2.1, execute: -∞, summarize: 0.4, ops: 0.2]
softmax    → [investigate: 0.85, execute: ~0.00, summarize: 0.10, ops: 0.05]
argmax     → investigate ✅  (execute cannot win by construction)
```

---

### Example 2 — Execute Allowed (Imperative): `"check memory usage on db-server-7"`

#### Step 1 — Facts

| Signal | Value |
|---|---|
| `is_question` | `False` |
| `has_imperative` | `True` |
| `has_temporal_ref` | `False` |

#### Step 2 — Constraints

```
question_blocks_execute:       False
prefer_investigate_when_uncertain: False

allowed_intents:    ['execute', 'investigate', 'ops', 'summarize']  (all)
suppressed_intents: []  (none)
```

#### Step 3 — Both Paths

No suppression applies. Both 4.5 and L5 pass the model's raw prediction through unchanged. Model predicts `investigate` (imperative "check" maps to investigation).

```json
{"predicted_intent": "investigate", "suppressed_intents": [], "constraint_applied": false}
```

---

### Example 3 — Out of Scope / Unknown Intent

Any utterance where the model's top intent after gating maps to a class not in `allowed_intents` falls back to uniform over allowed in the 4.5 path (renormalize over all-zero vector). In L5, large negative masking ensures suppressed classes cannot win softmax.

---

### Summary Comparison Table

| Utterance | is_question | suppressed | 4.5 result | L5 result | Structural guarantee? |
|:---|:---:|:---|:---|:---|:---:|
| "why is server host123 cpu high" | ✅ | `['execute']` | `investigate` (post-hoc fix) | `investigate` (pre-softmax) | ✅ L5 only |
| "check memory usage on db-server-7" | ❌ | `[]` | `investigate` (raw) | `investigate` (raw) | N/A |
| "what caused the outage last night" | ✅ | `['execute']` | `investigate` (post-hoc fix) | `investigate` (pre-softmax) | ✅ L5 only |

---

## Comparison with Other Levels

| Level | Dataset extra signals | Symbolic layer | Logic timing | Structural guarantee |
|---|---|---|---|---|
| 0 | None — utterance + intent only | None | N/A | None |
| 1 | None | Symbolization predicates + JSON rule engine | Inference-time (post-model) | Blocked states only |
| 2 | None | Clause extractor + validator + feedback | Inference-time (post-model) | Hard rules block |
| 3 | None | Symbol emitter + rule dispatch | Inference-time (post-model) | `requires_approval` flag |
| 4 | None | Ontology + domain guard + reasoner | Inference-time (post-model) | Domain guard blocks |
| **4.5** | **facts · constraints · allowed_intents · suppressed_intents per row** | **Logic gate (pre-softmax or post-hoc)** | **Inside forward pass (pre-softmax) or post-model (post-hoc)** | **Zero violations by construction (pre-softmax path)** |

The key progression: in Levels 1–4, the training data is constraint-free and all symbolic logic is applied after training. In Level 4.5, the constraints travel with the data — the model is trained with explicit knowledge of which outputs are valid for each utterance. However, the constraint logic remains an external per-utterance filter, not a differentiable rule module compiled into the architecture. That is the distinction that motivates strict Level 5.

---

**Level 4.5 Status**: ✅ PoC Complete (reclassified from Level 5)  
**Architecture**: Logic-Gated Neuro-Symbolic Classifier  
**Key Innovations**: Constraint-Enriched Training Data · Pre-Softmax Logic Gate · Zero Violation Guarantee (pre-softmax path) · Dataset-Level Constraint Encoding  
**Role in progression**: Transition experiment between Level 4 (training-time symbolic loss) and Level 5 (symbolic rules compiled into differentiable neural architecture). Retained as comparison evidence.
