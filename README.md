# NSAI POC: Neuro-Symbolic AI for Intent Classification

## Overview
This repository demonstrates a **progressive evolution from Level 0 to Level 6** of Neuro-Symbolic AI (NSAI) architectures for Natural Language Understanding (NLU), specifically intent classification in an SRE (Site Reliability Engineering) operations domain.

**Goal**: Show how pure statistical models (Level 0) can be progressively enhanced with symbolic reasoning, ontology constraints, differentiable rule architectures, and self-evolving symbol vocabularies to create more robust, explainable, and safe NLU systems.

The progression is anchored to the **Kautz (2022) neuro-symbolic typology**: each level corresponds to a distinct integration point between neural and symbolic components.

## Dataset
- **File**: `data/intents_base.csv`
- **Size**: 1,661 labeled utterances (shared baseline across Levels 0–3.5)
- **Intent Classes**: `investigate`, `execution`, `summarization`, `out_of_scope`
- **Format**: Pure text utterances with intent labels

Level 4 and above extend this with entity type and domain validity labels (`data/labeled_clean.csv` inside each level's `data/` folder). Level 6 adds 300 harder boundary-case utterances (`level6/data/level6_seed.csv`, total 1,961 rows).

## Environment Setup
```bash
# Clone repository
git clone <repo-url>
cd nsai_poc

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Python Version**: 3.13  
**Key Dependencies**: pandas, numpy, scikit-learn, torch, sentence-transformers, jupyter, matplotlib, seaborn

> Level 4 and Level 5 require `torch` and `sentence-transformers` (the `all-MiniLM-L6-v2` encoder). These are not listed in `requirements.txt` — install them separately if needed:
> ```bash
> pip install torch sentence-transformers
> ```

## Architecture Progression

| Level | Kautz Type | Neural–Symbolic Integration Point | Status |
|-------|-----------|-----------------------------------|--------|
| 0 | — | Pure ML (no symbolic) | ✅ Complete |
| 1 | Type 1 | Post-hoc symbolic rule layer | ✅ Complete |
| 2 | Type 1 | Symbolic clause extraction + validation | ✅ Complete |
| 3 | Type 2 | Neural backbone + symbolic pipeline | ✅ Complete |
| 3.5 | Type 3 | Inference-time ontology-grounded reasoning | ✅ Complete |
| 4 | Type 4 | Training-time differentiable constraint loss | ✅ Complete |
| 4.5 | Type 4.5 | Runtime logic-gating (transition experiment) | ✅ Complete |
| 5 | Type 5 | Rules compiled into differentiable architecture | ✅ Complete |
| 6 | Type 6 (ext.) | Self-evolving symbolic vocabulary | 🔄 In Progress |

---

### 📊 [Level 0: Baseline Statistical Classifier](level0/README.md)
**Kautz**: N/A — pure ML baseline  
**Algorithm**: TF-IDF + Logistic Regression

- Confidence thresholding (0.7) with abstain logic
- Token-level evidence extraction
- ~91% accuracy on 1,661 utterances

[→ View Level 0 Details](level0/README.md)

---

### 🧠 [Level 1: Neuro-Symbolic Decision Layer](level1/README.md)
**Kautz**: Type 1 — symbolic layer applied post-hoc to neural output  
**Innovation**: Separates **intent** (what user wants) from **decision state** (what system does)

**Layer A — Statistical**: TF-IDF + Logistic Regression → intent + confidence  
**Layer B — Symbolic**: 4 deterministic rules with strict priority precedence

| Rule | Priority | Category | Effect |
|------|----------|----------|--------|
| R1 | 100 | Quality | Block low-token inputs |
| R4 | 90 | Safety | Block risky executions |
| R2 | 50 | Ambiguity | Flag low-confidence predictions |
| R3 | 40 | Ambiguity | Flag close-margin predictions |

Decision states: `accepted` · `needs_clarification` · `blocked`

Level 1B additionally emits a `decision_trace` per utterance (`detectors_fired`, `hard_rules_failed`, `soft_rules_passed`, `alternatives_eliminated`) for full auditability.

[→ View Level 1 Details](level1/README.md)

---

### 🔩 [Level 2: Clause Extraction & Symbolic Validation](level2/README.md)
**Kautz**: Type 1 — neural adapter feeds a fully deterministic symbolic pipeline  
**Innovation**: Decomposes intent into structured **clause candidates**, validates against policy rules, and iteratively refines with feedback

**Pipeline**: Neural Adapter → Clause Extractor → Normalizer → Validator

Eight clause fields: `entity`, `operation`, `metric`, `time_window`, `environment`, `condition`, `constraint`, `output_format`

When `decision_state == "needs_clarification"`, a feedback signal is passed back to the adapter for a refined extraction pass (up to `MAX_ITERATIONS`). The critical path is fully deterministic — every adapter candidate is re-validated by the symbolic layer before any decision is made.

[→ View Level 2 Details](level2/README.md)

---

### 🔮 [Level 3: Neural Intent Classifier with Symbolic Pipeline](level3/README.md)
**Kautz**: Type 2 — raw neural output is converted to a typed symbol before any decision  
**Innovation**: Strict boundary between neural and symbolic layers via a **Symbol Emitter**

**3-stage pipeline**:
1. **Neural Model** (`nn.Embedding` → mean-pool → `Linear`) → class index (int)
2. **Symbol Emitter** → `IntentSymbol` (validated dataclass; no logits cross this boundary)
3. **Rule Engine** → `{action, requires_approval}` (pure symbol dispatch)

The neural backbone can be swapped without touching the symbolic layer at all.

[→ View Level 3 Details](level3/README.md)

---

### 🌐 [Level 3.5: Ontology-Grounded Neuro-Symbolic Pipeline](level3_5/README.md)
**Kautz**: Type 3 — symbolic components (EntityMatcher, Reasoner, Planner, Responder) remain active at inference time as post-hoc correctors  
**Innovation**: Extends intent recognition into structured reasoning, planning, and response generation using ontology-grounded matchers

**4-stage pipeline**:
1. **Semantic Parser** → `IntentFrame` (intent, entity, symptom, time_context, confidence)
2. **Reasoner** → deterministic mode selection + context enrichment
3. **Planner** → ordered list of plan steps
4. **Responder** → user-facing response string

No raw ML probabilities cross the symbolic boundary. The reasoner reads symbolic mode labels, not confidence scores.

> **Reclassification note**: Originally submitted as Level 4. Reclassified to Level 3.5 because symbolic components remain active at inference; a strict Kautz Type 4 system compiles symbolic knowledge into weights at training time.

[→ View Level 3.5 Details](level3_5/README.md)

---

### ⚙️ [Level 4: Symbolically Supervised Neural Model](level4/README.md)
**Kautz**: Type 4 — symbolic ontology knowledge shapes the model at training time via a differentiable constraint loss; inference is pure neural (no symbolic guards)  
**Encoder**: `all-MiniLM-L6-v2` (384-dim, frozen)  
**Architecture**: Shared trunk → 3 heads (intent, entity, domain) + symbolic constraint loss

**Constraint loss** — soft differentiable penalty over 10 disallowed `(intent, entity_type)` pairs:

$$\mathcal{L} = \mathcal{L}_\text{intent} + \alpha\mathcal{L}_\text{entity} + \beta\mathcal{L}_\text{domain} + \lambda\mathcal{L}_\text{constraint}$$

**Results** (test set, n=333):

| Run | Intent Acc | Violation Rate |
|-----|-----------|---------------|
| Level 3.5 (runtime symbolic) | 93.1% | 64.3% (schema mismatch) |
| Level 4 λ=0 (baseline neural) | 97.0% | 5.1% |
| **Level 4 λ=1.0** ✓ | **96.7%** | **3.6%** |
| Level 4 λ=2.0 | 96.1% | 2.1% |

Sweet spot: λ=1.0 — 30% violation reduction with no meaningful accuracy cost.

[→ View Level 4 Details](level4/README.md)

---

### 🔀 [Level 4.5: Runtime Logic-Gating](level4_5_logic_gating/README.md)
**Kautz**: Type 4.5 — symbolic rules applied as runtime filters external to the neural architecture (transition experiment between L4 and L5)  
**Compares two gating strategies**:

- **Post-hoc masking**: model predicts raw probs → zero suppressed intents → renormalize (non-differentiable)
- **Pre-softmax gate**: mask suppressed logits with −∞ before softmax (gradients flow, but mask is still an external per-utterance filter)

> **Reclassification note**: Originally submitted as Level 5. Reclassified to Level 4.5 because the symbolic rules remain external to the network architecture. A strict Kautz Type 5 system compiles rules into the architecture as differentiable rule units.

[→ View Level 4.5 Details](level4_5_logic_gating/README.md)

---

### 🧬 [Level 5: Rule-Compiled Network](level5/README.md)
**Kautz**: Type 5 — symbolic rules compiled directly into the neural architecture as a differentiable rule layer using product t-norm logic  
**Encoder**: `all-MiniLM-L6-v2` (384-dim, frozen)

**Architecture**:
```
Utterance → Encoder → Shared Trunk (MLP)
                          ├── Predicate Head (11 predicates, sigmoid)
                          │       └── RuleCompiler (product t-norm, 4 learnable rule strengths)
                          │               └── rule_score_projection → rule_logits
                          └── Intent Head → trunk_logits
                                    └── intent_logit = α·rule_logits + (1-α)·trunk_logits → softmax
```

**Product t-norm logic** (all operations differentiable):  
AND(a,b) = $a \cdot b$ · OR(a,b) = $a + b - ab$ · NOT(a) = $1 - a$

**Rule base** (4 rules, `level5/data/rule_base.json`):

| Rule | Logic | Target | Init Strength |
|------|-------|--------|--------------|
| R1 | OR(is_metric_query, is_known_incident) | investigate | 0.7 |
| R2 | AND(has_runbook, is_sre_domain) | execution | 0.9 |
| R3 | AND(is_known_incident, is_sre_domain, NOT has_runbook) | summarization | 0.6 |
| R4 | AND(is_unknown, NOT is_sre_domain) | out_of_scope | 0.8 |

The blend weight α is a learnable parameter (sigmoid, init 0.5) controlling how much rule scores vs trunk logits drive predictions.

[→ View Level 5 Details](level5/README.md)

---

### 🌱 [Level 6: Self-Evolving Neuro-Symbolic Architecture](level6/TODO.md)
**Kautz**: Type 5 architecture with a self-modifying symbolic substrate (practical extension toward Type 6)  
**Innovation**: The symbolic rule vocabulary is no longer static. Failures in the Level 5 model drive discovery of new predicate-space symbols → candidate rules → validation → promotion into the rule base.

**Evolution loop**:
```
L5 inference → failure collection → predicate-space clustering (HDBSCAN)
→ symbol birth → lifecycle registry (Proposed → Experimental → Active → Weakening → Deprecated)
→ candidate rule generation → no-retrain validation → [optional] fine-tune validation
→ rule_base update → L5 inference (next cycle)
```

**Implemented so far** (Tasks 1–5 of 19):
- `ReasoningState` dataclass (`level6/reasoning_state.py`) — anchors all L6 operations; tensor-backed with named predicate accessors
- `lifecycle.py` — `SymbolStatus` enum + promotion/deprecation criteria
- `build_seed_dataset.py` — 300 harder boundary-case utterances across 7 categories; combined dataset 1,961 rows saved to `level6/data/level6_seed.csv`
- Typology and scope documented in `level6/__init__.py`

**Remaining** (Tasks 6–19): FailureCollector, SymbolCluster, SymbolRegistry, RuleCandidateGen, RuleValidator, EvolutionEngine, evaluation metrics, experiments, notebook, README.

---

## Project Structure
```
nsai_poc/
├── README.md
├── requirements.txt
├── data/
│   └── intents_base.csv              # Shared dataset (1,661 utterances)
├── level0/                           # TF-IDF + LogReg baseline
├── level1/                           # 2-layer neuro-symbolic (statistical + rule)
├── level2/                           # Clause extraction & symbolic validation pipeline
├── level3/                           # Neural backbone + symbol emitter + rule engine
├── level3_5/                         # Ontology-grounded 4-stage pipeline (inference-time symbolic)
├── level4/                           # Symbolically supervised neural (training-time constraint loss)
├── level4_5_logic_gating/            # Runtime logic-gating experiment (L4→L5 transition)
├── level5/                           # Rule-compiled differentiable network (product t-norm)
├── level6/                           # Self-evolving symbolic vocabulary (in progress)
├── validation/                       # Multi-level comparison notebook
├── artifacts/                        # Cross-level evaluation outputs
├── models/                           # Shared model artifacts
└── tools/                            # Utility scripts
```

## Quick Start

### Run a specific level
```bash
# Level 0 — baseline
jupyter notebook level0/level0_tfidf_classification.ipynb

# Level 1 — neuro-symbolic decision layer
jupyter notebook level1/level1_symbolic_intent_classification.ipynb

# Level 2 — clause extraction pipeline
jupyter notebook level2/l2_clause_pipeline.ipynb

# Level 3 — neural + symbol emitter + rule engine
jupyter notebook level3/level3_intent_classifier.ipynb

# Level 3.5 — ontology-grounded pipeline
jupyter notebook level3_5/level3_5_ns_intent.ipynb

# Level 4 — symbolically supervised neural (requires torch + sentence-transformers)
python -m level4.train --lam 1.0 --epochs 20 --run-name lam1_0
jupyter notebook level4/level4_symbolic_loss.ipynb

# Level 4.5 — logic-gating experiment
jupyter notebook level4_5_logic_gating/level5_logic_gating_poc.ipynb

# Level 5 — rule-compiled network (requires torch + sentence-transformers)
jupyter notebook level5/level5_rule_compiled_network.ipynb

# Level 6 — seed dataset generation
python -m level6.build_seed_dataset
```

### Cross-level validation
```bash
jupyter notebook validation/validation.ipynb
```

## Key Concepts

### Kautz Typology
Each level corresponds to a distinct point in the Kautz (2022) neuro-symbolic integration spectrum:

| Type | Integration point | Example here |
|------|------------------|-------------|
| 1 | Symbolic layer applied post-hoc to neural output | Level 1, Level 2 |
| 2 | Neural output converted to typed symbol; symbolic layer operates on symbol only | Level 3 |
| 3 | Symbolic components active at inference as correctors | Level 3.5 |
| 4 | Symbolic knowledge shapes model at training time only | Level 4 |
| 4.5 | Runtime symbolic filters, external to architecture | Level 4.5 |
| 5 | Symbolic rules compiled into differentiable architecture | Level 5 |
| 6 (ext.) | Self-modifying symbolic substrate on top of Type 5 | Level 6 |

### Intent vs Decision State (Level 1+)
- **Intent**: What the user wants (classification task)
- **Decision State**: What the system decides to do (routing/safety task)

```
User says: "restart nginx on host123"
→ Intent:          execution   (user wants to run a command)
→ Decision State:  blocked     (system won't allow — confidence too low)
```

### Design Principles
- Each level is a self-contained, independently runnable unit
- Symbolic and neural components have strict, tested boundaries
- Every decision above Level 0 includes a human-readable explanation
- No level retroactively modifies an earlier level's behavior

---

**NSAI POC** | Progressive Neuro-Symbolic AI Architecture  
*From Pure Statistics to Self-Evolving Symbolic Reasoning*
