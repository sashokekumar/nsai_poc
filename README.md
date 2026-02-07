# NSAI POC: Neuro-Symbolic AI for Intent Classification

## Overview
This repository demonstrates a **progressive evolution from Level 0 to Level 6** of Neuro-Symbolic AI (NSAI) architectures for Natural Language Understanding (NLU), specifically intent classification.

**Goal**: Show how pure statistical models (Level 0) can be enhanced with symbolic reasoning, rules, and decision logic to create more robust, explainable, and safe NLU systems.

## Dataset
- **File**: `data/intents_base.csv`
- **Size**: 614 labeled utterances
- **Intent Classes**: 
  - `investigate` (149 samples)
  - `execution` (150 samples)
  - `summarization` (146 samples)
  - `out_of_scope` (169 samples)
- **Format**: Pure text utterances with intent labels (no timestamps or metadata)

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

**Python Version**: 3.13.9  
**Key Dependencies**: pandas, numpy, scikit-learn, jupyter, matplotlib, seaborn

## Architecture Progression

### 📊 [Level 0: Baseline Statistical Classifier](level0/README.md)
**Approach**: Pure Machine Learning  
**Algorithm**: TF-IDF + Logistic Regression  
**Features**:
- Confidence thresholding (0.7)
- Abstain logic for low-confidence predictions
- Token-level evidence extraction
- ~91% accuracy

**Key Output**:
```python
{
  'predicted_intent': 'execution',
  'confidence': 0.92,
  'evidence_tokens': [...]
}
```

[→ View Level 0 Details](level0/README.md)

---

### 🧠 [Level 1: Neuro-Symbolic Decision Layer](level1/README.md)
**Approach**: 2-Layer Architecture (Statistical + Symbolic)  
**Innovation**: Separates **intent** (what user wants) from **decision state** (what system does)

**Layer A - Statistical**: TF-IDF + Logistic Regression  
**Layer B - Symbolic**: 4 deterministic rules with strict precedence

**Rule System**:
- **R1** (Priority 100): Quality gate - blocks low-token inputs
- **R4** (Priority 90): Safety gate - blocks risky executions
- **R2** (Priority 50): Confidence gate - flags ambiguous predictions
- **R3** (Priority 40): Margin gate - detects close predictions

**Key Features**:
- Explicit rule priority (non-order-dependent)
- Rule categories: Quality, Safety, Ambiguity
- Decision states: `accepted`, `needs_clarification`, `blocked`
- Structured JSON output with full explainability

Note: Level1B now emits a structured `decision_trace` per utterance (for L3 path explainability) and provides a reusable `level1b_model.py` module. This trace includes `detectors_fired`, `hard_rules_failed`, `soft_rules_passed`, and `alternatives_eliminated` to support auditability.

**Key Output**:
```json
{
  "predicted_intent": "execution",
  "decision_state": "blocked",
  "decision_reason": "execution_safety_block",
  "triggered_rules": [{"rule_id": "R4", "category": "safety", ...}]
}
```

[→ View Level 1 Details](level1/README.md)

---

### 🔮 Level 2-6: Future Enhancements
**Status**: Not yet implemented

Planned features:
- **Level 2**: Multi-utterance context awareness
- **Level 3**: Adaptive thresholds and learned rules
- **Level 4**: Temporal reasoning and session state
- **Level 5**: LLM integration with symbolic grounding
- **Level 6**: Full neuro-symbolic fusion with feedback loops

## Project Structure
```
nsai_poc/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── intents_base.csv              # Shared training dataset
├── level0/
│   ├── README.md                     # Level 0 documentation
│   ├── level0_tfidf_classification.ipynb
│   ├── models/                       # Saved models
│   └── artifacts/                    # Outputs
├── level1/
│   ├── README.md                     # Level 1 documentation
│   ├── level1_tfidf_classification.ipynb
│   └── artifacts/                    # Outputs
├── validation/
│   ├── README.md                     # Validation documentation
│   └── validation.ipynb              # Multi-level model comparison
├── level2/                           # (TBD)
├── level3/                           # (TBD)
├── level4/                           # (TBD)
├── level5/                           # (TBD)
└── level6/                           # (TBD)
```

## Quick Start

### Run Level 0 (Baseline)
```bash
cd level0
jupyter notebook level0_tfidf_classification.ipynb
# Run all cells sequentially
```

### Run Level 1 (Neuro-Symbolic)
```bash
cd level1
jupyter notebook level1_tfidf_classification.ipynb
# Run all cells sequentially
```

### Compare Models (Validation)
```bash
cd validation
jupyter notebook validation.ipynb
# Run all cells to see side-by-side comparison
```

Note: `validation_results.csv` includes a `l1b_decision_trace` JSON column with the full decision path; validation outputs are not tracked in Git.

## Key Concepts

### Intent vs Decision State (Level 1+)
- **Intent**: What the user wants (classification task)
- **Decision State**: What the system decides to do (routing task)

**Example**:
```
User says: "restart nginx on host123"
Intent: execution (user wants to execute a command)
Decision State: blocked (system won't allow it due to low confidence)
```

### Rule Priority (Level 1+)
Rules are **not** evaluated sequentially. They follow explicit precedence:
1. **Quality gates** (R1) - highest priority
2. **Safety gates** (R4) - second priority
3. **Ambiguity gates** (R2, R3) - lowest priority

High-priority rules use **early returns** to prevent lower-priority overrides.

### Explainability
Every decision includes:
- **Signals**: Probabilities, confidence, margin, token evidence
- **Triggered Rules**: Which rules fired and why
- **Decision Reason**: Human-readable explanation
- **Transparent Voting**: Additional scoring context

## Design Principles
1. ✅ **No LLMs** (at lower levels) - pure statistical + symbolic
2. ✅ **No Timestamps** - text-only classification
3. ✅ **Explainable** - full transparency into decisions
4. ✅ **Deterministic Rules** - predictable, testable behavior
5. ✅ **Progressive Complexity** - each level builds on previous
6. ✅ **Single Notebook per Level** - self-contained, executable

## Documentation
- [Level 0 README](level0/README.md) - Baseline statistical classifier
- [Level 1 README](level1/README.md) - Neuro-symbolic decision layer
- [Validation README](validation/README.md) - Model comparison & analysis
- [Root README](README.md) - This file

## Contributing
This is a proof-of-concept demonstration repository. Each level is implemented as a standalone Jupyter notebook with complete documentation.

## Status
- ✅ **Level 0**: Complete (~91% accuracy)
- ✅ **Level 1**: Complete (2-layer neuro-symbolic)
- ⏳ **Level 2-6**: Planned

---

**NSAI POC** | Progressive Neuro-Symbolic AI Architecture  
*From Pure Statistics to Symbolic Reasoning*
