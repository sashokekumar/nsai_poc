# Level 1: Neuro-Symbolic Intent Classification

## Overview
Level 1 introduces a **2-layer neuro-symbolic architecture** that combines statistical learning with deterministic symbolic rules. This level separates **what the user intends** (predicted_intent) from **what the system decides to do** (decision_state).

**Two Variants:**
- **Level 1A**: Single multi-class classifier with rule governance
- **Level 1B**: Multi-detector architecture (one binary classifier per intent)

## Architecture

### Layer A: Statistical Model
- **Algorithm**: TF-IDF + Logistic Regression
- **Features**: 5000 TF-IDF features with 1-2 grams
- **Output**: Probability distribution, confidence scores, margins, token evidence

### Layer B: Neuro-Symbolic Decision Layer
- **Rule-Based Gates**: 4 deterministic rules with strict precedence
- **Categories**: Quality, Safety, Ambiguity
- **Decision States**: `accepted`, `needs_clarification`, `blocked`
- **Priority System**: Explicit non-order-dependent precedence

## Dataset
- **Source**: `../data/intents_base.csv`
- **Records**: 614 utterances
- **Intent Classes**: `investigate`, `execution`, `summarization`, `out_of_scope`
- **No Timestamps**: Pure text classification

## Configuration Constants
```python
CONFIG = {
    'BASE_MIN_CONF': 0.60,           # Minimum confidence threshold
    'MIN_MARGIN': 0.10,              # Margin between top-2 predictions
    'EXECUTION_MIN_CONF': 0.85,      # Higher bar for execution
    'MIN_TOKENS_OUT_OF_SCOPE': 3,    # Token count gate
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2
}

RULE_PRIORITY = ["R1", "R4", "R2", "R3"]  # Explicit precedence
```

## Rule System

### R1: Quality Gate (Priority 100)
- **Category**: Quality
- **Condition**: `meaningful_tokens < 3`
- **Action**: Set `decision_state = 'blocked'`
- **Reason**: `insufficient_tokens`
- **Behavior**: Blocks all low-token inputs immediately

### R4: Safety Gate (Priority 90)
- **Category**: Safety
- **Condition**: `predicted_intent == 'execution' AND max_confidence < 0.85`
- **Action**: 
  - If `confidence >= 0.60` → `decision_state = 'blocked'` (execution_safety_block)
  - If `confidence < 0.60` → `decision_state = 'needs_clarification'` (execution_low_confidence)
- **Behavior**: Prevents risky execution commands

### R2: Confidence Gate (Priority 50)
- **Category**: Ambiguity
- **Condition**: `max_confidence < 0.60`
- **Action**: Set `decision_state = 'needs_clarification'`
- **Reason**: `ambiguous_prediction`

### R3: Margin Gate (Priority 40)
- **Category**: Ambiguity
- **Condition**: `margin < 0.10`
- **Action**: Set `decision_state = 'needs_clarification'`
- **Reason**: `ambiguous_prediction`

## Key Innovation: Intent vs Decision State

### Previous Approach (Level 0)
- Single output: predicted intent
- No distinction between classification and routing

### Level 1 Approach
```python
{
  "predicted_intent": "execution",      # What user wants (from model)
  "decision_state": "blocked",          # What system decides
  "decision_reason": "execution_safety_block"
}
```

**Intent** = Classification (always from model)  
**Decision State** = Routing/Action (controlled by rules)

## Output Structure
```json
{
  "utterance": "restart nginx on host123",
  "predicted_intent": "execution",
  "signals": {
    "max_confidence": 0.78,
    "margin": 0.25,
    "meaningful_tokens": 5,
    "probabilities": {
      "execution": 0.78,
      "investigate": 0.15,
      "summarization": 0.05,
      "out_of_scope": 0.02
    }
  },
  "triggered_rules": [
    {
      "rule_id": "R4",
      "category": "safety",
      "priority": 90,
      "condition": "predicted_intent==execution AND max_confidence < 0.85",
      "value": 0.78
    }
  ],
  "decision_state": "blocked",
  "decision_reason": "execution_safety_block"
}
```

## Files

### Level 1A (Multi-Class Architecture)
- `level1_tfidf_classification.ipynb`: Main notebook (11 cells)
- `level1_model.py`: Shared module with CONFIG, signal extraction, decision rules, voting
- `artifacts/level1a/`: Output files and visualizations

#### Examples (Level 1A)

These examples contrast model outputs with the symbolic rule decisions.

```json
{
  "utterance": "restart nginx on host123",
  "predicted_intent": "execution",
  "signals": {"max_confidence": 0.78, "margin": 0.25, "meaningful_tokens": 5},
  "triggered_rules": ["R4"],
  "decision_state": "blocked",
  "decision_reason": "execution_safety_block"
}
```

```json
{
  "utterance": "hello",
  "predicted_intent": "out_of_scope",
  "signals": {"meaningful_tokens": 1},
  "triggered_rules": ["R1"],
  "decision_state": "blocked",
  "decision_reason": "insufficient_tokens"
}
```

```json
{
  "utterance": "why is server cpu high",
  "predicted_intent": "investigate",
  "signals": {"max_confidence": 0.92, "margin": 0.45, "meaningful_tokens": 6},
  "triggered_rules": [],
  "decision_state": "accepted",
  "decision_reason": "model_prediction"
}
```

### Level 1B (Multi-Detector Architecture)
- `level1b_multi_detector_classification.ipynb`: Multi-detector notebook
- `level1b_model.py`: Multi-detector module with binary classifiers + rule engine
- `models/level1b/`: Trained detectors (one per intent)
- `artifacts/level1b/`: Predictions, metrics, rule analysis

#### Examples (Level 1B)

```json
{
  "utterance": "restart nginx on host123",
  "detector_scores": {"execution": 0.65, "investigate": 0.40},
  "triggered_rules": ["R_EXEC_SAFETY"],
  "decision_state": "needs_clarification",
  "decision_reason": "R_EXEC_SAFETY"
}
```

```json
{
  "utterance": "server issues",
  "detector_scores": {"investigate": 0.55, "execution": 0.52},
  "triggered_rules": ["R_AMBIGUOUS"],
  "decision_state": "needs_clarification",
  "decision_reason": "ambiguous_detectors"
}
```

```json
{
  "utterance": "summarize last error logs",
  "detector_scores": {"summarization": 0.88, "investigate": 0.05},
  "triggered_rules": ["R_DEFAULT"],
  "decision_state": "accepted",
  "decision_reason": "R_DEFAULT"
}
```

**Note**: Both notebooks import from their respective modules to avoid code duplication.

## Notebook Structure

### Part 1: Statistical Model Training (Cells 1-9)
1. Title and architecture description
2. Import dependencies
3. Load and validate data
4. Train/test split (stratified)
5. Train TF-IDF + LR pipeline
6. Evaluate performance
7. Token-level evidence extraction
8. Sanity checks

### Part 2: NSAI Symbolic Decision Layer (Cells 10-15)
10. Configuration constants (CONFIG + RULE_PRIORITY)
11. Signal extraction (probabilities, confidence, margin, tokens)
12. Decision rules with strict precedence
13. Transparent voting (for explainability only)
14. Demo with explainable inference

## Usage
```bash
# Navigate to level1 directory
cd level1

# Open notebook
jupyter notebook level1_tfidf_classification.ipynb

# Or use VS Code
code level1_tfidf_classification.ipynb
```

## Execution Flow
1. **Extract Signals** (Layer A): Model produces probabilities, confidence, margin, token evidence
2. **Apply Rules** (Layer B): Rules check conditions in priority order (R1 → R4 → R2 → R3)
3. **Early Returns**: High-priority rules (R1, R4) override all lower-priority rules
4. **Transparent Voting**: Computed for explainability only; does NOT control final decision
5. **Structured Output**: JSON with intent, decision state, reason, triggered rules

## Example Scenarios

### Scenario 1: High-Confidence Execution (Blocked by R4)
```
Input: "restart nginx on host123"
predicted_intent: "execution"
max_confidence: 0.78
decision_state: "blocked"
decision_reason: "execution_safety_block"
triggered_rules: [R4]
```

### Scenario 2: Low-Token Input (Blocked by R1)
```
Input: "hello"
predicted_intent: "out_of_scope"
meaningful_tokens: 1
decision_state: "blocked"
decision_reason: "insufficient_tokens"
triggered_rules: [R1]
```

### Scenario 3: Ambiguous Prediction (R2 + R3)
```
Input: "server issues"
predicted_intent: "investigate"
max_confidence: 0.55
margin: 0.08
decision_state: "needs_clarification"
decision_reason: "ambiguous_prediction"
triggered_rules: [R2, R3]
```

### Scenario 4: Clean Prediction (No Rules)
```
Input: "why is server cpu high"
predicted_intent: "investigate"
max_confidence: 0.92
margin: 0.45
decision_state: "accepted"
decision_reason: "model_prediction"
triggered_rules: []
```

## Key Features
1. **Strict Precedence**: R1 > R4 > R2 > R3 (no order-dependent conflicts)
2. **Explicit Categories**: Quality, Safety, Ambiguity
3. **Early Returns**: High-priority rules block lower-priority evaluation
4. **Transparent Voting**: Shows scoring but doesn't control decisions
5. **Structured JSON**: Full explainability with signals + rules + reasons

## Improvements Over Level 0
- ✅ Separates intent from decision state
- ✅ Adds execution safety gates
- ✅ Implements quality checks (token count)
- ✅ Detects ambiguity (confidence + margin)
- ✅ Explicit rule priority (no execution order dependency)
- ✅ Structured, explainable output
---

## Level 1B: Multi-Detector Architecture

### Key Differences from Level 1A

**Level 1A**: Single multi-class classifier  
**Level 1B**: One binary detector per intent

### Architecture

#### Multi-Detector Setup
```python
# Each intent gets its own binary detector
detectors = {
    'investigate': BinaryClassifier,    # "Is this investigate?"
    'execution': BinaryClassifier,      # "Is this execution?"
    'summarization': BinaryClassifier,  # "Is this summarization?"
    'out_of_scope': BinaryClassifier    # "Is this out_of_scope?"
}
```

**Critical Properties:**
- Scores are **independent** (do NOT sum to 1)
- Each detector answers: "Does this match MY intent?"
- Multiple detectors can be confident simultaneously
- Rules (not models) make final decision

### Rule System (6 Rules)

#### R_LOW_TOKEN_COUNT (Priority 100) - Quality
- **Condition**: `meaningful_tokens < 3`
- **Action**: `decision_state = 'blocked'`, `predicted_intent = 'out_of_scope'`
- **Type**: `quality`

#### R_NO_CONFIDENT_DETECTOR (Priority 95) - Quality
- **Condition**: `all detector_scores < 0.50`
- **Action**: `decision_state = 'blocked'`, `predicted_intent = 'out_of_scope'`
- **Type**: `quality`
- **Semantic Meaning**: No detector is confident enough to be trusted

#### R_EXEC_SAFETY (Priority 90) - Safety
- **Condition**: `execution_score >= 0.50 AND < 0.85`
- **Action**: 
  - If `investigate_score >= 0.50` → `predicted_intent = 'investigate'`, `accepted`
  - Else → `predicted_intent = 'execution'`, `needs_clarification`
- **Type**: `safety`

#### R_MULTI_DETECTOR_CONCURRENCE (Priority 60) - Compound Intent
- **Condition**: `>= 2 detectors with score >= 0.70`
- **Action**: Log for L2 (reserved, not blocking in L1B)
- **Type**: `compound_intent`
- **Example**: investigate=0.72, summarization=0.70
- **Note**: Reserved for Level 2 orchestration

#### R_AMBIGUOUS (Priority 50) - Ambiguity
- **Condition**: `>= 2 detectors >= 0.50 AND margin < 0.10`
- **Action**: `decision_state = 'needs_clarification'`
- **Type**: `ambiguity`

#### R_DEFAULT (Priority 10) - Default
- **Condition**: Always
- **Action**: Select highest scoring detector
- **Type**: `default`

### Configuration
```python
BASE_MIN_SCORE = 0.50          # Minimum confidence threshold
AMBIGUITY_MARGIN = 0.10        # Margin for ambiguity detection
EXECUTION_MIN_SCORE = 0.85     # Higher bar for execution
MIN_TOKENS_OUT_OF_SCOPE = 3    # Token count gate
CONCURRENCE_THRESHOLD = 0.70   # Multi-detector concurrence
```

### Output Structure
```json
{
  "predicted_intent": "investigate",
  "decision_state": "accepted",
  "decision_reason": "R_DEFAULT",
  "detector_scores": {
    "investigate": 0.89,
    "execution": 0.12,
    "summarization": 0.05,
    "out_of_scope": 0.03
  },
  "top_detector": "investigate",
  "score_margin": 0.77,
  "meaningful_tokens": 6,
  "triggered_rules": [
    {
      "rule_id": "R_DEFAULT",
      "rule_type": "default",
      "priority": 10,
      "condition": "default to highest scoring detector",
      "value": 0.89
    }
  ]
}
```

Additional explainability (Level1B):
- `decision_trace` (returned per-utterance) contains a structured proof of the decision path with fields:
  - `detectors_fired`: intents with score >= BASE_MIN_SCORE
  - `hard_rules_failed`: list of hard rule ids that caused rejection/blocking
  - `soft_rules_passed`: list of non-blocking rules that influenced confidence
  - `alternatives_eliminated`: mapping of eliminated alternatives to the reason

This enables L3-style path explainability (not just final justification) and supports counterfactual analysis ("why not X?").

### Usage Example
```python
from level1b_model import Level1BClassifier

# Load trained model
classifier = Level1BClassifier.load('models/level1b')

# Predict
result = classifier.predict("why is server cpu high")

print(f"Intent: {result['predicted_intent']}")
print(f"Decision: {result['decision_state']}")
print(f"Reason: {result['decision_reason']}")
print(f"Detector Scores: {result['detector_scores']}")
```

### Artifacts Generated
1. **level1b_predictions.csv** - Full test set results with detector scores
2. **level1b_predictions.jsonl** - JSONL format for programmatic access
3. **evaluation_metrics.json** - Accuracy, rule stats, configuration
4. **rule_activation_stats.csv** - Per-rule trigger analysis
5. **decision_state_breakdown.csv** - State distribution by intent

### Key Features
1. **Independent Detector Scores**: Not constrained to sum to 1
2. **Explicit Rule Types**: quality, safety, ambiguity, compound_intent, default
3. **Compound Intent Detection**: Reserved for L2 orchestration
4. **Semantic Rule Separation**: Low tokens ≠ No confident detector
5. **Full Explainability**: All signals, rules, and decisions tracked

### Why Multi-Detector Architecture?

**Advantages:**
- Can detect compound/mixed intents (multiple high scores)
- More explainable (each detector is interpretable)
- Better handles edge cases (nothing confident vs. low tokens)
- Prepares for L2 orchestration (multi-intent workflows)

**Trade-offs:**
- More models to train/maintain (4 binary vs. 1 multi-class)
- Slightly higher inference cost (4 predictions vs. 1)
- Requires rule governance (scores don't directly map to decision)

---

## Level 1C: Symbol-Aligned Neuro-Symbolic Classification

Level 1C refines Level 1B by introducing a strict three-layer separation: numeric scores never leave the neural layer, a symbolization layer converts all signals into predicates, and the rule engine operates exclusively on those predicates with no numeric comparisons in rules.

### Architecture

```
Neural Detectors → Symbolization Layer → Pure Symbolic Rule Engine
```

#### Neural Layer (numeric, internal only)
- Four independent binary TF-IDF + Logistic Regression detectors, one per intent
- Scores are **not** constrained to sum to 1
- Also computes the number of active TF-IDF features (model token count) for input-quality evidence

#### Symbolization Layer (all thresholds live here)
Converts numeric signals into symbolic predicates:

| Predicate | Condition |
|---|---|
| `CANDIDATE_<INTENT>` | detector score ≥ `BASE_MIN_SCORE` (0.30) |
| `HIGH_CONFIDENCE_<INTENT>` | detector score ≥ `HIGH_CONFIDENCE_SCORE` (0.85) |
| `NOT_HIGH_CONFIDENCE_EXECUTION` | negation, for rule convenience |
| `NO_CANDIDATE_INTENT` | no detector reached `BASE_MIN_SCORE` |
| `AMBIGUOUS` | top-to-second score margin < `AMBIGUITY_MARGIN` (0.10) |
| `RAW_TOKEN_COUNT_SUFFICIENT/INSUFFICIENT` | raw token count vs `RAW_MIN_TOKENS` (2) |
| `UNIQUE_TOKEN_COUNT_SUFFICIENT/INSUFFICIENT` | unique token count vs `UNIQUE_MIN_TOKENS` (2) |
| `MODEL_TOKEN_COUNT_SUFFICIENT/INSUFFICIENT` | active TF-IDF features vs `MODEL_MIN_TOKENS` (2) |
| `VERY_SHORT_UTTERANCE` | raw token count ≤ 1 |

The symbolization layer is **neutral** — it produces only data-derived predicates. No domain keyword lists (e.g. `OOS_KEYWORDS`, `TECH_KEYWORDS`) are used; domain-specific intent biasing is the responsibility of the detectors, not the symbolization layer.

#### Rule Engine (symbolic only, no numeric thresholds)
Rules are loaded from `models/level1c/rules.json` at runtime and operate purely on the predicate set. Rules are data — they can be inspected, versioned, and modified without changing Python code.

| Rule | Priority | Condition | Outcome |
|---|---|---|---|
| `R_INSUFFICIENT_INPUT` | 100 | ALL three token predicates insufficient | `blocked` → `out_of_scope` |
| `R_NO_CANDIDATE_INTENT` | 95 | `NO_CANDIDATE_INTENT` | `blocked` → `out_of_scope` |
| `R_EXECUTION_LOW_CONFIDENCE` | 90 | `CANDIDATE_EXECUTION` + `NOT_HIGH_CONFIDENCE_EXECUTION` | `needs_clarification` → `execution` (or downgrades to `investigate` if `CANDIDATE_INVESTIGATE`) |
| `R_AMBIGUOUS` | 50 | `AMBIGUOUS` | `needs_clarification` → top intent |
| `R_DEFAULT` | 0 | always | `accepted` → top intent |

### Files

- `level1c_model.py` — `Level1CClassifier` with neural, symbolization, and rule-engine layers; `CONFIG` dict (thresholds only)
- `level1c_symbolic_intent_classification.ipynb` — end-to-end notebook: train, spot-check, evaluate, save
- `models/level1c/config.json` — serialised `CONFIG` and intents list
- `models/level1c/rules.json` — symbolic rules as plain JSON (human-readable, no numeric values)
- `models/level1c/detector_<intent>.pkl` — one binary detector per intent (4 files)
- `artifacts/level1c/level1c_predictions.csv` — full test-set predictions including symbols and triggered rules
- `artifacts/level1c/evaluation_metrics.json` — accuracy, decision-state distribution, rule trigger counts, config snapshot

### Output Structure

```json
{
  "utterance": "restart nginx on host123",
  "symbols": ["CANDIDATE_EXECUTION", "NOT_HIGH_CONFIDENCE_EXECUTION", "RAW_TOKEN_COUNT_SUFFICIENT", "UNIQUE_TOKEN_COUNT_SUFFICIENT", "MODEL_TOKEN_COUNT_SUFFICIENT"],
  "predicted_intent": "execution",
  "decision_state": "needs_clarification",
  "decision_reason": "R_EXECUTION_LOW_CONFIDENCE",
  "triggered_rules": ["R_EXECUTION_LOW_CONFIDENCE"],
  "detector_scores": {"execution": 0.78, "investigate": 0.15, "summarization": 0.04, "out_of_scope": 0.03}
}
```

```json
{
  "utterance": "why is server cpu high",
  "symbols": ["CANDIDATE_INVESTIGATE", "HIGH_CONFIDENCE_INVESTIGATE", "RAW_TOKEN_COUNT_SUFFICIENT", "UNIQUE_TOKEN_COUNT_SUFFICIENT", "MODEL_TOKEN_COUNT_SUFFICIENT"],
  "predicted_intent": "investigate",
  "decision_state": "accepted",
  "decision_reason": "R_DEFAULT",
  "triggered_rules": ["R_DEFAULT"],
  "detector_scores": {"investigate": 0.91, "execution": 0.08, "summarization": 0.01, "out_of_scope": 0.00}
}
```

```json
{
  "utterance": "hello",
  "symbols": ["RAW_TOKEN_COUNT_INSUFFICIENT", "UNIQUE_TOKEN_COUNT_INSUFFICIENT", "MODEL_TOKEN_COUNT_INSUFFICIENT", "VERY_SHORT_UTTERANCE", "NO_CANDIDATE_INTENT"],
  "predicted_intent": "out_of_scope",
  "decision_state": "blocked",
  "decision_reason": "R_INSUFFICIENT_INPUT",
  "triggered_rules": ["R_INSUFFICIENT_INPUT"],
  "detector_scores": {"investigate": 0.10, "execution": 0.05, "summarization": 0.02, "out_of_scope": 0.20}
}
```

### Key Design Decisions

1. **Neutral symbolization layer** — predicate derivation is purely statistical; no domain keyword lists. Detectors handle domain specificity.
2. **Hybrid input quality evidence** — `R_INSUFFICIENT_INPUT` requires *all three* token signals to be insufficient, preventing false blocks on short but model-known vocabulary.
3. **Externalized rules** — `rules.json` is plain data; rules can be added or tuned without modifying `level1c_model.py`.
4. **Strict numeric/symbolic separation** — numeric thresholds appear only in `CONFIG` and only inside `_symbolize()`; the rule engine never sees a number.

---

## Next Level
→ **Level 2**: TBD (will leverage L1B's compound intent detection)

---

**Level 1 Status**: ✅ Complete (1A + 1B + 1C)  
**Architecture**: 2-Layer Neuro-Symbolic (1A/1B) → 3-Layer Symbol-Aligned (1C)  
**Approach**: Statistical Learning + Deterministic Symbolic Rules  
**Key Innovation**: Intent/Decision State Separation  
**Level 1B Innovation**: Multi-Detector Independence + Explicit Rule Types  
**Level 1C Innovation**: Strict Numeric/Symbolic Separation, Externalized Rules, Neutral Symbolization Layer  
→ **Level 2**: TBD (will leverage L1C structured predicate outputs)
