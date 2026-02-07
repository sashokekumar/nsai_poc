# Level 1: Neuro-Symbolic Intent Classification

## Overview
Level 1 introduces a **2-layer neuro-symbolic architecture** that combines statistical learning with deterministic symbolic rules. This level separates **what the user intends** (predicted_intent) from **what the system decides to do** (decision_state).

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
- `level1_tfidf_classification.ipynb`: Main notebook (15 cells)
- `artifacts/`: Output files and visualizations

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

## Limitations
- No temporal reasoning (no timestamps)
- No multi-utterance context
- No adaptive thresholds
- No LLM integration
- Fixed rule set (not learned)

## Next Level
→ **Level 2**: TBD (not yet implemented)

---

**Level 1 Status**: ✅ Complete  
**Architecture**: 2-Layer Neuro-Symbolic  
**Approach**: Statistical Learning + Deterministic Rules  
**Key Innovation**: Intent/Decision State Separation
