# Validation: Model Comparison & Analysis

## Overview
This folder contains validation notebooks that compare different NSAI levels to identify improvements, disagreements, and ambiguity handling across models. As new levels are implemented, this notebook will expand to include multi-level comparisons.

## Files

### [validation.ipynb](validation.ipynb)
**Purpose**: Multi-level model comparison (currently Level 0 vs Level 1, expandable for Level 2+)

**Key Features**:
- Sends identical test utterances to both models
- Compares predictions, confidence scores, and decision logic
- Identifies intent disagreements between models
- Analyzes ambiguity detection differences
- Tracks rule triggers in Level 1
- Exports comparison results to CSV
- **Imports from `level1/level1_model.py`** - no code duplication

**Test Coverage**:
- ✅ High-confidence predictions
- ✅ Execution commands (safety gates)
- ✅ Low-token inputs (quality gates)
- ✅ Ambiguous utterances
- ✅ Edge cases

**Output**:
- Side-by-side comparison display
- Disagreement analysis
- Rule trigger statistics
- Exportable CSV: `validation_results.csv`

**Expandable**: When Level 2+ are implemented, this notebook will include multi-level comparisons.

## Usage

### Run Comparison Notebook
```bash
cd validation
jupyter notebook validation.ipynb
# Run all cells sequentially
```

### Expected Outputs
1. **Console Output**: Detailed side-by-side comparisons for each test utterance
2. **Summary Statistics**: Disagreements, abstains, blocks, clarifications
3. **CSV Export**: `validation_results.csv` with all results

## Key Metrics

### Agreement Analysis
- **Intent Agreement Rate**: Percentage of cases where both models predict the same intent
- **Disagreements**: Cases where Level 0 and Level 1 predict different intents

### Ambiguity Detection
- **Level 0 Abstains**: Cases where confidence < 0.7
- **Level 1 Clarifications**: Cases flagged by ambiguity rules (R2, R3)
- **Level 1 Blocks**: Cases blocked by quality (R1) or safety (R4) rules

### Rule Effectiveness
- **R1 Triggers**: Quality gates (insufficient tokens)
- **R4 Triggers**: Safety gates (execution risk)
- **R2 Triggers**: Low confidence ambiguity
- **R3 Triggers**: Low margin ambiguity

## Example Comparison Output

```
==================================================================================
UTTERANCE: restart nginx on host123
==================================================================================

LEVEL 0 (Baseline)                         | LEVEL 1 (Neuro-Symbolic)
----------------------------------------------------------------------------------
Predicted Intent: execution                | Predicted Intent: execution
Final Decision: execution                  | Decision State: blocked
Confidence: 0.7834                         | Confidence: 0.7834
Abstain: False                             | Margin: 0.2541
—                                          | Tokens: 5
—                                          | Triggered Rules: R4
—                                          | Decision Reason: execution_safety_block

----------------------------------------------------------------------------------
AMBIGUITY ANALYSIS:
  Intent Agreement: ✓ YES
  Level 0 Ambiguity: ✓ Confident
  Level 1 Ambiguity: 🚫 BLOCKED (execution_safety_block)
```

## Key Insights

### 1. Model Agreement
Both models share the same statistical base (TF-IDF + LogisticRegression), so **predicted intents** should typically match. Disagreements indicate:
- Potential randomness in model training
- Edge cases near decision boundaries
- Need for further investigation

### 2. Decision Logic Differences
- **Level 0**: Simple confidence threshold (0.7)
  - Above threshold → accept
  - Below threshold → abstain
- **Level 1**: Multi-criteria rule system
  - Quality gates (token count)
  - Safety gates (execution risk)
  - Ambiguity gates (confidence + margin)

### 3. Level 1 Advantages
- ✅ **Safety**: Blocks risky execution commands even with moderate confidence
- ✅ **Quality**: Rejects low-quality inputs (e.g., "hi", "ok")
- ✅ **Explainability**: Shows which rules triggered and why
- ✅ **Flexibility**: Separates intent from decision state
- ✅ **Margin Detection**: Uses prediction margin in addition to confidence

### 4. Ambiguity Handling
Level 1 is **more conservative** than Level 0:
- Catches more ambiguous cases (confidence + margin)
- Blocks execution commands that Level 0 would accept
- Provides structured reasons for ambiguity

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load identical training data                            │
│    └─ data/intents_base.csv                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Train both models                                        │
│    ├─ Level 0: TF-IDF + LR (confidence threshold)          │
│    └─ Level 1: TF-IDF + LR + Rules                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Send identical test utterances                          │
│    └─ Same inputs to both models                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Compare outputs                                          │
│    ├─ Predicted intents                                    │
│    ├─ Confidence scores                                    │
│    ├─ Decision states                                      │
│    └─ Triggered rules (Level 1)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Analyze differences                                      │
│    ├─ Intent disagreements                                 │
│    ├─ Ambiguity detection                                  │
│    ├─ Rule effectiveness                                   │
│    └─ Export to CSV                                        │
└─────────────────────────────────────────────────────────────┘
```

## Future Validation

### Planned Comparisons
- **Level 1 vs Level 2**: When Level 2 is implemented
- **Multi-level comparison**: All levels side-by-side
- **Production data validation**: Test with real-world utterances

### Suggested Analyses
- Confusion matrix comparison
- Confidence distribution plots
- Rule co-occurrence analysis
- False positive/negative tracking

---

**Validation Status**: ✅ Level 0 vs Level 1 Complete  
**Next**: Await Level 2 implementation for multi-level comparison
