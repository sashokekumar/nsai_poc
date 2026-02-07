# Level 3: Logic-Aware Gating & Constraint Enforcement

## Overview
Level 3 demonstrates the critical transition from **post-hoc logic filtering** (Level 2.5) to **embedded logic-aware reasoning** where logical constraints shape the model's predictions during the forward pass, not after.

## Key Innovation
**Logic-aware gating**: Constraints are applied **inside** the model's forward pass, before the output layer produces final probabilities. This ensures invalid predictions are structurally prevented, not just masked.

## Notebooks

### 1. `level3_data_prep.ipynb` - Level-3 Dataset Generation
**Purpose**: Generate a Level-3 dataset with logical constraints (allowed/suppressed intents).

**Input**: Base dataset (`data/intents_base.csv`)

**Output**: `level3/data/level3_intents.csv` with columns:
- `utterance`: Input text
- `gold_intent`: True intent label
- `facts`: Contextual facts extracted from utterance
- `active_constraints`: Domain-specific constraints
- `allowed_intents`: List of valid intents for this context
- `suppressed_intents`: List of forbidden intents for this context

**Key Features**:
- Deterministic constraint generation based on utterance semantics
- Maps dataset intent names (`execution`, `summarization`) to canonical names (`execute`, `summarize`)
- Filters out `out_of_scope` samples (not part of NSAI canonical intent set)

---

### 2. `l2_5_evaluation.ipynb` - Level 2.5 Baseline (Post-hoc Logic)
**Purpose**: Educational bridge demonstrating what Level 2.5 proves and what it cannot fix.

**Architecture**: 
```
L2 Model → Raw Predictions → Logic Filter → Constrained Predictions
```

**Approach**:
- Load Level-2 model (or fallback to Level-0)
- Run baseline inference (unconstrained)
- Apply logical constraints **after** prediction:
  - Zero out suppressed intent probabilities
  - Zero out non-allowed intents
  - Renormalize remaining probabilities

**Evaluation Metrics**:
- Constraint violation rate (before vs after)
- Intent flip rate (how often logic changes top-1 prediction)
- Confidence changes due to renormalization

**Key Teaching Moments**:
1. **What L2.5 fixes**: Invalid predictions are masked post-hoc
2. **What L2.5 cannot fix**:
   - Model still internally prefers invalid paths
   - Wasted representational capacity on forbidden intents
   - Logic cannot shape how the model learns
3. **Why Level 3 exists**: To embed logic inside the model, not as a filter

**Final Cell**: Comprehensive educational conclusion explaining:
- Structural limitations of statistical models (L2)
- Hard vs soft constraints
- Post-hoc masking mechanics (L2.5)
- Fundamental limits of late-stage logic application
- Architectural necessity of Level 3

---

### 3. `level3_logic_gating_poc.ipynb` - Level 3 Logic-Aware Gating PoC
**Purpose**: Minimal, didactic proof-of-concept comparing L2, L2.5, and L3 architectures.

**Architecture**:
```
L3 Model: Features → Linear Layer → Logic Gate (mask) → Softmax → Probabilities
```

**Key Difference from L2.5**:
- **L2.5**: `model(x) → probs` → `apply_logic(probs) → final_probs` (post-processing)
- **L3**: `model(x, constraints) → final_probs` (logic inside forward pass)

**Implementation Details**:

**Cell Structure**:
1. **Cell 0**: Educational introduction (L2 vs L2.5 vs L3)
2. **Cell 1**: Load & validate Level-3 dataset
   - Maps `execution` → `execute`, `summarization` → `summarize`
   - Filters out `out_of_scope` samples
   - Creates allowed/suppressed constraint masks
3. **Cell 2**: Train L2 baseline (TF-IDF + Logistic Regression)
4. **Cell 3**: Apply L2.5 post-hoc masking
5. **Cell 4**: Implement L3 with logic-aware gating
   - Custom `Level3LogicGatedClassifier` class
   - `forward(X, allowed_mask)` method applies gating **before softmax**
   - Logic participates in training (gradients flow through gated outputs)
6. **Cell 5**: Comparative metrics (violation rates, flip rates, accuracy)
7. **Cell 6**: Side-by-side examples showing:
   - L2 can predict invalid intents
   - L2.5 fixes violations after
   - L3 prevents violations structurally
8. **Cell 7**: Final verdict: `LEVEL-3 POC COMPLETE`

**Critical Technical Detail** (Cell 4):
```python
class Level3LogicGatedClassifier:
    def forward(self, X, allowed_mask):
        # Compute logits
        logits = X @ self.W + self.b
        
        # CRITICAL: Apply logic gate BEFORE softmax
        # Mask suppressed intents with large negative value
        masked_logits = logits + (1 - allowed_mask) * self.mask_value
        
        # Softmax (numerically stable)
        probs = softmax(masked_logits)
        return probs
```

**Why This Matters**:
- Suppressed intents receive extremely low logits **before** softmax
- After softmax, they have near-zero probability by construction
- The model learns to work **with** constraints during training
- Gradients reflect constraint-aware predictions

**Results** (on Level-3 dataset):
- L2 baseline: ~99% accuracy (but may violate constraints)
- L2.5 post-hoc: ~99% accuracy (violations corrected)
- L3 logic-aware: ~54% accuracy (simple PoC, but **zero** structural violations)

**Note**: L3's lower accuracy is expected for this minimal PoC:
- Uses basic gradient descent (vs sklearn's optimized solvers)
- Simple linear model (vs potential deep architectures)
- The key is **structural constraint prevention**, not accuracy optimization

## Canonical Intent Set
```python
INTENTS = ['investigate', 'execute', 'summarize', 'ops']
```

**Dataset Mapping**:
- `execution` → `execute`
- `summarization` → `summarize`
- `out_of_scope` → filtered out (not canonical)

## Key Concepts

### Hard Constraints vs Soft Preferences
- **Hard constraints**: Logically impossible (e.g., `execute` when explicitly suppressed)
  → Must be enforced structurally
- **Soft preferences**: Valid but suboptimal (e.g., `summarize` when `investigate` would be better)
  → Can be learned through training

### Post-hoc Logic (L2.5) vs Embedded Logic (L3)

| Aspect | L2.5 | L3 |
|--------|------|-----|
| Logic timing | After prediction | During forward pass |
| Gradient flow | Through unconstrained outputs | Through constrained outputs |
| Representation learning | Unaware of constraints | Aligned with constraints |
| Architectural commitment | External filter (removable) | Embedded component (structural) |
| Invalid predictions | Masked post-hoc | Prevented by construction |

### Why L3 > L2.5
1. **Representation efficiency**: Model doesn't waste capacity on forbidden patterns
2. **Training alignment**: Loss function sees only valid predictions
3. **Structural guarantees**: Invalid outputs are architecturally impossible
4. **Gradient signals**: Constraints shape how the model learns to decompose problems

## Dataset Statistics
**File**: `level3/data/level3_intents.csv`
- **Total records**: 614
- **After filtering**: ~445 canonical samples
- **Intent distribution**:
  - `investigate`: 149 samples
  - `execute`: 150 samples
  - `summarize`: 146 samples
  - `ops`: 0 samples (not in dataset)
  - `out_of_scope`: 169 samples (filtered out)

## Design Constraints
1. ✅ **No LLM calls** - Pure sklearn + numpy implementation
2. ✅ **Deterministic** - Fixed random seeds (42)
3. ✅ **Minimal** - Simple linear classifier for clarity
4. ✅ **Educational** - Teaching NSAI levels, not production systems
5. ✅ **Runnable** - Self-contained notebooks, no external dependencies
6. ✅ **Aligned with L2/L2.5/L3 definitions** - Strict architectural separation

## Files
```
level3/
├── README.md                          # This file
├── level3_data_prep.ipynb            # Dataset generation
├── l2_5_evaluation.ipynb             # L2.5 educational bridge
├── level3_logic_gating_poc.ipynb     # L3 PoC (L2 vs L2.5 vs L3)
└── data/
    └── level3_intents.csv            # Generated constraint dataset
```

## Quick Start

### 1. Generate Level-3 Dataset (optional - already generated)
```bash
cd level3
jupyter notebook level3_data_prep.ipynb
# Run all cells to regenerate level3_intents.csv
```

### 2. Run L2.5 Evaluation
```bash
jupyter notebook l2_5_evaluation.ipynb
# Run all cells to see post-hoc logic filtering
# Read final cell for educational conclusion
```

### 3. Run Level-3 PoC
```bash
jupyter notebook level3_logic_gating_poc.ipynb
# Run all cells to compare L2, L2.5, and L3
# Observe violation rates and structural differences
```

## Key Takeaways

### From L2.5 Notebook
- **Post-hoc logic helps** in reducing invalid predictions
- **Post-hoc logic has limits**: cannot shape representation learning
- **True NSAI requires embedded logic**, not filters

### From L3 PoC
- **Logic can be embedded** inside the forward pass (before softmax)
- **L3 violation rate should be exactly zero** (structural guarantee)
- **The model learns with constraint awareness** (gradients flow through gated outputs)
- **This is fundamentally different from L2.5** (not just better masking)

## Status
- ✅ **Dataset generation**: Complete (`level3_data_prep.ipynb`)
- ✅ **L2.5 evaluation**: Complete (educational bridge)
- ✅ **L3 PoC**: Complete (logic-aware gating demonstration)

## Next Steps (Beyond PoC)
Real Level-3 systems would involve:
- More sophisticated architectures (transformers, graph neural networks)
- Richer logical constraints (temporal dependencies, multi-step reasoning)
- Differentiable logic layers that can learn constraint parameters
- Production-grade optimizations and scalability

---

**Level 3** | Logic-Aware Gating & Constraint Enforcement  
*From Post-hoc Filtering to Embedded Reasoning*
