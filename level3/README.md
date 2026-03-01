# Level 3 — Neural Intent Classifier (Neuro-Symbolic Pipeline)

## Overview

Level 3 implements a **lightweight neural intent classifier** trained with PyTorch and wired into a full neuro-symbolic pipeline:

```
Utterance → Tokenizer → Encoder → NeuralIntentModel → SymbolEmitter → RuleEngine → Decision
```

The neural model learns intent embeddings; the symbolic layer (emitter + rule engine) converts predictions into structured, auditable decisions. This is the architectural step beyond Level 2's TF-IDF baseline — logic is now an integral part of the pipeline, not a bolt-on filter.

---

## Notebook

### `level3_intent_classifier.ipynb`

End-to-end training and evaluation notebook. Run all cells top-to-bottom to go from raw CSV to a saved model and a working inference pipeline.

| Cell | Purpose |
|------|---------|
| 1 | Imports — loads all level3 modules; purges `sys.modules` cache so edits to `.py` files are always picked up without a kernel restart |
| 2 | Load dataset (`data/intents_base.csv`) |
| 3 | Label mapping — maps intent strings to integer indices |
| 4 | Tokenizer — lowercase, strip punctuation, whitespace split |
| 5 | Build vocabulary — frequency-ordered word index, index 0 = `<PAD>` |
| 6 | Sentence encoder — pads/truncates to `MAX_LEN=12` |
| 7 | `IntentDataset` + `DataLoader` |
| 8 | Initialise `NeuralIntentModel` (embedding → mean-pool → linear) |
| 9 | Training loop — 40 epochs, Adam, cross-entropy loss |
| 10 | Save model weights to `level3_intent_model.pt` |
| 11 | Load `Level3Pipeline` from saved weights |
| 12 | Inference helper — encodes text, runs full pipeline |
| 13 | Test suite — four canonical examples with pass/fail reporting |

**Quick start:**
```bash
cd level3
jupyter notebook level3_intent_classifier.ipynb
```

---

## Module Architecture

```
level3/
├── intent_constants.py        # Canonical intent name constants + ALLOWED_INTENTS set
├── symbol_schema.py           # IntentSymbol dataclass
├── symbol_emitter.py          # Maps a predicted class index → IntentSymbol
├── neural_intent_model.py     # nn.Embedding + mean-pool + Linear classifier
├── rule_engine.py             # Symbolic rule table: intent → action + requires_approval
├── level3_pipeline.py         # Chains model → emitter → rule engine
└── level3_intent_classifier.ipynb  # Training & evaluation notebook
```

### Dependency Order

```
intent_constants
    └── symbol_schema
            └── symbol_emitter
    └── rule_engine
    └── neural_intent_model
            └── level3_pipeline  (imports all of the above)
```

---

## Intents

| Intent | Index | Rule-Engine Action | Requires Approval |
|--------|-------|--------------------|-------------------|
| `summarization` | 0 | `call_summarization_pipeline` | No |
| `execution` | 1 | `route_to_executor` | **Yes** |
| `investigate` | 2 | `trigger_diagnostics` | No |
| `out_of_scope` | 3 | `return_safe_fallback` | No |

---

## Generated Artifact

| File | Description | Committed? |
|------|-------------|------------|
| `level3_intent_model.pt` | Trained PyTorch weights | **No** — in `.gitignore` |

The `.pt` file is produced by Cell 10 and consumed by Cell 11. It is excluded from version control because it is a large binary artifact that can be fully reproduced by running the notebook.

---

## Design Principles

1. **No LLM calls** — pure PyTorch + standard library
2. **Deterministic** — vocabulary built in frequency order; no random seeds needed for inference
3. **No kernel restart required** — Cell 1 purges `sys.modules` so any edit to a `.py` file is immediately reflected on the next cell run
4. **Auditable** — every prediction produces a structured `{symbol, decision}` dict traceable through the rule engine
5. **Minimal** — the model is intentionally simple (embedding + mean-pool + linear) to keep the focus on the neuro-symbolic wiring, not model complexity

---

## Comparison with Other Levels

| Level | Classifier | Logic | Decision Layer |
|-------|-----------|-------|----------------|
| 0 | TF-IDF + LogReg | None | Raw label |
| 1 | Rule-based | Symbolic rules | Structured output |
| 2 | TF-IDF + LogReg | Post-hoc filter | Constrained label |
| 2.5 | TF-IDF + LogReg | Post-hoc masking | Renormalised probs |
| **3** | **Neural (PyTorch)** | **Embedded in pipeline** | **Symbol + Rule action** |

---

**Level 3** | Neural Intent Classifier (Neuro-Symbolic Pipeline)
