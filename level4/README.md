# Level 4: Ontology-Grounded Neuro-Symbolic Intent Pipeline

## Overview

Level 4 implements a **modular neuro-symbolic pipeline** for intent understanding, reasoning, and response generation in SRE/DevOps domains. The system combines ML-based intent prediction, ontology-driven symbolic parsing, deterministic reasoning, and stepwise planning. All vocabularies and mappings are centralized in an ontology, and every stage is auditable and extensible.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 – Semantic Parser (semantic_parser.py)                     │
│  - IntentClassifier (ML, intent_model.py)                           │
│  - EntityMatcher (symbolic+TFIDF, entity_matcher.py)                │
│  - Symptom/Time extractors (pattern, ontology.py)                   │
│  - Domain guard: out_of_scope fallback                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  IntentFrame (intent, entity, symptom, time)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2 – Reasoner (reasoner.py)                                   │
│  - Symbolic rules for mode selection, context enrichment            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  reasoning dict
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3 – Planner (planner.py)                                     │
│  - Converts reasoning into stepwise plan                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  plan dict
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4 – Responder (responder.py)                                 │
│  - Generates user-facing response                                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
               { frame, reasoning, plan, response }
```

**Pipeline wiring**: `run_pipeline(utterance)` in `pipeline.py` chains all four stages.

---

## Files

| File | Description |
|---|---|
| `__init__.py` | Package marker |
| `intent_model.py` | ML model for intent classification (TF-IDF + LogisticRegression) |
| `entity_matcher.py` | Symbolic alias substring matching + TF-IDF fallback |
| `frame.py` | `IntentFrame` dataclass (intent, entity, symptom, time, confidence) |
| `ontology.py` | Canonical vocabularies, aliases, and mappings |
| `semantic_parser.py` | Orchestrates all extraction logic, including domain gating |
| `reasoner.py` | Symbolic rules for mode selection and context enrichment |
| `planner.py` | Converts reasoning into stepwise plans |
| `responder.py` | Generates user-facing responses |
| `pipeline.py` | Chains all above modules for a single utterance |
| `level4_ns_intent.ipynb` | End-to-end notebook: demo, diagnostics, and evaluation |

---

## Dataset

- **Source**: `data/intents_base.csv` (repo root)
- **Records**: 1,661 utterances
- **Intent Classes**: `summarization`, `investigate`, `execution`, `out_of_scope`
- **Entities, Symptoms, Time Contexts**: Canonicalized and aliased in `ontology.py`

---

## Model Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `TFIDF max_features` | 2000 | Vocabulary size for intent model |
| `TFIDF ngram_range` | (1, 2) | Unigrams and bigrams |
| `LogisticRegression max_iter` | 400 | Training epochs |
| `random_state` | 42 | Reproducibility |
| `EntityMatcher threshold` | 0.28 | TF-IDF similarity fallback |

---

## Module Details

- **IntentClassifier** (`intent_model.py`):
  - Trained on utterance → intent pairs.
  - `predict(utterance)` returns `(intent, confidence)`.
- **EntityMatcher** (`entity_matcher.py`):
  - 1st: Exact alias substring match (symbolic, from ontology).
  - 2nd: TF-IDF similarity fallback (if no alias match).
- **Frame** (`frame.py`):
  - `IntentFrame(intent, entity, symptom, time_context, confidence)`
  - `.to_dict()` for serialization.
- **Ontology** (`ontology.py`):
  - Canonical lists for intents, entities, symptoms, time contexts, and all aliases.
- **Semantic Parser** (`semantic_parser.py`):
  - Predicts intent, extracts entity/symptom/time, applies domain guard.
- **Reasoner** (`reasoner.py`):
  - Symbolic rules for mode selection: `diagnostic`, `action`, `reporting`, `reject`.
  - Context enrichment: target, entity type, symptom focus, time context.
- **Planner** (`planner.py`):
  - Maps mode to stepwise plan (e.g., `diagnostic` → collect_metrics, analyze_correlations).
- **Responder** (`responder.py`):
  - Generates user-facing response string from all context.
- **Pipeline** (`pipeline.py`):
  - `run_pipeline(utterance)` returns `{frame, reasoning, plan, response}`.

---

## System Flow Diagram

```mermaid
flowchart TD
    A([User Utterance]) --> B[Semantic Parser\nIntentClassifier · EntityMatcher · Symptom/Time Extractors]
    B --> C[IntentFrame\n(intent, entity, symptom, time, confidence)]
    C --> D[Reasoner\nSymbolic rules for mode/context]
    D --> E[Planner\nStepwise plan]
    E --> F[Responder\nUser-facing response]
    F --> G([Structured Output\nframe · reasoning · plan · response])
```

---

## Worked Inference Example

Utterance:
> "Why are API response times increasing today?"

| Stage | Output |
|---|---|
| IntentClassifier | `investigate` |
| EntityMatcher | `api` |
| Symptom extractor | `latency_high` |
| Time extractor | `today` |
| Reasoner | `mode: diagnostic`, `target: api`, `symptom_focus: latency_high`, `time_context: today` |
| Planner | `collect_metrics`, `collect_logs`, `analyze_correlations`, ... |
| Responder | Structured response string |

Sample output:
```json
{
  "frame": {
    "intent": "investigate",
    "entity": "api",
    "symptom": "latency_high",
    "time_context": "today",
    "confidence": {"investigate": 0.92}
  },
  "reasoning": {
    "mode": "diagnostic",
    "target": "api",
    "entity_type": "application_interface",
    "symptom_focus": "latency_high",
    "time_context": "today"
  },
  "plan": {
    "steps": ["collect_metrics", "collect_logs", "analyze_correlations", "summarize_findings"]
  },
  "response": "Intent: investigate\nEntity: api\nSymptom: latency_high\nTime Context: today\nPlanned Steps: ['collect_metrics', 'collect_logs', 'analyze_correlations', 'summarize_findings']"
}
```

---

## Key Design Decisions

1. **Ontology-driven**: All vocabularies, aliases, and mappings are centralized in `ontology.py`.
2. **Symbolic fallback**: If ML intent is in-scope but no entity/symptom is found, domain guard forces `out_of_scope`.
3. **Separation of concerns**: Each module has a single responsibility; pipeline is fully modular.
4. **Auditable**: Every stage produces structured, inspectable output.
5. **Extensible**: New entities, symptoms, or rules can be added in the ontology or reasoner without retraining the model.
6. **Notebook-orchestrated**: The notebook is the main entrypoint for experimentation and diagnostics.

---

## Comparison with Other Levels

| Level | Classifier | Symbolic Layer | Decision Output |
|-------|-----------|----------------|-----------------|
| 0 | TF-IDF + LogReg | None | Raw label only |
| 1 | TF-IDF binary detectors | Symbolization predicates + JSON rule engine | `predicted_intent` + `decision_state` + `triggered_rules` |
| 2 | Deterministic heuristics + neural adapter stub | Clause extractor + normalizer + policy validator | `decision_state` + `ambiguity_report` + `feedback` |
| 3 | PyTorch Embedding + mean-pool + Linear | Symbol emitter + rule engine (no numeric values) | `symbol` + `action` + `requires_approval` |
| **4** | **TF-IDF + LogReg + Ontology** | **Semantic parser + reasoner + planner + responder** | **frame + reasoning + plan + response** |

---

**Level 4 Status**: ✅ Complete
**Architecture**: Ontology-Grounded Neuro-Symbolic Pipeline
**Key Innovations**: Ontology-Driven Extraction · Modular Reasoning · Auditable Stepwise Output
