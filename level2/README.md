# Level 2: Clause Extraction & Symbolic Validation Pipeline

## Overview
Level 2 takes the **intent** produced by Level 1 and decomposes the utterance into structured **clause candidates** — the parsed arguments needed to act on that intent. It then validates those candidates against intent-specific rules and policy gates, generating actionable feedback for iterative refinement.

**Key Separation**:
- **What the user wants** (intent) comes from Level 1
- **How to act on it** (structured clauses + decision) is Level 2's responsibility

## Architecture

```
Utterance + Intent (L1)
        │
        ▼
┌─────────────────────┐
│   Neural Adapter    │  ← structured clause candidates + per-clause confidence
│  (llm_adapter/)     │    accepts feedback signal for iterative refinement
└─────────┬───────────┘
          │ adapter_result (clauses, confidence, notes)
          ▼
┌─────────────────────┐
│  Clause Extractor   │  ← merges adapter output with deterministic detectors
│ (clause_extractor)  │    regex/keyword extraction (entity, operation, metric…)
│                     │    feedback override: collapses ambiguous candidates
└─────────┬───────────┘
          │ clauses: Dict[str, List[str]]
          ▼
┌─────────────────────┐
│    Normalizer       │  ← alias collapsing, ISO-8601 time canonicalisation
│   (normalizer)      │
└─────────┬───────────┘
          │ normalised clauses
          ▼
┌─────────────────────┐
│    Validator        │  ← intent-specific hard rules, policy gates, soft rules
│   (validator)       │    produces decision_state + feedback for next iteration
└─────────────────────┘
          │
          ▼
  (decision_state, details, ambiguity, feedback)
```

**Iteration loop**: when `decision_state == "needs_clarification"`, the feedback signal is passed back to the adapter for a refined extraction pass (up to `MAX_ITERATIONS`).

## Clause Schema

Eight clause fields are tracked throughout the pipeline:

| Field | Description | Example values |
|---|---|---|
| `entity` | Target resource | `nginx-1`, `db-prod-42` |
| `operation` | Action to perform | `restart`, `deploy`, `delete` |
| `metric` | Observable signal | `cpu`, `memory`, `latency` |
| `time_window` | ISO-8601 duration | `PT30M`, `PT1H`, `P1D` |
| `environment` | Deployment tier | `prod`, `staging`, `dev` |
| `condition` | Trigger condition | `when cpu > 90%` |
| `constraint` | Policy constraint | `requires_approval` |
| `output_format` | Desired response format | `json`, `table` |

## Modules

### `clause_extractor.py` — Candidate Extraction
Merges two parallel signal sources into a unified `Dict[str, List[str]]` of candidates per clause.

**Step 1 — Adapter merge**: adapter-provided candidates are inserted first; adapter ordering is trusted as the primary signal.

**Step 2 — Deterministic merge** (deduplication-safe `_merge_unique`):

| Detector | Method | Clause |
|---|---|---|
| `regex_entity` | Pattern `[a-zA-Z0-9_-]+-\d+` | `entity` |
| `keyword_operation` | Keyword list + synonym map | `operation` |
| `keyword_environment` | Keyword list + alias collapsing | `environment` |
| `regex_time_window` | `last N unit` → ISO-8601 | `time_window` |
| `keyword_metric` | Keyword list | `metric` |
| `regex_condition` | `if/when …` patterns | `condition` |
| `keyword_constraint` | `requires approval` patterns | `constraint` |

**Step 3 — Feedback override**: when a `feedback_used:<clause>` note is present in adapter metadata, indicates the adapter acted on symbolic feedback. The extractor collapses the named clause to its single top candidate, guaranteeing the validator sees no `conflicting_candidates` on the next pass. This is what makes the iteration loop converge.

### `normalizer.py` — Canonical Form
Applied after extraction, before validation:

- **`normalize_aliases`**: collapses operation synonyms (`reboot` → `restart`, `svc` → `service`)
- **`normalize_time_windows`**: ensures ISO-8601 durations are uppercase (`pt30m` → `PT30M`)
- **`normalize_clauses`**: dispatches per-field normalization + strips whitespace from all other fields

### `validator.py` — Intent-Specific Rules & Policy Gates
Returns `(decision_state, details, ambiguity, feedback)`.

#### Decision States

| State | Meaning |
|---|---|
| `accepted` | All hard rules passed, no conflicting candidates |
| `needs_clarification` | Missing or ambiguous clauses — iteration is possible |
| `blocked` | Policy violation — cannot proceed without constraint resolution |

#### Acceptance Logic

```python
if not hard_failed and not ambiguity["conflicting_candidates"]:
    state = "accepted"
elif ambiguity["conflicting_candidates"]:
    state = "needs_clarification"   # adapter can resolve via feedback
    ambiguity["needs_iteration"] = True
else:
    # missing clauses → needs_clarification
    # policy conflicts → blocked
```

#### Hard Rules (by intent)

**`execute`**:

| Rule | Condition | Failure key |
|---|---|---|
| Entity required | `clauses["entity"]` is empty | `missing_entity` |
| Operation required | `clauses["operation"]` is empty | `missing_operation` |
| Operation unambiguous | `len(clauses["operation"]) > 1` | → `conflicting_candidates` |

**`investigate`**:

| Rule | Condition | Failure key |
|---|---|---|
| Metric or condition required | both `metric` and `condition` empty | `missing_metric_or_condition` |

#### Policy Gates

| Gate | Condition | Outcome |
|---|---|---|
| Delete in prod | `delete` in operations AND `prod` in environment AND `requires_approval` absent | `blocked` — `forbidden_delete_in_prod_without_approval` |

#### Soft Rules (non-blocking)

| Rule | Condition | Signal |
|---|---|---|
| Time window present | `clauses["time_window"]` non-empty | `has_time_window` added to `soft_passed` |

#### Feedback Signal

When validation cannot accept, structured feedback is returned to guide the next adapter pass:

```python
{
  "focus_clause": "operation",          # which clause to re-examine
  "reason": "conflicting_candidates",   # why it failed
  "hints": ["restart"]                  # heuristic preferred candidates
}
```

### `llm_adapter/adapter.py` — Neural Adapter (Structured Stub)
Deterministic by default (fully testable), designed to be drop-in replaced with a real LLM or NN.

- Returns `AdapterResult(clauses, confidence, notes)` — one ordered candidate list per clause
- Accepts an `AdapterFeedback` object to refine output on subsequent iterations
- When feedback is used, appends a `feedback_used:<clause>` note to `notes` — this is the hook the extractor's feedback override reads
- Per-clause `confidence` dict allows the validator to weight signals if needed

## Output Structure

```json
{
  "utterance": "restart nginx-1 in prod",
  "intent": "execute",
  "clauses": {
    "entity": ["nginx-1"],
    "operation": ["restart"],
    "environment": ["prod"],
    "metric": [],
    "time_window": [],
    "condition": [],
    "constraint": [],
    "output_format": []
  },
  "decision_state": "accepted",
  "decision_reason": "all_hard_rules_passed",
  "ambiguity": {
    "missing_clauses": [],
    "conflicting_candidates": {},
    "policy_conflicts": [],
    "needs_iteration": false
  },
  "feedback": {
    "focus_clause": null,
    "reason": null,
    "hints": []
  },
  "detectors_fired": ["adapter", "regex_entity", "keyword_operation", "keyword_environment"],
  "adapter_meta": {
    "confidence": {"entity": 0.95, "operation": 0.90},
    "notes": []
  }
}
```

## Example Scenarios

### Accepted — Clean Execution Intent
```
Input: "restart nginx-1 in prod"
Intent: execute
clauses: entity=["nginx-1"], operation=["restart"], environment=["prod"]
decision_state: accepted
```

### Needs Clarification — Conflicting Operation Candidates
```
Input: "start or restart nginx-1"
Intent: execute
clauses: operation=["start", "restart"]
conflicting_candidates: {"operation": ["start", "restart"]}
decision_state: needs_clarification
feedback: {focus_clause: "operation", reason: "conflicting_candidates", hints: ["restart"]}
→ adapter re-runs with feedback → extractor collapses to ["restart"] → accepted
```

### Needs Clarification — Missing Metric
```
Input: "check nginx-1"
Intent: investigate
clauses: metric=[], condition=[]
decision_state: needs_clarification
feedback: {focus_clause: "metric", reason: "missing_clause", hints: ["cpu","memory","latency","errors","throughput"]}
```

### Blocked — Policy Violation
```
Input: "delete nginx-1 in prod"
Intent: execute
clauses: operation=["delete"], environment=["prod"], constraint=[]
decision_state: blocked
feedback: {focus_clause: "constraint", reason: "policy_requires_constraint", hints: ["requires_approval"]}
```

## Design Constraints
1. **Critical path is fully deterministic** — adapter usage is opt-in and flagged in `detectors_fired`; all adapter outputs are re-validated by the symbolic layer
2. **No free-text policy decisions** in canonical output — `decision_trace` / `notes` are audit artifacts only
3. **Iteration converges** — feedback override in the extractor guarantees conflicting candidates are collapsed on the feedback pass, preventing infinite loops
4. **Adapter is swappable** — `AdapterResult` / `AdapterFeedback` interfaces are stable; replace the stub with a real LLM without touching the rest of the pipeline

## Files
```
level2/
├── README.md
├── clause_extractor.py          # adapter merge + deterministic extraction + feedback override
├── normalizer.py                # alias collapsing, ISO-8601 canonicalisation
├── validator.py                 # intent rules, policy gates, decision state, feedback
└── llm_adapter/
    └── adapter.py               # structured neural stub (AdapterResult, AdapterFeedback)
notebooks/
└── l2_clause_pipeline.ipynb     # end-to-end wiring + iteration loop demo
```

---

**Level 2 Status**: ✅ Complete  
**Architecture**: Adapter → Extraction → Normalization → Symbolic Validation (iterative)  
**Key Innovation**: Structured clause decomposition with convergent feedback loop  
**Critical Path**: Fully deterministic — adapter is required but symbolic layer always re-validates
