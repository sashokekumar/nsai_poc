# Level-2 Clause Pipeline

This folder contains a deterministic clause extraction and gating pipeline intended for use as the L2 layer.

- `llm_adapter/adapter.py`: optional adapter stub (LLM use is opt-in and outside the critical path).
- `clause_extractor.py`: deterministic candidate extraction from utterances.
- `normalizer.py`: deterministic normalization utilities (alias collapsing, time canonicalization).
- `validator.py`: deterministic validators and policy gates.
- `notebooks/l2_clause_pipeline.ipynb`: design + executable reference that wires the pieces together.

Design constraints:
- Critical path must be deterministic; adapter usage is flagged and re-validated.
- No free-text policy decisions in the canonical output; `decision_trace` is an audit artifact.
