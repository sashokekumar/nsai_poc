# Level 4 — Symbolically Supervised Neural Model
## Implementation TODO

| # | Area | Task | Notes | Status |
|---|------|------|-------|--------|
| 1 | Environment | Confirm `sentence-transformers` (`all-MiniLM-L6-v2`) is available in venv | Installed v5.4.1; model downloaded, embedding dim=384 | Done |
| 2 | Data | Audit `intents_base.csv` — confirm utterance + intent coverage | 1,661 rows, 2 cols (utterance, intent), no nulls, balanced across 4 intents; needs entity_type + domain_valid columns | Done |
| 3 | Data | Programmatically generate `entity_type` and `domain_valid` labels using existing `level3_5` EntityMatcher and ontology | Done — 1,661 rows labeled; entity_type mapped to 7 categories (infrastructure/service/metric/incident/job/pipeline/unknown); 110 SRE-intent rows with domain_valid=False (ontology coverage gaps, kept as honest signal); saved to `level4/data/labeled_raw.csv` | Done |
| 4 | Data | Manually review and correct generated labels | 224 out_of_scope rows had spurious TF-IDF entity matches corrected to `unknown`; 110 SRE-intent rows with domain_valid=False retained as honest ontology-gap signal; saved to `level4/data/labeled_clean.csv` | Done |
| 5 | Data | Split into `train.csv` and `test.csv` | Stratified 80/20 split by intent; train=1328 rows, test=333 rows; intent proportions preserved across both splits | Done |
| 6 | Ontology | Create `ontology/constraint_rules.json` — define allowed and disallowed intent/entity_type pairs | 9 disallowed rules: 6 TYPE_A false-rejection (out_of_scope + SRE entity, weight=1.0), 1 TYPE_B false-execution (execution + incident, weight=0.75), 2 TYPE_C ungrounded (unknown entity for SRE intent, weight=0.5); includes violation taxonomy and λ ablation values | Done |
| 7 | Model | Implement `model/dataset.py` — PyTorch Dataset class over `train.csv` / `test.csv` | Label vocab matches constraint_rules.json; validates unknown intents/entities at load time; returns utterance text + 3 label tensors | Done |
| 8 | Model | Implement `model/neural_intent_model.py` — all-MiniLM encoder (frozen) + shared dense + intent/entity/domain heads | 384→256 shared trunk, 3 heads (4/7/1 outputs); `.clone()` on embeddings to exit inference mode for autograd; clean `predict()` with no symbolic post-processing | Done |
| 9 | Model | Implement `model/losses.py` — constraint violation loss | Differentiable soft penalty: sum over disallowed pairs of w_ij * mean(p_intent[:,i] * p_entity[:,j]); Level4Loss = intent_loss + α*entity_loss + β*domain_loss + λ*constraint_loss; lam=0 gives clean baseline | Done |
| 10 | Training | Implement `train.py` — training loop with `total_loss = intent_loss + α*entity_loss + β*domain_loss + λ*constraint_loss` | λ = 0 run first as baseline | Open |
| 11 | Inference | Implement `infer.py` — clean inference path: `utterance → model → prediction` with zero symbolic post-processing | No guard, no reasoner, no planner | Open |
| 12 | Evaluation | Implement `evaluation/violation_metrics.py` — compute constraint violation rate, false execution rate, false rejection rate | Offline only, does not correct predictions | Open |
| 13 | Experiment A | Train baseline neural model (λ = 0, no symbolic loss, no runtime guard) | Establish base violation rate | Open |
| 14 | Experiment B | Run Level 3.5 runtime pipeline on same test set and record metrics | Comparison reference | Open |
| 15 | Experiment C | Train Level 4 model with symbolic constraint loss (λ > 0, no runtime guard) | Core Level 4 claim | Open |
| 16 | Experiment D | λ ablation sweep: 0.0, 0.1, 0.25, 0.5, 1.0, 2.0 | Find best accuracy/violation-rate tradeoff; identify breakpoint where constraint loss dominates | Open |
| 17 | Evaluation | Produce comparison table: Baseline / Level 3.5 / Level 4 across all metrics | Intent accuracy, entity accuracy, domain accuracy, constraint violation rate, false execution rate, false rejection rate | Open |
| 18 | Documentation | Update `level3_5/README.md` — add explicit reclassification note | State it was originally Level 4, reclassified to 3.5 per Kautz typology review | Open |
| 19 | Documentation | Write `level4/README.md` — define the level, architecture, experiment design, and what success looks like | Include architecture diagram and paper framing | Open |
| 20 | Notebook | Create `level4/level4_symbolic_loss.ipynb` — end-to-end walkthrough of train → infer → evaluate | For presentation and reproducibility | Open |
