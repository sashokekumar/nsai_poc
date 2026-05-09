# Level 4: Symbolically Supervised Neural Model

> **Note — Work in progress (reclassification in effect):**  
> This folder is being rebuilt as a strict **Kautz Type 4** implementation.  
> The earlier runtime neuro-symbolic pipeline (TF-IDF + EntityMatcher + Domain Guard + Reasoner + Planner + Responder) has been **reclassified as Level 3.5** and moved to `level3_5/` — it was reclassified because symbolic components remain active at inference time, which is below the Type-4 threshold.  
>  
> **In this corrected Level 4:**  
> - Symbolic ontology constraints are compiled into a differentiable loss term **at training time only**  
> - **At inference, the neural model runs alone** — no symbolic guards, reasoners, planners, or post-processing  
> - Success is measured by whether training-time symbolic supervision reduces constraint violation rates **without any runtime correction**

---

> **`domain_valid` semantics note:**  
> In this dataset, `domain_valid=True` means the utterance is SRE-relevant and routable.  
> `domain_valid=False` maps exactly to `out_of_scope` intent — it does **not** mean "ontology grounding failed."  
> SRE-intent utterances with `entity_type=unknown` are `domain_valid=True` but weakly grounded.  
> The constraint loss (TYPE_C penalty) handles grounding quality separately from domain validity.

---

*(Full architecture description, experiment design, and results to be written after experiments complete.)*

