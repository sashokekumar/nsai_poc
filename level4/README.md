> **Note — Work in progress**: This README is being rewritten.  
> The earlier runtime pipeline (EntityMatcher, Domain Guard, Reasoner, Planner, Responder) that
> was originally named Level 4 has been **reclassified as Level 3.5** and moved to `level3_5/`.  
> It was reclassified because symbolic components remain active at inference time rather than
> being encoded into model weights — which places it below the neuro-symbolic integration
> threshold for a true Type-4 system (Kautz typology).
>
> The **current Level 4 implementation** is being rebuilt from scratch as a
> **symbolically supervised neural model**: ontology-derived constraint rules are compiled into
> an auxiliary loss term at training time, and inference is performed by the neural network
> alone with zero symbolic post-processing.

---

# Level 4 — Symbolically Supervised Neural Intent Classifier

*(Full README to follow once experiments are complete.)*
