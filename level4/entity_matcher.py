from typing import Optional, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from level4.ontology import ENTITIES, ENTITY_ALIASES


class EntityMatcher:
    """
    Matches utterances to ontology entities.
    Priority:
    1. Exact alias substring match (symbolic)
    2. TF-IDF similarity fallback
    """

    def __init__(self, threshold: float = 0.28):
        self.threshold = threshold

        # Build alias list
        alias_pairs: List[Tuple[str, str]] = []
        for canonical in ENTITIES.keys():
            for a in ENTITY_ALIASES.get(canonical, []):
                alias_pairs.append((a.lower(), canonical))

        self.alias_texts = [p[0] for p in alias_pairs]
        self.alias_to_canonical = [p[1] for p in alias_pairs]

        # TF-IDF over alias phrases
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3)).fit(self.alias_texts)
        self.alias_vectors = self.vectorizer.transform(self.alias_texts)

    def match(self, utterance: str) -> Optional[str]:
        u = str(utterance).lower().strip()
        if not u:
            return None

        # ----------------------------------
        # 1️⃣ Exact substring match (symbolic priority)
        # ----------------------------------
        for alias, canonical in zip(self.alias_texts, self.alias_to_canonical):
            if alias in u:
                return canonical

        # ----------------------------------
        # 2️⃣ Similarity fallback
        # ----------------------------------
        u_vec = self.vectorizer.transform([u])
        sims = cosine_similarity(u_vec, self.alias_vectors)[0]

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return None

        return self.alias_to_canonical[best_idx]