"""Simple LLM adapter stub for L2.

This adapter is intentionally minimal and returns a deterministic empty result
unless replaced by a real adapter implementation. The function signature is
kept stable so it can be swapped without changing the pipeline.
"""
from typing import Dict, List


def extract_clauses(utterance: str, timeout_s: float = 2.0) -> Dict[str, List[str]]:
    """Stub extractor.

    Parameters
    - utterance: raw input text
    - timeout_s: adapter timeout in seconds (unused in stub)

    Returns a mapping clause -> list of candidate strings.
    """
    # No LLM used in critical path; adapter returns nothing by default.
    return {}
