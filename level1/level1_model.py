"""
Level 1 Neuro-Symbolic Model Components

This module contains the shared components for Level 1 intent classification:
- Configuration constants
- Signal extraction
- Decision rules
- Prediction pipeline

Used by both level1_tfidf_classification.ipynb and validation.ipynb
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# Configuration Constants
CONFIG = {
    'BASE_MIN_CONF': 0.60,           # Minimum confidence for accepting prediction
    'MIN_MARGIN': 0.10,              # Minimum margin between top-2 predictions
    'EXECUTION_MIN_CONF': 0.85,      # Higher bar for execution intent
    'MIN_TOKENS_OUT_OF_SCOPE': 3,    # Token count threshold for out_of_scope
    'RANDOM_STATE': 42,              # For reproducibility
    'TEST_SIZE': 0.2                 # Train/test split ratio
}

# Explicit Rule Priority (highest to lowest)
RULE_PRIORITY = ["R1", "R4", "R2", "R3"]


def create_level1_pipeline():
    """Create and return the Level 1 pipeline (TF-IDF + LogisticRegression)"""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            solver='lbfgs',
            random_state=CONFIG['RANDOM_STATE'],
            max_iter=1000
        ))
    ])


def extract_signals(text, pipeline, top_k=5):
    """Extract all signals from the model for a given utterance
    
    Args:
        text: Input utterance
        pipeline: Trained sklearn pipeline (TF-IDF + classifier)
        top_k: Number of top tokens to extract per intent
        
    Returns:
        Dictionary with probabilities, confidence, margin, tokens, etc.
    """
    # Get probabilities
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps['classifier'].classes_
    
    # Sort by probability
    sorted_indices = np.argsort(proba)[::-1]
    max_confidence = float(proba[sorted_indices[0]])
    second_best_confidence = float(proba[sorted_indices[1]])
    margin = max_confidence - second_best_confidence
    
    # Get TF-IDF representation
    tfidf_vec = pipeline.named_steps['tfidf'].transform([text])
    active_features = tfidf_vec.toarray()[0]
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    
    # Count meaningful tokens (non-zero TF-IDF)
    meaningful_tokens = np.sum(active_features > 0)
    
    # Extract top contributing tokens per intent
    top_tokens_per_intent = {}
    for intent_idx, intent in enumerate(classes):
        coef = pipeline.named_steps['classifier'].coef_[intent_idx]
        contributions = active_features * coef
        top_indices = np.argsort(contributions)[-top_k:][::-1]
        
        tokens = []
        for idx in top_indices:
            if active_features[idx] > 0:
                tokens.append({
                    'token': feature_names[idx],
                    'weight': float(contributions[idx])
                })
        top_tokens_per_intent[intent] = tokens
    
    # Build signals dictionary
    signals = {
        'probabilities': {intent: float(proba[i]) for i, intent in enumerate(classes)},
        'max_confidence': max_confidence,
        'second_best_confidence': second_best_confidence,
        'margin': margin,
        'predicted_intent': classes[sorted_indices[0]],
        'meaningful_tokens': int(meaningful_tokens),
        'top_tokens_per_intent': top_tokens_per_intent
    }
    
    return signals


def apply_decision_rules(signals):
    """Apply deterministic rules with strict precedence hierarchy
    
    Intent = what user wants (always from model)
    Decision State = what system decides to do
    
    Rule Categories:
    - quality: input sufficiency (R1)
    - safety: execution controls (R4)
    - ambiguity: confidence/margin (R2, R3)
    
    Precedence follows RULE_PRIORITY: R1 > R4 > R2 > R3
    
    Args:
        signals: Dictionary from extract_signals()
        
    Returns:
        Dictionary with triggered_rules, predicted_intent, decision_state, decision_reason
    """
    triggered_rules = []
    predicted_intent = signals['predicted_intent']  # Always keep model prediction
    decision_state = 'accepted'
    decision_reason = 'model_prediction'
    
    # R1: QUALITY GATE (highest priority)
    if signals['meaningful_tokens'] < CONFIG['MIN_TOKENS_OUT_OF_SCOPE']:
        triggered_rules.append({
            'rule_id': 'R1',
            'category': 'quality',
            'priority': 100,
            'condition': f"meaningful_tokens < {CONFIG['MIN_TOKENS_OUT_OF_SCOPE']}",
            'signal_value': signals['meaningful_tokens']
        })
        decision_state = 'blocked'
        decision_reason = 'insufficient_tokens'
        # R1 overrides everything - return immediately
        return {
            'triggered_rules': triggered_rules,
            'predicted_intent': predicted_intent,
            'decision_state': decision_state,
            'decision_reason': decision_reason
        }
    
    # R4: SAFETY GATE
    if signals['predicted_intent'] == 'execution' and signals['max_confidence'] < CONFIG['EXECUTION_MIN_CONF']:
        triggered_rules.append({
            'rule_id': 'R4',
            'category': 'safety',
            'priority': 90,
            'condition': f"predicted_intent==execution AND max_confidence < {CONFIG['EXECUTION_MIN_CONF']}",
            'signal_value': signals['max_confidence']
        })
        # Block execution if confidence insufficient
        if signals['max_confidence'] >= CONFIG['BASE_MIN_CONF']:
            decision_state = 'blocked'
            decision_reason = 'execution_safety_block'
        else:
            decision_state = 'needs_clarification'
            decision_reason = 'execution_low_confidence'
        # R4 overrides R2 and R3 - return immediately
        return {
            'triggered_rules': triggered_rules,
            'predicted_intent': predicted_intent,
            'decision_state': decision_state,
            'decision_reason': decision_reason
        }
    
    # R2 & R3: AMBIGUITY GATES
    ambiguity_detected = False
    
    # R2: Low confidence
    if signals['max_confidence'] < CONFIG['BASE_MIN_CONF']:
        triggered_rules.append({
            'rule_id': 'R2',
            'category': 'ambiguity',
            'priority': 50,
            'condition': f"max_confidence < {CONFIG['BASE_MIN_CONF']}",
            'signal_value': signals['max_confidence']
        })
        ambiguity_detected = True
    
    # R3: Low margin
    if signals['margin'] < CONFIG['MIN_MARGIN']:
        triggered_rules.append({
            'rule_id': 'R3',
            'category': 'ambiguity',
            'priority': 40,
            'condition': f"margin < {CONFIG['MIN_MARGIN']}",
            'signal_value': signals['margin']
        })
        ambiguity_detected = True
    
    if ambiguity_detected:
        decision_state = 'needs_clarification'
        decision_reason = 'ambiguous_prediction'
    
    return {
        'triggered_rules': triggered_rules,
        'predicted_intent': predicted_intent,
        'decision_state': decision_state,
        'decision_reason': decision_reason
    }


def weighted_voting(signals):
    """Compute vote scores for transparency
    
    NOTE: Votes are for explainability only.
    Rules have final authority and override votes.
    
    Args:
        signals: Dictionary from extract_signals()
        
    Returns:
        Dictionary with score_per_intent, vote_log, vote_winner
    """
    scores = {intent: 0.0 for intent in signals['probabilities'].keys()}
    vote_log = []
    
    # Vote 1: Base probability scores
    for intent, prob in signals['probabilities'].items():
        scores[intent] += prob * 1.0
        vote_log.append({
            'vote_id': 'V1',
            'source': 'model_probability',
            'intent': intent,
            'contribution': prob * 1.0
        })
    
    # Vote 2: Confidence bonus for high-confidence predictions
    if signals['max_confidence'] > CONFIG['EXECUTION_MIN_CONF']:
        bonus = 0.3
        scores[signals['predicted_intent']] += bonus
        vote_log.append({
            'vote_id': 'V2',
            'source': 'high_confidence_bonus',
            'intent': signals['predicted_intent'],
            'contribution': bonus
        })
    
    # Vote 3: Margin bonus
    if signals['margin'] > 0.20:
        bonus = 0.2
        scores[signals['predicted_intent']] += bonus
        vote_log.append({
            'vote_id': 'V3',
            'source': 'high_margin_bonus',
            'intent': signals['predicted_intent'],
            'contribution': bonus
        })
    
    return {
        'score_per_intent': scores,
        'vote_log': vote_log,
        'vote_winner': max(scores, key=scores.get)
    }
