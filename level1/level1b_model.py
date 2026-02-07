"""
Level 1B: Multi-Detector Neuro-Symbolic Intent Classification

Architecture:
- One binary detector per intent (not multi-class)
- Detectors score independently (scores don't sum to 1)
- Symbolic rules govern final decisions
- Explicit rule types for L2/L3 escalation
"""

import numpy as np
import pickle
import json
from pathlib import Path

# Rule Configuration
BASE_MIN_SCORE = 0.50
AMBIGUITY_MARGIN = 0.10
EXECUTION_MIN_SCORE = 0.85
MIN_TOKENS_OUT_OF_SCOPE = 3
CONCURRENCE_THRESHOLD = 0.70
MIN_PREDICATE_COUNT = 2
MIN_PREDICATE_DIVERSITY = 2

# Rule priority (highest first)
RULE_PRIORITY = [
    "R_LOW_TOKEN_COUNT",
    "R_NO_CONFIDENT_DETECTOR",
    "R_EXEC_SAFETY",
    "R_MULTI_DETECTOR_CONCURRENCE",
    "R_AMBIGUOUS",
    "R_DEFAULT"
]

# Rule types for explicit escalation
RULE_TYPES = {
    "R_LOW_TOKEN_COUNT": "quality",
    "R_NO_CONFIDENT_DETECTOR": "quality",
    "R_EXEC_SAFETY": "safety",
    "R_MULTI_DETECTOR_CONCURRENCE": "compound_intent",
    "R_AMBIGUOUS": "ambiguity",
    "R_DEFAULT": "default",
    "R_INSUFFICIENT_PREDICATES": "quality"
}

# Rule class severity (hard / soft / contextual)
RULE_CLASS = {
    'R_LOW_TOKEN_COUNT': 'hard',
    'R_NO_CONFIDENT_DETECTOR': 'hard',
    'R_EXEC_SAFETY': 'hard',
    'R_MULTI_DETECTOR_CONCURRENCE': 'contextual',
    'R_AMBIGUOUS': 'soft',
    'R_DEFAULT': 'soft',
    'R_INSUFFICIENT_PREDICATES': 'hard'
}


class Level1BClassifier:
    """Multi-Detector Neuro-Symbolic Intent Classifier"""
    
    def __init__(self, detectors=None, intents=None):
        """
        Initialize Level 1B classifier
        
        Args:
            detectors: Dict of {intent: trained_detector_pipeline}
            intents: List of intent names
        """
        self.detectors = detectors or {}
        self.intents = intents or ['investigate', 'execution', 'summarization', 'out_of_scope']
        
    def extract_detector_signals(self, text):
        """Extract independent scores from all intent detectors
        
        Returns:
            Dictionary with detector_scores, top_detector, margin, tokens
        """
        detector_scores = {}
        
        # Get score from each detector independently
        for intent, detector in self.detectors.items():
            # Get probability of positive class (utterance matches this intent)
            proba = detector.predict_proba([text])[0]
            # Score is probability of class=1 (match)
            detector_scores[intent] = float(proba[1])
        
        # Sort by score
        sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
        top_detector = sorted_detectors[0][0]
        top_score = sorted_detectors[0][1]
        second_detector = sorted_detectors[1][0]
        second_score = sorted_detectors[1][1]
        score_margin = top_score - second_score
        
        # Get token count from any detector's TF-IDF (all use same settings)
        first_detector = self.detectors[self.intents[0]]
        tfidf_vec = first_detector.named_steps['tfidf'].transform([text])
        active_features = tfidf_vec.toarray()[0]
        meaningful_tokens = int(np.sum(active_features > 0))

        # Build predicate keyword matches per intent based on detector feature weights
        intent_keyword_matches = {}
        for intent, detector in self.detectors.items():
            try:
                tfidf = detector.named_steps['tfidf']
                clf = detector.named_steps['clf']
                feature_names = tfidf.get_feature_names_out()
                # coef_ shape (1, n_features) for binary classifiers
                coefs = clf.coef_[0]
                # take features with positive contribution above small threshold
                pos_idx = [i for i, w in enumerate(coefs) if w > 0.0]
                keywords = set(feature_names[i] for i in pos_idx)
            except Exception:
                keywords = set()

            # count how many keywords appear in the text (simple token match)
            text_tokens = set([t.lower() for t in text.split()])
            matched = sorted([k for k in keywords if k in text_tokens])
            intent_keyword_matches[intent] = {
                'matched_keywords': matched,
                'matched_count': len(matched)
            }
        
        return {
            'detector_scores': detector_scores,
            'top_detector': top_detector,
            'top_score': top_score,
            'second_detector': second_detector,
            'second_score': second_score,
            'score_margin': score_margin,
            'meaningful_tokens': meaningful_tokens,
            'intent_keyword_matches': intent_keyword_matches
        }
    
    def apply_rules(self, signals):
        """Apply deterministic rules to detector signals
        
        Rules operate ONLY on numeric signals.
        Rules ALWAYS override raw scores.
        Each rule includes explicit rule_type for L2/L3 escalation.
        
        Returns:
            Dictionary with predicted_intent, decision_state, decision_reason, triggered_rules
        """
        triggered_rules = []
        predicted_intent = signals['top_detector']  # Default
        decision_state = 'accepted'
        decision_reason = 'R_DEFAULT'
        
        detector_scores = signals['detector_scores']
        
        # Count how many detectors score above threshold
        above_threshold = sum(1 for score in detector_scores.values() if score >= BASE_MIN_SCORE)
        high_confidence_detectors = [intent for intent, score in detector_scores.items() if score >= CONCURRENCE_THRESHOLD]

        # Start building decision trace
        intent_candidates = [intent for intent, score in detector_scores.items() if score >= BASE_MIN_SCORE]
        eliminated = {}
        accepted = {}
        # detectors_fired: intents with score >= BASE_MIN_SCORE
        detectors_fired = intent_candidates.copy()
        
        # R_LOW_TOKEN_COUNT (Highest Priority - Quality Issue)
        if signals['meaningful_tokens'] < MIN_TOKENS_OUT_OF_SCOPE:
            triggered_rules.append({
                'rule_id': 'R_LOW_TOKEN_COUNT',
                'rule_type': RULE_TYPES.get('R_LOW_TOKEN_COUNT', 'quality'),
                'priority': 100,
                'condition': f"meaningful_tokens < {MIN_TOKENS_OUT_OF_SCOPE}",
                'value': signals['meaningful_tokens']
            })
            predicted_intent = 'out_of_scope'
            decision_state = 'blocked'
            decision_reason = 'R_LOW_TOKEN_COUNT'
            eliminated = {i: ['low_token_count'] for i in self.intents if i != 'out_of_scope'}
            return {
                'predicted_intent': predicted_intent,
                'decision_state': decision_state,
                'decision_reason': decision_reason,
                'triggered_rules': triggered_rules,
                'decision_trace': {
                    'detectors_fired': detectors_fired,
                    'hard_rules_failed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'hard'],
                    'soft_rules_passed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'soft'],
                    'alternatives_eliminated': eliminated,
                    'intent_candidates': intent_candidates,
                    'eliminated': eliminated,
                    'accepted': accepted
                }
            }
        
        # R_NO_CONFIDENT_DETECTOR (Quality Issue - Semantic Gap 1)
        if all(score < BASE_MIN_SCORE for score in detector_scores.values()):
            triggered_rules.append({
                'rule_id': 'R_NO_CONFIDENT_DETECTOR',
                'rule_type': RULE_TYPES.get('R_NO_CONFIDENT_DETECTOR', 'quality'),
                'priority': 95,
                'condition': f"all detector_scores < {BASE_MIN_SCORE}",
                'value': max(detector_scores.values()),
                'explanation': 'No detector confident enough to classify'
            })
            predicted_intent = 'out_of_scope'
            decision_state = 'blocked'
            decision_reason = 'R_NO_CONFIDENT_DETECTOR'
            eliminated = {intent: ['score_below_threshold'] for intent in self.intents}
            return {
                'predicted_intent': predicted_intent,
                'decision_state': decision_state,
                'decision_reason': decision_reason,
                'triggered_rules': triggered_rules,
                'decision_trace': {
                    'detectors_fired': detectors_fired,
                    'hard_rules_failed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'hard'],
                    'soft_rules_passed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'soft'],
                    'alternatives_eliminated': eliminated,
                    'intent_candidates': intent_candidates,
                    'eliminated': eliminated,
                    'accepted': accepted
                }
            }
        
        # R_EXEC_SAFETY (Safety Rule)
        exec_score = detector_scores['execution']
        if exec_score >= BASE_MIN_SCORE and exec_score < EXECUTION_MIN_SCORE:
            triggered_rules.append({
                'rule_id': 'R_EXEC_SAFETY',
                'rule_type': RULE_TYPES.get('R_EXEC_SAFETY', 'safety'),
                'priority': 90,
                'condition': f"execution_score >= {BASE_MIN_SCORE} AND < {EXECUTION_MIN_SCORE}",
                'value': exec_score
            })
            
            # Check if investigate is viable alternative
            if detector_scores['investigate'] >= BASE_MIN_SCORE:
                predicted_intent = 'investigate'
                decision_state = 'accepted'
                decision_reason = 'R_EXEC_SAFETY'
            else:
                predicted_intent = 'execution'
                decision_state = 'needs_clarification'
                decision_reason = 'R_EXEC_SAFETY'
            
            return {
                'predicted_intent': predicted_intent,
                'decision_state': decision_state,
                'decision_reason': decision_reason,
                'triggered_rules': triggered_rules,
                'decision_trace': {
                    'detectors_fired': detectors_fired,
                    'hard_rules_failed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'hard'],
                    'soft_rules_passed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'soft'],
                    'alternatives_eliminated': eliminated,
                    'intent_candidates': intent_candidates,
                    'eliminated': eliminated,
                    'accepted': {predicted_intent: [r['rule_id'] for r in triggered_rules]}
                }
            }
        
        # R_MULTI_DETECTOR_CONCURRENCE (Reserved for L2 - Compound Intent)
        if len(high_confidence_detectors) >= 2:
            triggered_rules.append({
                'rule_id': 'R_MULTI_DETECTOR_CONCURRENCE',
                'rule_type': RULE_TYPES.get('R_MULTI_DETECTOR_CONCURRENCE', 'compound_intent'),
                'priority': 60,
                'condition': f"multiple detectors >= {CONCURRENCE_THRESHOLD}",
                'value': len(high_confidence_detectors),
                'active_detectors': high_confidence_detectors,
                'explanation': 'Reserved for L2: Compound intent detected'
            })
            # For L1B: Mark for clarification, but don't block
            # L2 will orchestrate multiple intents
            # For now, fall through to other rules
        
        # R_AMBIGUOUS (Ambiguity Rule)
        if above_threshold >= 2 and signals['score_margin'] < AMBIGUITY_MARGIN:
            triggered_rules.append({
                'rule_id': 'R_AMBIGUOUS',
                'rule_type': RULE_TYPES.get('R_AMBIGUOUS', 'ambiguity'),
                'priority': 50,
                'condition': f"multiple scores >= {BASE_MIN_SCORE} AND margin < {AMBIGUITY_MARGIN}",
                'value': signals['score_margin']
            })
            predicted_intent = signals['top_detector']
            decision_state = 'needs_clarification'
            decision_reason = 'R_AMBIGUOUS'
            eliminated = {intent: ['ambiguous_competition'] for intent in self.intents if intent != predicted_intent}
            return {
                'predicted_intent': predicted_intent,
                'decision_state': decision_state,
                'decision_reason': decision_reason,
                'triggered_rules': triggered_rules,
                'decision_trace': {
                    'intent_candidates': intent_candidates,
                    'eliminated': eliminated,
                    'accepted': {predicted_intent: [r['rule_id'] for r in triggered_rules]}
                }
            }
        
        # R_DEFAULT - Select highest scoring detector
        # Minimal sufficiency check for predicates
        top_intent = signals['top_detector']
        match_info = signals.get('intent_keyword_matches', {}).get(top_intent, {})
        matched_count = match_info.get('matched_count', 0)

        if matched_count < MIN_PREDICATE_COUNT:
            triggered_rules.append({
                'rule_id': 'R_INSUFFICIENT_PREDICATES',
                'rule_type': RULE_TYPES.get('R_INSUFFICIENT_PREDICATES', 'quality'),
                'priority': 20,
                'condition': f"matched_predicate_count < {MIN_PREDICATE_COUNT}",
                'value': matched_count,
                'matched_keywords': match_info.get('matched_keywords', [])
            })
            predicted_intent = 'out_of_scope'
            decision_state = 'blocked'
            decision_reason = 'R_INSUFFICIENT_PREDICATES'
            eliminated = {intent: ['insufficient_predicates'] for intent in self.intents if intent != 'out_of_scope'}
            return {
                'predicted_intent': predicted_intent,
                'decision_state': decision_state,
                'decision_reason': decision_reason,
                'triggered_rules': triggered_rules,
                'decision_trace': {
                    'detectors_fired': detectors_fired,
                    'hard_rules_failed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'hard'],
                    'soft_rules_passed': [r['rule_id'] for r in triggered_rules if RULE_CLASS.get(r['rule_id']) == 'soft'],
                    'alternatives_eliminated': eliminated,
                    'intent_candidates': intent_candidates,
                    'eliminated': eliminated,
                    'accepted': accepted
                }
            }

        triggered_rules.append({
            'rule_id': 'R_DEFAULT',
            'rule_type': 'default',
            'priority': 10,
            'condition': 'default to highest scoring detector',
            'value': signals['top_score']
        })
        predicted_intent = signals['top_detector']
        decision_state = 'accepted'
        decision_reason = 'R_DEFAULT'
        
        accepted = {predicted_intent: [r['rule_id'] for r in triggered_rules]}
        return {
            'predicted_intent': predicted_intent,
            'decision_state': decision_state,
            'decision_reason': decision_reason,
            'triggered_rules': triggered_rules,
            'decision_trace': {
                'intent_candidates': intent_candidates,
                'eliminated': eliminated,
                'accepted': accepted
            }
        }
    
    def predict(self, text):
        """Complete Level 1B inference pipeline
        
        Returns:
            Structured decision with detector scores, rules, and final outcome
        """
        if not self.detectors:
            raise ValueError("No detectors loaded. Train or load detectors first.")
        
        # Extract detector signals
        signals = self.extract_detector_signals(text)
        
        # Apply symbolic rules
        rules_output = self.apply_rules(signals)
        
        # Build final output
        return {
            'predicted_intent': rules_output['predicted_intent'],
            'decision_state': rules_output['decision_state'],
            'decision_reason': rules_output['decision_reason'],
            'detector_scores': signals['detector_scores'],
            'top_detector': signals['top_detector'],
            'score_margin': signals['score_margin'],
            'meaningful_tokens': signals['meaningful_tokens'],
            'triggered_rules': rules_output['triggered_rules'],
            'decision_trace': rules_output.get('decision_trace', {}),
            'intent_keyword_matches': signals.get('intent_keyword_matches', {})
        }
    
    def save(self, model_dir):
        """Save detectors and configuration"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each detector
        for intent, detector in self.detectors.items():
            detector_path = model_dir / f"detector_{intent}.pkl"
            with open(detector_path, 'wb') as f:
                pickle.dump(detector, f)
        
        # Save configuration
        config = {
            'intents': self.intents,
            'rule_config': {
                'BASE_MIN_SCORE': BASE_MIN_SCORE,
                'AMBIGUITY_MARGIN': AMBIGUITY_MARGIN,
                'EXECUTION_MIN_SCORE': EXECUTION_MIN_SCORE,
                'MIN_TOKENS_OUT_OF_SCOPE': MIN_TOKENS_OUT_OF_SCOPE,
                'CONCURRENCE_THRESHOLD': CONCURRENCE_THRESHOLD
            },
            'rule_priority': RULE_PRIORITY,
            'rule_types': RULE_TYPES
        }
        
        config_path = model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved Level 1B model to {model_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """Load detectors and configuration"""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load configuration
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        intents = config['intents']
        
        # Load detectors
        detectors = {}
        for intent in intents:
            detector_path = model_dir / f"detector_{intent}.pkl"
            with open(detector_path, 'rb') as f:
                detectors[intent] = pickle.load(f)
        
        print(f"✓ Loaded Level 1B model from {model_dir}")
        print(f"  Detectors: {list(detectors.keys())}")
        
        return cls(detectors=detectors, intents=intents)


def get_configuration():
    """Get current rule configuration"""
    return {
        'BASE_MIN_SCORE': BASE_MIN_SCORE,
        'AMBIGUITY_MARGIN': AMBIGUITY_MARGIN,
        'EXECUTION_MIN_SCORE': EXECUTION_MIN_SCORE,
        'MIN_TOKENS_OUT_OF_SCOPE': MIN_TOKENS_OUT_OF_SCOPE,
        'CONCURRENCE_THRESHOLD': CONCURRENCE_THRESHOLD,
        'RULE_PRIORITY': RULE_PRIORITY,
        'RULE_TYPES': RULE_TYPES
    }
