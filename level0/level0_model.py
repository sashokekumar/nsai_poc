"""
Level 0: Baseline TF-IDF Intent Classification

Simple TF-IDF + Logistic Regression intent classifier with confidence thresholding.
This is the baseline model without neuro-symbolic rules.
"""

import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class Level0Classifier:
    """Baseline TF-IDF + Logistic Regression Intent Classifier"""
    
    def __init__(self, vectorizer=None, classifier=None, label_encoder=None, confidence_threshold=0.7):
        """
        Initialize Level 0 classifier
        
        Args:
            vectorizer: Trained TfidfVectorizer
            classifier: Trained LogisticRegression model
            label_encoder: Fitted LabelEncoder
            confidence_threshold: Minimum confidence to not abstain (default 0.7)
        """
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.confidence_threshold = confidence_threshold
        
    @staticmethod
    def preprocess_text(text):
        """Preprocess text for classification"""
        text = text.lower()
        text = re.sub(r'^"|"$', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, utterance, return_probabilities=False):
        """
        Predict intent for a single utterance
        
        Args:
            utterance: Text string to classify
            return_probabilities: If True, include full probability distribution
        
        Returns:
            Dictionary with prediction results
        """
        if not self.classifier or not self.vectorizer or not self.label_encoder:
            raise ValueError("Model not trained or loaded. Train or load model first.")
        
        # Preprocess
        processed = self.preprocess_text(utterance)
        
        # Vectorize
        X_vec = self.vectorizer.transform([processed])
        
        # Predict
        y_pred = self.classifier.predict(X_vec)[0]
        y_proba = self.classifier.predict_proba(X_vec)[0]
        
        # Get predictions
        max_conf = float(np.max(y_proba))
        pred_idx = int(np.argmax(y_proba))
        pred_intent = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Apply confidence threshold
        abstained = max_conf < self.confidence_threshold
        output_intent = pred_intent if not abstained else "unknown"
        
        result = {
            "utterance": utterance,
            "intent": output_intent,
            "confidence": max_conf,
            "abstained": abstained,
            "model": "tfidf-linear-v1",
            "version": "2026.02"
        }
        
        if return_probabilities:
            probabilities = {}
            for i, intent in enumerate(self.label_encoder.classes_):
                probabilities[intent] = float(y_proba[i])
            result['probabilities'] = probabilities
        
        return result
    
    def predict_batch(self, utterances, return_probabilities=False):
        """
        Predict intents for multiple utterances
        
        Args:
            utterances: List of text strings
            return_probabilities: If True, include full probability distribution
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for utterance in utterances:
            result = self.predict(utterance, return_probabilities=return_probabilities)
            results.append(result)
        return results
    
    def save(self, model_dir):
        """Save model artifacts"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'logistic_regression_classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(model_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(model_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✓ Saved Level 0 model to {model_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """Load model artifacts"""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        with open(model_dir / 'logistic_regression_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        with open(model_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(model_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"✓ Loaded Level 0 model from {model_dir}")
        
        return cls(vectorizer=vectorizer, classifier=classifier, label_encoder=label_encoder)
