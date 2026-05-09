# level3_5/intent_model.py

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = os.path.join(os.path.dirname(__file__), "intent_model.joblib")

class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=400, random_state=42)

    def train(self, df):
        X = self.vectorizer.fit_transform(df["utterance"].astype(str))
        y = df["intent"].astype(str)
        self.model.fit(X, y)
        joblib.dump((self.vectorizer, self.model), MODEL_PATH)

    def load(self):
        self.vectorizer, self.model = joblib.load(MODEL_PATH)

    def predict(self, utterance: str):
        X = self.vectorizer.transform([str(utterance)])
        probs = self.model.predict_proba(X)[0]
        idx = probs.argmax()
        intent = self.model.classes_[idx]
        conf = float(probs[idx])
        return intent, conf