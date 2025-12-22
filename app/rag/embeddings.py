import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

VECTORIZER_PATH = "vectorstore/vectorizer.pkl"


class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    def fit_transform(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        return vectors.toarray()

    def transform(self, texts):
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()

    def save(self):
        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
