from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        # all-MiniLM-L6-v2 is a fast and accurate model for semantic search
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit_transform(self, texts):
        # SentenceTransformers are pre-trained; we just encode.
        return self.transform(texts)

    def transform(self, texts):
        # Returns a numpy array of embeddings
        return self.model.encode(texts, convert_to_numpy=True)
