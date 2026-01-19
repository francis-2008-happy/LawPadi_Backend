import os

# Hugging Face Hub defaults the metadata (ETag/HEAD) timeout to ~10s.
# On slower connections this causes repeated ReadTimeoutError during model load.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", os.getenv("HF_HUB_ETAG_TIMEOUT", "120"))
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT", os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "300")
)

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
