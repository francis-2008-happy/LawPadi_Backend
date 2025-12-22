import faiss
import numpy as np
from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.rag.embeddings import TfidfEmbedder
from app.rag.retriever import save as save_index


def build_index():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    chunks = []
    metadata = []

    print("Chunking documents...")
    for doc in documents:
        doc_chunks = chunk_text(doc["text"])
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadata.append({"text": chunk, "path": doc["path"]})

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No chunks to process.")
        return

    print("Embedding chunks...")
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(chunks)
    embedder.save()

    print("Building index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors.astype(np.float32))

    print("Saving index...")
    save_index(index, metadata)
    print("Index built and saved successfully.")


if __name__ == "__main__":
    build_index()