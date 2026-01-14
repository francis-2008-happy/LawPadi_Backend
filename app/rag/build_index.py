import faiss
import numpy as np
import pickle
import os
from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.rag.embeddings import Embedder
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
    embedder = Embedder()
    vectors = embedder.fit_transform(chunks)

    print("Building index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors.astype(np.float32))

    print("Saving index...")
    save_index(index, metadata)
    
    # Save corpus for BM25 keyword search
    corpus_path = "vectorstore/corpus.pkl"
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Corpus saved to {corpus_path}")
    
    print("Index built and saved successfully.")


if __name__ == "__main__":
    build_index()