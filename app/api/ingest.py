from fastapi import APIRouter, HTTPException
import faiss
import numpy as np
import pickle
from pathlib import Path

from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.utils.text_cleaning import clean_text
from app.utils.metadata import extract_metadata
from app.rag.embeddings import Embedder

router = APIRouter()

VECTOR_DIR = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)

INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.pkl"


@router.post("/")
def ingest_documents():
    """
    Scans /data directory, chunks Nigerian legal documents,
    builds TF-IDF embeddings, and stores them in FAISS.
    """

    documents = load_documents()

    if not documents:
        raise HTTPException(status_code=400, detail="No legal documents found in /data")

    all_chunks = []
    all_metadata = []

    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        chunks = chunk_text(cleaned_text)

        base_metadata = extract_metadata(doc["path"])

        for chunk in chunks:
            all_chunks.append(chunk)
            # Create metadata dict containing the text chunk
            chunk_meta = base_metadata.copy() if base_metadata else {}
            chunk_meta.update({"text": chunk, "path": doc["path"]})
            all_metadata.append(chunk_meta)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No valid text chunks generated")

    # -------- TF-IDF EMBEDDING --------
    embedder = Embedder()
    vectors = embedder.fit_transform(all_chunks)

    vectors = np.array(vectors).astype("float32")

    # -------- FAISS INDEX --------
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # -------- SAVE EVERYTHING --------
    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(all_metadata, f)

    return {
        "status": "success",
        "documents_processed": len(documents),
        "chunks_indexed": len(all_chunks),
    }
