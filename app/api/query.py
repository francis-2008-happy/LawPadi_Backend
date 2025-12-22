from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from pathlib import Path

from app.rag.generator import generate

router = APIRouter()

VECTOR_DIR = Path("vectorstore")
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.pkl"
VECTORIZER_PATH = VECTOR_DIR / "vectorizer.pkl"


class QueryRequest(BaseModel):
    question: str


@router.post("/")
def query_law(req: QueryRequest):
    """
    Accepts a legal question and returns a strictly grounded
    Nigerian-law-based answer with citations.
    """

    # -------- HANDLE GREETINGS --------
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    normalized_q = req.question.strip().lower().rstrip("!.,?")

    if normalized_q in greetings:
        return {
            "answer": "Hello! I am LawPadi, your Nigerian legal research assistant. How can I help you today?",
            "sources": []
        }

    if normalized_q in ["who made you", "who built you", "who created you"]:
        return {
            "answer": "I was developed by Francis Happy, An AI Engineer Base In Lagos State, Nigeria.",
            "sources": []
        }

    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=400, detail="Vector index not found. Run /ingest first."
        )

    # -------- LOAD INDEX & DATA --------
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # -------- EMBED QUERY --------
    query_vector = vectorizer.transform([req.question]).toarray().astype("float32")

    # -------- SEARCH --------
    k = 5
    distances, indices = index.search(query_vector, k)

    if len(indices[0]) == 0:
        return {"answer": "Insufficient Nigerian legal authority found.", "sources": []}

    retrieved_chunks = []

    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            item = metadata[idx]
            retrieved_chunks.append(item["text"])

    # -------- BUILD CONTEXT --------
    context = "\n\n".join(retrieved_chunks)

    # -------- GENERATE ANSWER --------
    answer = generate(req.question, context)

    # -------- FAIL-SAFE --------
    if not answer or "insufficient" in answer.lower() or "cannot find" in answer.lower():
        return {"answer": "Insufficient Nigerian legal authority found.", "sources": []}

    return {"answer": answer, "sources": []}
