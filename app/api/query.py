from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from pathlib import Path
import re

from app.rag.generator import generate
from app.rag.embeddings import Embedder
from app.rag.hybrid_retriever import HybridRetriever

router = APIRouter()

VECTOR_DIR = Path("vectorstore")
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.pkl"
CORPUS_PATH = VECTOR_DIR / "corpus.pkl"

embedder = Embedder()

# Global retriever instance (lazy loaded)
_hybrid_retriever = None


class QueryRequest(BaseModel):
    question: str


@router.post("/")
def query_law(req: QueryRequest):
    """
    Accepts a legal question and returns a strictly grounded
    Nigerian-law-based answer with citations.
    """

    # -------- HANDLE GREETINGS --------
    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "how far", "what's up", "sup", "yo", "good day", 
        "how are you", "hiya", "hello there", "hi there"
    ]
    # Normalize: lowercase and remove punctuation except apostrophes (for "what's up")
    normalized_q = re.sub(r"[^\w\s']", "", req.question.lower()).strip()

    # Check for exact match or short greeting phrases (e.g., "Hi LawPadi")
    if normalized_q in greetings or (len(normalized_q) < 30 and any(normalized_q.startswith(g + " ") for g in greetings)):
        return {
            "answer": "Hello! I am LawPadi, your Nigerian legal research assistant. How can I help you today?",
            "sources": []
        } 

    if normalized_q in ["who made you", "who built you", "who created you"]:
        return {
            "answer": "I was developed by Francis Happy, An AI Engineer Base In Lagos State, Nigeria.",
            "sources": []
        }

    if not INDEX_PATH.exists() or not CORPUS_PATH.exists():
        raise HTTPException(
            status_code=400, detail="Vector index or corpus not found. Please run 'python -m app.rag.build_index' locally."
        )

    # -------- LOAD HYBRID RETRIEVER --------
    global _hybrid_retriever
    if _hybrid_retriever is None:
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        
        _hybrid_retriever = HybridRetriever(
            corpus=corpus,
            faiss_index=index,
            metadata=metadata,
            embedder=embedder
        )

    # -------- HYBRID SEARCH --------
    retrieved_chunks, indices = _hybrid_retriever.search(
        query=req.question,
        top_k=5,
        semantic_weight=0.6,  # 60% semantic, 40% keyword
        keyword_weight=0.4
    )

    # -------- BUILD CONTEXT --------
    context = "\n\n".join(retrieved_chunks)

    # -------- GENERATE ANSWER --------
    answer = generate(req.question, context)

    # -------- FAIL-SAFE --------
    if not answer or "insufficient" in answer.lower() or "cannot find" in answer.lower():
        return {"answer": "Insufficient Nigerian legal authority found.", "sources": []}

    return {"answer": answer, "sources": []}
