from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import faiss
import pickle
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


@router.get("")
@router.get("/")
def query_help():
    return {
        "message": "Use POST /query with JSON: { 'question': '...' }",
        "example": {
            "method": "POST",
            "path": "/query",
            "json": {"question": "What is the legal effect of a contract signed under duress?"},
        },
    }


@router.post("")
@router.post("/")
def query_law(req: QueryRequest):
    """
    Accepts a legal question and returns a strictly grounded
    Nigerian-law-based answer with citations.
    """

    # -------- HANDLE GREETINGS / SMALL TALK --------
    # Normalize: lowercase, remove punctuation except apostrophes, collapse whitespace.
    normalized_q = re.sub(r"[^\w\s']", "", req.question.lower())
    normalized_q = re.sub(r"\s+", " ", normalized_q).strip()
    normalized_q_no_apostrophe = normalized_q.replace("'", "")

    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "greetings",
        "hiya",
        "yo",
        "sup",
        "hello there",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "how far",
        "whats up",
        "what's up",
    }

    def _is_small_talk(text: str, text_no_apostrophe: str) -> bool:
        if not text:
            return False
        if text in greeting_phrases or text_no_apostrophe in {p.replace("'", "") for p in greeting_phrases}:
            return True
        if len(text) < 50 and any(text.startswith(p + " ") for p in greeting_phrases):
            return True
        # Wellbeing / casual check-ins (e.g., "how are you doing today")
        if text.startswith("how are you") or text.startswith("how you dey"):
            return True
        return False

    if _is_small_talk(normalized_q, normalized_q_no_apostrophe):
        if normalized_q.startswith("how are you") or normalized_q.startswith("how you dey"):
            return {"answer": "I’m well, thank you. How can I help you today?", "sources": []}
        if normalized_q.startswith("good morning"):
            return {"answer": "Good morning. How can I help you today?", "sources": []}
        if normalized_q.startswith("good afternoon"):
            return {"answer": "Good afternoon. How can I help you today?", "sources": []}
        if normalized_q.startswith("good evening"):
            return {"answer": "Good evening. How can I help you today?", "sources": []}
        return {"answer": "Hello. How can I help you today?", "sources": []}

    # -------- HANDLE IDENTITY / CREATOR QUESTIONS --------
    if normalized_q in ["who are you", "what are you", "what can you do", "what do you do"]:
        return {
            "answer": "I am LawPadi, a Nigerian legal research assistant. Ask a Nigerian law question and I will answer with citations to the relevant authority.",
            "sources": [],
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
