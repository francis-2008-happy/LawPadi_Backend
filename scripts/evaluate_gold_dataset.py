#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pickle


DEFAULT_DATASET_PATH = "lawpadi_gold_dataset_100- json-file.json"
DEFAULT_VECTOR_DIR = "vectorstore"


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize_text(text))


def _f1_overlap(pred: str, gold: str) -> Dict[str, float]:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))

    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _difflib_ratio(a: str, b: str) -> float:
    return float(SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio())


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    import numpy as np

    a = vec_a.astype(np.float32)
    b = vec_b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _expected_basis_strings(expected_legal_basis: Any) -> List[str]:
    if not isinstance(expected_legal_basis, list):
        return []

    parts: List[str] = []
    for item in expected_legal_basis:
        if not isinstance(item, dict):
            continue
        law = str(item.get("law", "")).strip()
        sections = str(item.get("sections", "")).strip()
        if law:
            parts.append(law)
        if sections:
            parts.append(sections)
    return [p for p in parts if p]


def _basis_match(answer: str, expected_legal_basis: Any) -> Dict[str, Any]:
    basis_strings = _expected_basis_strings(expected_legal_basis)
    if not basis_strings:
        return {"has_expected_basis": None, "matched": []}

    normalized_answer = _normalize_text(answer)
    matched = []
    for s in basis_strings:
        if _normalize_text(s) and _normalize_text(s) in normalized_answer:
            matched.append(s)

    return {"has_expected_basis": bool(matched), "matched": matched}


@dataclass
class LawPadiRagClient:
    vector_dir: Path
    top_k: int
    semantic_weight: float
    keyword_weight: float

    def __post_init__(self) -> None:
        from app.rag.embeddings import Embedder

        self.vector_dir = Path(self.vector_dir)
        self.index_path = self.vector_dir / "index.faiss"
        self.meta_path = self.vector_dir / "meta.pkl"
        self.corpus_path = self.vector_dir / "corpus.pkl"

        self._embedder = Embedder()
        self._retriever = self._load_retriever()

    def _load_retriever(self) -> HybridRetriever:
        import faiss
        from app.rag.hybrid_retriever import HybridRetriever

        if not self.index_path.exists() or not self.meta_path.exists() or not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Vectorstore files missing. Expected: {self.index_path}, {self.meta_path}, {self.corpus_path}"
            )

        index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            metadata = pickle.load(f)
        with open(self.corpus_path, "rb") as f:
            corpus = pickle.load(f)

        return HybridRetriever(corpus=corpus, faiss_index=index, metadata=metadata, embedder=self._embedder)

    def answer(self, question: str) -> Tuple[str, str]:
        from app.rag.generator import generate

        retrieved_chunks, _indices = self._retriever.search(
            query=question,
            top_k=self.top_k,
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight,
        )
        context = "\n\n".join(retrieved_chunks)
        answer = generate(question, context)
        return answer, context

    def embed_similarity(self, a: str, b: str) -> float:
        import numpy as np

        vecs = self._embedder.transform([a, b])
        return _cosine_similarity(vecs[0], vecs[1])


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects")

    normalized: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {i} is not an object")
        if "question" not in row or "expected_answer" not in row:
            raise ValueError(f"Dataset row {i} missing required keys: question, expected_answer")
        normalized.append(row)
    return normalized


def _read_existing_predictions_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id", "")).strip()
            if qid:
                results[qid] = obj
    return results


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate LawPadi answers against a gold (human) Q/A dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to gold dataset JSON")
    parser.add_argument("--vector-dir", default=DEFAULT_VECTOR_DIR, help="Path to vectorstore directory")
    parser.add_argument("--out-dir", default="eval_outputs", help="Directory to write outputs")
    parser.add_argument("--run-name", default=None, help="Optional run name (defaults to timestamp)")

    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N items (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="Start index into dataset list")

    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks to retrieve")
    parser.add_argument("--semantic-weight", type=float, default=0.6)
    parser.add_argument("--keyword-weight", type=float, default=0.4)

    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip questions already present in predictions JSONL",
    )
    parser.add_argument(
        "--no-embed-metric",
        action="store_true",
        help="Disable embedding cosine similarity metric (faster, less dependencies at runtime)",
    )
    parser.add_argument(
        "--save-context",
        action="store_true",
        help="If set, store retrieved context in the predictions file (bigger outputs)",
    )

    filtered_argv = [
        a
        for a in sys.argv[1:]
        if not (a.startswith("http://_vscodecontentref_") or a.startswith("https://_vscodecontentref_"))
    ]
    args = parser.parse_args(filtered_argv)

    try:
        from app.config.settings import GROQ_API_KEY
    except Exception as e:
        raise SystemExit(f"Failed to import app config. Are you running from repo root? Error: {e!r}")

    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Put it in your environment or a .env file.")

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    vector_dir = Path(args.vector_dir)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_path = out_dir / f"{run_name}.predictions.jsonl"
    report_path = out_dir / f"{run_name}.report.json"
    report_csv_path = out_dir / f"{run_name}.report.csv"

    dataset = _load_dataset(dataset_path)

    end = len(dataset)
    if args.limit and args.limit > 0:
        end = min(end, args.start + args.limit)

    slice_items = dataset[args.start:end]

    existing = _read_existing_predictions_jsonl(predictions_path) if args.resume else {}

    try:
        client = LawPadiRagClient(
            vector_dir=vector_dir,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
        )
    except ImportError as e:
        raise SystemExit(
            "Dependency import failed (often numpy/faiss). "
            "Make sure you're using the venv interpreter (./venv/bin/python) and reinstall requirements. "
            f"Original error: {e!r}"
        )

    per_item: List[Dict[str, Any]] = []

    total = len(slice_items)
    asked = 0
    skipped = 0
    errors = 0

    for idx, row in enumerate(slice_items):
        qid = str(row.get("id", f"ROW_{args.start + idx:04d}"))
        question = str(row.get("question", ""))
        expected_answer = str(row.get("expected_answer", ""))
        expected_legal_basis = row.get("expected_legal_basis")

        if args.resume and qid in existing and existing[qid].get("model_answer"):
            skipped += 1
            model_answer = str(existing[qid].get("model_answer", ""))
            context = str(existing[qid].get("context", ""))
        else:
            asked += 1
            t0 = time.time()
            try:
                model_answer, context = client.answer(question)
                elapsed_s = time.time() - t0

                pred_obj: Dict[str, Any] = {
                    "id": qid,
                    "question": question,
                    "model_answer": model_answer,
                    "created_at": _utc_now_iso(),
                    "latency_s": elapsed_s,
                }
                if args.save_context:
                    pred_obj["context"] = context

                _append_jsonl(predictions_path, pred_obj)
            except Exception as e:
                errors += 1
                pred_obj = {
                    "id": qid,
                    "question": question,
                    "error": repr(e),
                    "created_at": _utc_now_iso(),
                }
                _append_jsonl(predictions_path, pred_obj)
                model_answer = ""
                context = ""

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

        f1 = _f1_overlap(model_answer, expected_answer)
        ratio = _difflib_ratio(model_answer, expected_answer)
        basis = _basis_match(model_answer, expected_legal_basis)

        embed_sim: Optional[float] = None
        if not args.no_embed_metric and model_answer and expected_answer:
            try:
                embed_sim = client.embed_similarity(model_answer, expected_answer)
            except Exception:
                embed_sim = None

        item_result: Dict[str, Any] = {
            "id": qid,
            "category": row.get("category"),
            "confidence": row.get("confidence"),
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "metrics": {
                "token_f1": f1,
                "difflib_ratio": ratio,
                "embedding_cosine": embed_sim,
            },
            "legal_basis": {
                "expected": expected_legal_basis,
                "match": basis,
            },
        }

        per_item.append(item_result)

    def _avg(values: Iterable[float]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    summary = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "vector_dir": str(vector_dir),
        "created_at": _utc_now_iso(),
        "range": {"start": args.start, "end": end, "count": total},
        "counts": {"asked": asked, "skipped": skipped, "errors": errors},
        "params": {
            "top_k": args.top_k,
            "semantic_weight": args.semantic_weight,
            "keyword_weight": args.keyword_weight,
            "sleep": args.sleep,
            "resume": bool(args.resume),
            "no_embed_metric": bool(args.no_embed_metric),
            "save_context": bool(args.save_context),
        },
        "averages": {
            "token_f1": _avg(item["metrics"]["token_f1"]["f1"] for item in per_item),
            "token_precision": _avg(item["metrics"]["token_f1"]["precision"] for item in per_item),
            "token_recall": _avg(item["metrics"]["token_f1"]["recall"] for item in per_item),
            "difflib_ratio": _avg(item["metrics"]["difflib_ratio"] for item in per_item),
            "embedding_cosine": _avg(
                item["metrics"]["embedding_cosine"]
                for item in per_item
                if item["metrics"]["embedding_cosine"] is not None
            ),
            "legal_basis_match_rate": _avg(
                1.0 if item["legal_basis"]["match"]["has_expected_basis"] else 0.0
                for item in per_item
                if item["legal_basis"]["match"]["has_expected_basis"] is not None
            ),
        },
    }

    report_obj = {"summary": summary, "items": per_item}

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "confidence",
                "token_f1",
                "difflib_ratio",
                "embedding_cosine",
                "legal_basis_match",
                "question",
            ],
        )
        writer.writeheader()
        for item in per_item:
            writer.writerow(
                {
                    "id": item["id"],
                    "category": item.get("category"),
                    "confidence": item.get("confidence"),
                    "token_f1": item["metrics"]["token_f1"]["f1"],
                    "difflib_ratio": item["metrics"]["difflib_ratio"],
                    "embedding_cosine": item["metrics"]["embedding_cosine"],
                    "legal_basis_match": item["legal_basis"]["match"]["has_expected_basis"],
                    "question": item["question"],
                }
            )

    print("Wrote predictions:", predictions_path)
    print("Wrote report JSON:", report_path)
    print("Wrote report CSV:", report_csv_path)
    print("Summary averages:", json.dumps(summary["averages"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
