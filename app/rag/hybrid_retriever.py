"""
Hybrid retrieval: combines BM25 (keyword) and FAISS (semantic) search.
"""
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple


class HybridRetriever:
    """
    Combines BM25 keyword search with FAISS semantic search.
    """
    
    def __init__(self, corpus: List[str], faiss_index, metadata: List[dict], embedder):
        """
        Args:
            corpus: List of text chunks (strings) for BM25
            faiss_index: FAISS index for semantic search
            metadata: Metadata for each chunk (includes text and path)
            embedder: Embedder instance for query vectorization
        """
        self.corpus = corpus
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedder = embedder
        
        # Tokenize corpus for BM25
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        top_k_candidates: int = 20
    ) -> Tuple[List[str], List[int]]:
        """
        Perform hybrid search combining semantic and keyword scoring.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            semantic_weight: Weight for semantic (FAISS) scores (0-1)
            keyword_weight: Weight for keyword (BM25) scores (0-1)
            top_k_candidates: Number of candidates to retrieve from each method
        
        Returns:
            Tuple of (retrieved_chunks, indices)
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # --- BM25 Keyword Search ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top candidates from BM25
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k_candidates]
        
        # --- FAISS Semantic Search ---
        query_vector = self.embedder.transform([query]).astype("float32")
        distances, faiss_indices = self.faiss_index.search(query_vector, top_k_candidates)
        
        # Convert FAISS L2 distances to similarity scores (lower distance = higher similarity)
        # Using: similarity = 1 / (1 + distance)
        faiss_similarities = 1.0 / (1.0 + distances[0])
        
        # --- Combine Scores ---
        # Create a dictionary to accumulate scores
        combined_scores = {}
        
        # Add BM25 scores (normalize to 0-1 range)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for idx in bm25_top_indices:
            normalized_bm25 = bm25_scores[idx] / max_bm25
            combined_scores[idx] = keyword_weight * normalized_bm25
        
        # Add FAISS scores (already normalized by conversion)
        max_faiss = max(faiss_similarities) if max(faiss_similarities) > 0 else 1.0
        for i, idx in enumerate(faiss_indices[0]):
            if idx != -1 and idx < len(self.metadata):
                normalized_faiss = faiss_similarities[i] / max_faiss
                if idx in combined_scores:
                    combined_scores[idx] += semantic_weight * normalized_faiss
                else:
                    combined_scores[idx] = semantic_weight * normalized_faiss
        
        # Sort by combined score and get top-k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Extract chunks and indices
        final_indices = [idx for idx, score in sorted_indices]
        retrieved_chunks = [self.metadata[idx]["text"] for idx in final_indices]
        
        return retrieved_chunks, final_indices
