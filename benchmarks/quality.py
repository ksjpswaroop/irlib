import numpy as np
from irlib import BM25Retriever, DenseRetriever, SearchPipeline
from typing import List

# Setup
docs = [
    "The quick brown fox jumps over the lazy dog.", # Doc 0
    "A fast brown fox leaps over a sleepy dog.",    # Doc 1 (Semantic match)
    "Machine learning and AI are changing the world.", # Doc 2
    "Natural language processing is a subset of AI.",  # Doc 3 (Semantic match)
]
queries = [
    "quick fox", # Lexical & Semantic
    "artificial intelligence" # Semantic (AI)
]
ground_truth = {
    "quick fox": [0, 1],
    "artificial intelligence": [2, 3]
}

def calculate_mrr(results: List[int], relevant: List[int]) -> float:
    """Mean Reciprocal Rank."""
    for rank, doc_id in enumerate(results):
        if doc_id in relevant:
            return 1 / (rank + 1)
    return 0.0

def run_quality_bench():
    retrievers = {
        "BM25": BM25Retriever(),
        "Dense": DenseRetriever(),
        "Hybrid": SearchPipeline()
    }
    
    for name, ret in retrievers.items():
        ret.index(docs)
        mrrs = []
        for q, expected in ground_truth.items():
            res = ret.search(q, top_k=2)
            # handle list of tuples (id, score) or list of ids
            ids = [r[0] if isinstance(r, tuple) else r for r in res]
            mrrs.append(calculate_mrr(ids, expected))
        print(f"{name} MRR: {np.mean(mrrs):.3f}")

if __name__ == "__main__":
    run_quality_bench()
