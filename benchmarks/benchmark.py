import time
import numpy as np
from irlib import BM25Retriever, DenseRetriever, SearchPipeline
from irlib.models import BM25Retriever as BM25

# Generate synthetic dataset for reproducibility
documents = [f"This is document number {i} regarding AI and search." for i in range(1000)]
queries = ["AI search", "information retrieval", "document number 500"]

# Setup retrievers
bm25 = BM25Retriever()
bm25.index(documents)

dense = DenseRetriever()
dense.index(documents)

pipeline = SearchPipeline(lexical_retriever=bm25, dense_retriever=dense)

def run_benchmark():
    results = {}
    for q in queries:
        start = time.perf_counter()
        bm25_res = bm25.search(q, top_k=5)
        bm25_time = time.perf_counter() - start
        
        start = time.perf_counter()
        dense_res = dense.search(q, top_k=5)
        dense_time = time.perf_counter() - start
        
        start = time.perf_counter()
        pipe_res = pipeline.search(q, top_k=5)
        pipe_time = time.perf_counter() - start
        
        results[q] = {
            "bm25_time": bm25_time,
            "dense_time": dense_time,
            "pipeline_time": pipe_time
        }
    return results

if __name__ == "__main__":
    benchmark_data = run_benchmark()
    for q, metrics in benchmark_data.items():
        print(f"Query: {q}")
        print(f"  BM25: {metrics['bm25_time']:.4f}s | Dense: {metrics['dense_time']:.4f}s | Pipeline: {metrics['pipeline_time']:.4f}s")
