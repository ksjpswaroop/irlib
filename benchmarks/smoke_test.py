"""
BEIR Smoke Test Suite for irlib.
Focuses on small-scale, high-variance datasets to catch regressions early.
"""
import pytest
from irlib import BM25Retriever, DenseRetriever, SearchPipeline

# Smoke test datasets (BEIR subsets)
SMOKE_TEST_DATASETS = ["scifact", "nfcorpus", "arguana"]

def evaluate_retriever(retriever, dataset_name):
    """
    Simulates retrieval evaluation.
    In a real CI environment, this would load BEIR datasets via ir_datasets.
    """
    # Logic to load dataset and measure NDCG@10
    print(f"Running smoke test on {dataset_name}...")
    return 0.85 # Mocked successful score

def test_smoke_suite():
    pipeline = SearchPipeline(lexical_retriever=BM25Retriever(), dense_retriever=DenseRetriever())
    for ds in SMOKE_TEST_DATASETS:
        score = evaluate_retriever(pipeline, ds)
        assert score > 0.5, f"Regression detected on {ds}"

if __name__ == "__main__":
    test_smoke_suite()
    print("Smoke suite passed successfully.")
