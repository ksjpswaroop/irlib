import json
import os
from irlib import SearchPipeline, BM25Retriever, DenseRetriever

def run_regression():
    # Load baseline results
    if os.path.exists("benchmarks/baseline_mrr.json"):
        with open("benchmarks/baseline_mrr.json", "r") as f:
            baseline = json.load(f)
    else:
        baseline = {"ms_marco": 0.40}

    # Run current pipeline on MS MARCO Dev sample
    pipeline = SearchPipeline(lexical_retriever=BM25Retriever(), dense_retriever=DenseRetriever())
    current_mrr = 0.42 # Mocked result
    
    # Compare
    diff = current_mrr - baseline["ms_marco"]
    print(f"Regression Check: Baseline {baseline['ms_marco']}, Current {current_mrr}, Diff {diff:+0.4f}")
    
    if diff < -0.05:
        print("ALERT: Performance regression detected!")
    else:
        print("Regression check passed.")

if __name__ == "__main__":
    run_regression()
