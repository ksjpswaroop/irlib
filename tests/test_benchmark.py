import json
import os
from pathlib import Path

import pytest

from irlib.benchmark import BenchmarkDataset, evaluate_retrieval, load_beir_dataset, load_beir_directory, run_benchmark


def test_evaluate_retrieval_metrics_are_deterministic():
    runs = {"q1": [1, 2, 3], "q2": [2, 3, 1]}
    qrels = {"q1": {2: 1}, "q2": {2: 1, 3: 1}}

    metrics = evaluate_retrieval(runs, qrels, k_values=(1, 2))

    assert metrics["precision@1"] == 0.5
    assert metrics["recall@1"] == 0.25
    assert metrics["mrr@2"] == 0.75
    assert metrics["ndcg@2"] > 0


def test_run_benchmark_on_tiny_dataset():
    dataset = BenchmarkDataset(
        name="tiny",
        source="local",
        corpus={"d1": "python retrieval tutorial", "d2": "java enterprise search"},
        queries={"q1": "python retrieval"},
        qrels={"q1": {"d1": 1}},
    )

    results = run_benchmark(dataset, retriever_names=("bm25", "tfidf", "hybrid-hash"), k_values=(1, 2))

    assert results["dataset"]["query_count"] == 1
    assert results["dataset"]["document_count"] == 2
    assert set(results["retrievers"]) == {"bm25", "tfidf", "hybrid-hash"}
    assert results["retrievers"]["bm25"]["metrics"]["recall@1"] == 1.0


def test_load_local_beir_directory(tmp_path):
    dataset_dir = tmp_path / "beir"
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (dataset_dir / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "d1", "title": "Python", "text": "retrieval benchmark"}),
                json.dumps({"_id": "d2", "title": "Java", "text": "enterprise search"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "python retrieval"}) + "\n",
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

    dataset = load_beir_directory(dataset_dir, name="fixture", sample_size=10)

    assert dataset.name == "fixture"
    assert dataset.corpus["d1"] == "Python retrieval benchmark"
    assert dataset.queries["q1"] == "python retrieval"
    assert dataset.qrels["q1"]["d1"] == 1


@pytest.mark.benchmark
def test_download_scifact_benchmark_when_enabled(tmp_path):
    if os.environ.get("IRLIB_RUN_BENCHMARKS") != "1":
        pytest.skip("Set IRLIB_RUN_BENCHMARKS=1 to download and test BEIR SciFact.")
    dataset = load_beir_dataset("scifact", sample_size=2, cache_dir=tmp_path)
    assert dataset.queries
    assert dataset.corpus
    assert dataset.qrels
