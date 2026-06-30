"""RAG retrieval benchmark loading, execution, and evaluation.

The benchmark harness evaluates the retrieval/context-selection part of RAG.
It does not generate answers or require an LLM judge.

Pseudocode:
    dataset = load benchmark corpus, queries, and qrels
    retriever.index(corpus_documents)
    for query in queries: retrieve top-k document ids
    compare retrieved ids with qrels using IR metrics
    write JSON and Markdown summaries
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from irlib.core import Document
from irlib.dense import DenseRetriever, HashingEncoder
from irlib.hybrid import HybridRetriever
from irlib.models import BM25Retriever, TFIDFRetriever


BEIR_DATASET_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}


@dataclass()
class BenchmarkDataset:
    """A retrieval benchmark dataset.

    Pseudocode:
        corpus[doc_id] = text
        queries[query_id] = query text
        qrels[query_id][doc_id] = relevance grade

    Use this as the common shape for BEIR, RAGBench-derived, and local fixtures.
    """

    name: str
    source: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def sample(self, sample_size: int) -> "BenchmarkDataset":
        """Return the first `sample_size` judged queries and the same corpus."""

        if sample_size <= 0 or sample_size >= len(self.queries):
            return self
        query_ids = [qid for qid in sorted(self.queries) if qid in self.qrels][:sample_size]
        return BenchmarkDataset(
            name=self.name,
            source=self.source,
            corpus=dict(self.corpus),
            queries={qid: self.queries[qid] for qid in query_ids},
            qrels={qid: self.qrels[qid] for qid in query_ids},
        )

    def to_documents(self) -> tuple[list[Document], dict[str, int], dict[int, str]]:
        """Convert external string ids into irlib integer document ids."""

        documents: list[Document] = []
        external_to_internal: dict[str, int] = {}
        internal_to_external: dict[int, str] = {}
        for internal_id, external_id in enumerate(sorted(self.corpus)):
            external_to_internal[external_id] = internal_id
            internal_to_external[internal_id] = external_id
            documents.append(
                Document(
                    id=internal_id,
                    text=self.corpus[external_id],
                    metadata={"external_id": external_id, "benchmark": self.name, "source": self.source},
                )
            )
        return documents, external_to_internal, internal_to_external


def get_benchmark_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the benchmark cache directory."""

    return Path(cache_dir or os.environ.get("IRLIB_BENCHMARK_CACHE", ".cache/irlib-benchmarks"))


def download_beir_dataset(name: str, cache_dir: str | Path | None = None) -> Path:
    """Download and extract a BEIR dataset if missing.

    Pseudocode:
        url = BEIR_DATASET_URLS[name]
        download zip into cache
        extract safely into cache/name
        return extracted dataset directory
    """

    normalized = name.lower()
    if normalized not in BEIR_DATASET_URLS:
        raise ValueError(f"Unsupported BEIR dataset: {name}. Choose from {sorted(BEIR_DATASET_URLS)}.")
    cache = get_benchmark_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dataset_dir = cache / normalized
    corpus_path = dataset_dir / "corpus.jsonl"
    if corpus_path.exists():
        return dataset_dir

    zip_path = cache / f"{normalized}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(BEIR_DATASET_URLS[normalized], zip_path)
    _safe_extract_zip(zip_path, cache)
    if corpus_path.exists():
        return dataset_dir

    nested = cache / normalized
    if nested.exists():
        return nested
    raise FileNotFoundError(f"BEIR dataset extraction did not produce {corpus_path}")


def load_beir_dataset(
    name: str = "scifact",
    *,
    split: str = "test",
    sample_size: int = 100,
    cache_dir: str | Path | None = None,
) -> BenchmarkDataset:
    """Load a BEIR dataset as `BenchmarkDataset`."""

    dataset_dir = download_beir_dataset(name, cache_dir=cache_dir)
    return load_beir_directory(dataset_dir, name=name, split=split, sample_size=sample_size)


def load_beir_directory(
    dataset_dir: str | Path,
    *,
    name: str = "local",
    split: str = "test",
    sample_size: int = 100,
) -> BenchmarkDataset:
    """Load a local BEIR-format directory.

    The expected shape is `corpus.jsonl`, `queries.jsonl`, and
    `qrels/<split>.tsv`.
    """

    dataset_dir = Path(dataset_dir)
    corpus = _read_beir_corpus(dataset_dir / "corpus.jsonl")
    queries = _read_beir_queries(dataset_dir / "queries.jsonl")
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        candidates = sorted((dataset_dir / "qrels").glob("*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"No qrels TSV files found under {dataset_dir / 'qrels'}")
        qrels_path = candidates[0]
    qrels = _read_beir_qrels(qrels_path)
    judged_queries = {qid: query for qid, query in queries.items() if qid in qrels}
    dataset = BenchmarkDataset(
        name=name.lower(),
        source=f"beir:{qrels_path.stem}",
        corpus=corpus,
        queries=judged_queries,
        qrels=qrels,
    )
    return dataset.sample(sample_size)


def load_ragbench_dataset(
    name: str = "covidqa",
    *,
    split: str = "test",
    sample_size: int = 100,
) -> BenchmarkDataset:
    """Load a RAGBench subset through Hugging Face `datasets`.

    Pseudocode:
        load dataset config and split
        treat each provided context as a relevant document for its question
        return retrieval-style corpus, queries, and qrels

    This adapter is intentionally defensive because RAGBench subsets may expose
    slightly different column names.
    """

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError("Install `datasets` to use the RAGBench adapter.") from exc

    last_error: Exception | None = None
    for repo in ("galileo-ai/ragbench", "rungalileo/ragbench"):
        try:
            rows = load_dataset(repo, name, split=split)
            return _ragbench_rows_to_dataset(rows, name=name, source=repo, sample_size=sample_size)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load RAGBench subset {name!r}") from last_error


def build_retriever(name: str):
    """Create a benchmark retriever by short name."""

    normalized = name.strip().lower()
    if normalized == "bm25":
        return BM25Retriever()
    if normalized == "tfidf":
        return TFIDFRetriever()
    if normalized == "dense-hash":
        return DenseRetriever(encoder=HashingEncoder())
    if normalized == "hybrid-hash":
        return HybridRetriever(dense_retriever=DenseRetriever(encoder=HashingEncoder()))
    raise ValueError("Unknown retriever. Choose from bm25, tfidf, dense-hash, hybrid-hash.")


def run_benchmark(
    dataset: BenchmarkDataset,
    retriever_names: Sequence[str] = ("bm25", "tfidf", "hybrid-hash"),
    *,
    k_values: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    """Run retrievers against a benchmark dataset and return JSON-ready results."""

    max_k = max(k_values)
    documents, external_to_internal, internal_to_external = dataset.to_documents()
    qrels_internal = _map_qrels_to_internal(dataset.qrels, external_to_internal)
    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset.name,
            "source": dataset.source,
            "query_count": len(dataset.queries),
            "document_count": len(dataset.corpus),
        },
        "k_values": list(k_values),
        "retrievers": {},
    }

    for retriever_name in retriever_names:
        retriever = build_retriever(retriever_name)
        start = time.perf_counter()
        retriever.index(documents)
        indexing_seconds = time.perf_counter() - start

        runs: dict[str, list[int]] = {}
        query_start = time.perf_counter()
        for query_id, query_text in dataset.queries.items():
            runs[query_id] = [doc_id for doc_id, _score in retriever.search(query_text, top_k=max_k)]
        query_seconds = time.perf_counter() - query_start

        metrics = evaluate_retrieval(runs, qrels_internal, k_values=k_values)
        output["retrievers"][retriever_name] = {
            "metrics": metrics,
            "indexing_seconds": indexing_seconds,
            "total_query_seconds": query_seconds,
            "mean_query_latency_ms": (query_seconds / max(1, len(dataset.queries))) * 1000,
            "sample_results": _external_sample_results(runs, internal_to_external),
        }
    return output


def evaluate_retrieval(
    runs: Mapping[str, Sequence[int]],
    qrels: Mapping[str, Mapping[int, int]],
    *,
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Evaluate ranked retrieval output against qrels."""

    metrics: dict[str, float] = {}
    query_ids = [qid for qid in runs if qid in qrels]
    for k in k_values:
        precision_values = []
        recall_values = []
        mrr_values = []
        ap_values = []
        ndcg_values = []
        for query_id in query_ids:
            ranked = list(runs[query_id])[:k]
            relevant = {doc_id for doc_id, rel in qrels[query_id].items() if rel > 0}
            gains = qrels[query_id]
            precision_values.append(_precision_at_k(ranked, relevant, k))
            recall_values.append(_recall_at_k(ranked, relevant))
            mrr_values.append(_mrr_at_k(ranked, relevant))
            ap_values.append(_average_precision_at_k(ranked, relevant, k))
            ndcg_values.append(_ndcg_at_k(ranked, gains, k))
        suffix = f"@{k}"
        metrics[f"precision{suffix}"] = _mean(precision_values)
        metrics[f"recall{suffix}"] = _mean(recall_values)
        metrics[f"mrr{suffix}"] = _mean(mrr_values)
        metrics[f"map{suffix}"] = _mean(ap_values)
        metrics[f"ndcg{suffix}"] = _mean(ndcg_values)
    return metrics


def write_benchmark_outputs(results: Mapping[str, Any], output_path: str | Path) -> tuple[Path, Path]:
    """Write JSON results and a Markdown summary."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = output.parent / "README.md"
    summary_path.write_text(format_benchmark_summary(results, output.name), encoding="utf-8")
    return output, summary_path


def format_benchmark_summary(results: Mapping[str, Any], json_file_name: str | None = None) -> str:
    """Format benchmark results as Markdown."""

    dataset = results["dataset"]
    lines = [
        "# RAG Retrieval Benchmark Results",
        "",
        f"- Dataset: `{dataset['name']}` ({dataset['source']})",
        f"- Queries: `{dataset['query_count']}`",
        f"- Documents: `{dataset['document_count']}`",
        f"- Created at: `{results['created_at']}`",
    ]
    if json_file_name:
        lines.append(f"- JSON: `{json_file_name}`")
    lines.extend(["", "| Retriever | NDCG@10 | Recall@10 | MRR@10 | MAP@10 | Mean latency ms |", "|---|---:|---:|---:|---:|---:|"])
    for name, payload in results["retrievers"].items():
        metrics = payload["metrics"]
        lines.append(
            f"| `{name}` | {metrics.get('ndcg@10', 0.0):.4f} | {metrics.get('recall@10', 0.0):.4f} | "
            f"{metrics.get('mrr@10', 0.0):.4f} | {metrics.get('map@10', 0.0):.4f} | "
            f"{payload['mean_query_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "This benchmark evaluates retrieval/context selection only. It does not evaluate generated answer quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run irlib RAG retrieval benchmarks.")
    parser.add_argument("--source", choices=["beir", "ragbench"], default="beir")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=100, help="0 means full dataset.")
    parser.add_argument("--retrievers", default="bm25,tfidf,hybrid-hash")
    parser.add_argument("--k", default="5,10", help="Comma-separated k values.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default="benchmarks/results/scifact_sample.json")
    args = parser.parse_args(argv)

    sample_size = args.sample_size
    retrievers = [item.strip() for item in args.retrievers.split(",") if item.strip()]
    k_values = [int(item.strip()) for item in args.k.split(",") if item.strip()]
    if args.source == "beir":
        dataset = load_beir_dataset(args.dataset, split=args.split, sample_size=sample_size, cache_dir=args.cache_dir)
    else:
        dataset = load_ragbench_dataset(args.dataset, split=args.split, sample_size=sample_size)
    results = run_benchmark(dataset, retriever_names=retrievers, k_values=k_values)
    json_path, summary_path = write_benchmark_outputs(results, args.output)
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_beir_corpus(path: Path) -> dict[str, str]:
    corpus = {}
    for row in _read_jsonl(path):
        doc_id = str(row["_id"])
        text = " ".join(part for part in [row.get("title", ""), row.get("text", "")] if part)
        corpus[doc_id] = text
    return corpus


def _read_beir_queries(path: Path) -> dict[str, str]:
    return {str(row["_id"]): str(row["text"]) for row in _read_jsonl(path)}


def _read_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if line_number == 0 and parts[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            query_id, doc_id, relevance = str(parts[0]), str(parts[1]), int(float(parts[2]))
            qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = (target_dir / member.filename).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(target_dir)


def _ragbench_rows_to_dataset(rows: Any, *, name: str, source: str, sample_size: int) -> BenchmarkDataset:
    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    limit = sample_size if sample_size > 0 else len(rows)
    for i, row in enumerate(rows):
        if i >= limit:
            break
        query = _first_present(row, ["question", "query", "prompt", "user_input"]) or ""
        contexts = _extract_contexts(row)
        query_id = str(row.get("id", row.get("question_id", i))) if isinstance(row, Mapping) else str(i)
        queries[query_id] = str(query)
        qrels[query_id] = {}
        for j, context in enumerate(contexts):
            doc_id = f"{query_id}:{j}"
            corpus[doc_id] = context
            qrels[query_id][doc_id] = 1
    return BenchmarkDataset(name=name, source=f"ragbench:{source}", corpus=corpus, queries=queries, qrels=qrels)


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in row and row[key]:
            return row[key]
    return None


def _extract_contexts(row: Mapping[str, Any]) -> list[str]:
    value = _first_present(row, ["documents", "contexts", "context", "passages", "retrieved_contexts"])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    contexts = []
    for item in value:
        if isinstance(item, str):
            contexts.append(item)
        elif isinstance(item, Mapping):
            text = _first_present(item, ["text", "content", "document", "passage"])
            if text:
                contexts.append(str(text))
    return contexts


def _map_qrels_to_internal(
    qrels: Mapping[str, Mapping[str, int]],
    external_to_internal: Mapping[str, int],
) -> dict[str, dict[int, int]]:
    mapped: dict[str, dict[int, int]] = {}
    for query_id, judgments in qrels.items():
        mapped[query_id] = {
            external_to_internal[doc_id]: rel
            for doc_id, rel in judgments.items()
            if doc_id in external_to_internal
        }
    return mapped


def _external_sample_results(
    runs: Mapping[str, Sequence[int]],
    internal_to_external: Mapping[int, str],
    *,
    max_queries: int = 3,
    max_docs: int = 5,
) -> dict[str, list[str]]:
    sample: dict[str, list[str]] = {}
    for query_id in sorted(runs)[:max_queries]:
        sample[query_id] = [internal_to_external[doc_id] for doc_id in runs[query_id][:max_docs]]
    return sample


def _precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / k


def _recall_at_k(ranked: Sequence[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc_id in ranked if doc_id in relevant) / len(relevant)


def _mrr_at_k(ranked: Sequence[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def _dcg(gains: Sequence[float]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _ndcg_at_k(ranked: Sequence[int], gains: Mapping[int, int], k: int) -> float:
    actual = [gains.get(doc_id, 0) for doc_id in ranked[:k]]
    ideal = sorted((gain for gain in gains.values() if gain > 0), reverse=True)[:k]
    ideal_score = _dcg(ideal)
    if ideal_score == 0:
        return 0.0
    return _dcg(actual) / ideal_score


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
