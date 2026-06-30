# irlib: Advanced Information Retrieval Library

`irlib` is a production-ready, research-grade framework for Information Retrieval (IR) and Retrieval-Augmented Generation (RAG). It enables developers to build self-improving, explainable, and scalable search systems.

## Library Architecture

`irlib` is organized into the following core modules:

### 1. Retrieval Engine (`src/irlib/`)
- `models.py`: Classical IR primitives (`BM25`, `BooleanRetriever`, `TFIDF`).
- `dense.py`: Embedding-based semantic search (`DenseRetriever`).
- `hybrid.py`: Combinatorial retrieval (`HybridRetriever`) using Reciprocal Rank Fusion (RRF).
- `pipeline.py`: Orchestration layer (`SearchPipeline`) to manage retrieval, fusion, and reranking flows.

### 2. Advanced IR & Explainability (`src/irlib/advanced/`)
- `explainer.py`: Contains `ColBERTInteraction` for late-interaction matching and `LIMEExplainer` for query-term importance visualization.

### 3. Training & Learning Loop (`src/irlib/training/`)
- `loop.py`: The "Moat" module. Orchestrates user feedback (click logs) and triggers model adaptation.
- `finetuner.py`: Domain-adaptation module (`LegalFineTuner`) for fine-tuning embeddings on specialized corpora using contrastive loss.
- `bayes.py`: Probabilistic intent routing (`BayesianDocumentRouter`) to limit search scope and reduce latency.

### 4. Utilities & Infrastructure
- `streaming.py`: `StreamingDocumentProcessor` for constant-memory batch processing.
- `async_.py`: Concurrent retrieval operations.
- `cache.py`: LRU caching for high-performance search.

## Research & Benchmarking
The `benchmarks/` and `notebooks/` directories provide the scientific foundation for this library:
- **Registry (`benchmarks/datasets/registry.json`)**: Tracks benchmarks for 300k+ doc datasets.
- **Runners**: Automated scripts to verify MRR, latency, and scalability.
- **Notebooks**: Reproducible experimental files for academic papers.

## Getting Started
```python
from irlib import SearchPipeline, BM25Retriever, DenseRetriever

# 1. Initialize Hybrid Search
pipeline = SearchPipeline(
    lexical_retriever=BM25Retriever(), 
    dense_retriever=DenseRetriever()
)

# 2. Index (Streaming approach)
pipeline.index(["Legal judgment content..."])

# 3. Retrieve
results = pipeline.search("POCSO bail conditions")
```

## Contributing
- **Benchmarks**: See `notebooks/README.md`.
- **Traceability**: All routing decisions are traceable via `traceability_id`.
