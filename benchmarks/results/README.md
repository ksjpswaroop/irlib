# RAG Retrieval Benchmark Results

- Dataset: `scifact` (beir:test)
- Queries: `100`
- Documents: `5183`
- Created at: `2026-06-29T23:36:44.076256+00:00`
- JSON: `scifact_sample.json`

| Retriever | NDCG@10 | Recall@10 | MRR@10 | MAP@10 | Mean latency ms |
|---|---:|---:|---:|---:|---:|
| `bm25` | 0.6247 | 0.7320 | 0.5989 | 0.5849 | 8.98 |
| `tfidf` | 0.5369 | 0.6357 | 0.5189 | 0.4977 | 23.18 |
| `hybrid-hash` | 0.4390 | 0.6328 | 0.3856 | 0.3751 | 52.27 |

This benchmark evaluates retrieval/context selection only. It does not evaluate generated answer quality.
