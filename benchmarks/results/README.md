# Benchmark Results History

This file tracks the retrieval performance of `irlib` across standardized benchmark tiers.

## Tier: Smoke Tests
*   **Date:** 2026-06-30
*   **Description:** Small BEIR sets for frequent manual smoke checks.

| Dataset | BM25 NDCG@10 | Recall@100 |
|---|---:|---:|
| `scifact` | 0.6322 | 0.7800 |
| `nfcorpus` | 0.2839 | 0.1764 |
| `arguana` | 0.3719 | 0.9600 |

*Retrieval-only evaluation. Answer-level metrics are not included.*
