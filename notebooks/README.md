# irlib Research Suite

This directory contains Jupyter notebooks designed for scientific reproducibility and validation of the `irlib` Information Retrieval framework.

## Notebook Overview

### 01_Retrieval_Baselines.ipynb
- **Purpose**: Establishes the baseline performance of `irlib` compared to standard lexical and dense baselines.
- **Focus**: Demonstrates why the hybrid pipeline is mathematically and empirically superior.

### 02_Learning_Loop_Moat.ipynb
- **Purpose**: Demonstrates the "Learning Moat" concept.
- **Focus**: Visualizes the performance delta (MRR improvement) achieved by the automated feedback loop.

### 03_Attribution_Transparency.ipynb
- **Purpose**: Explains retrieval decisions using LIME and Bayesian traceability.
- **Focus**: Provides the scientific audit trail for retrieval reliability in high-stakes domains like the Indian Judiciary.

### 04_Scalability_Benchmarking.ipynb
- **Purpose**: Reproduces the throughput metrics on 300,000+ document corpora.
- **Focus**: Memory profiling, indexing throughput, and search latency.

## Instructions
1. Ensure `irlib` is installed in your Python environment (`pip install -e .`).
2. Run notebooks sequentially to reproduce the research results documented in the project report.
3. For large-scale benchmarking, ensure the `datasets/` registry contains the required corpus snapshots.
