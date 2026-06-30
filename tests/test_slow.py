import os

import pytest

from irlib import DenseRetriever, HashingEncoder, HNSWRetriever, IVFRetriever, LambdaMARTRanker


pytestmark = pytest.mark.slow


DOCS = [
    "python retrieval search",
    "java enterprise search",
    "semantic vector retrieval",
    "graph ranking authority",
]


def test_hnsw_backend_smoke():
    pytest.importorskip("hnswlib")
    retriever = HNSWRetriever(encoder=HashingEncoder())
    retriever.index(DOCS)
    assert retriever._index is not None
    assert retriever.search("semantic retrieval", top_k=2)


def test_faiss_ivf_backend_smoke():
    pytest.importorskip("faiss")
    retriever = IVFRetriever(nlist=1, nprobe=1, encoder=HashingEncoder())
    retriever.index(DOCS)
    assert retriever._faiss_index is not None
    assert retriever.search("semantic retrieval", top_k=2)


def test_lightgbm_lambdamart_smoke():
    if os.environ.get("IRLIB_RUN_LIGHTGBM_TESTS") != "1":
        pytest.skip("Set IRLIB_RUN_LIGHTGBM_TESTS=1 to exercise the native LightGBM ranker.")
    pytest.importorskip("lightgbm")
    ranker = LambdaMARTRanker(n_estimators=5, min_data_in_leaf=1, verbose=-1)
    ranker.fit([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], [2.0, 0.0, 1.0], group=[3])
    assert ranker.rank([[1.0, 0.0], [0.0, 1.0]], top_k=1)


def test_real_sentence_transformer_model_download_smoke():
    if os.environ.get("IRLIB_RUN_MODEL_TESTS") != "1":
        pytest.skip("Set IRLIB_RUN_MODEL_TESTS=1 to download and test a real sentence-transformer model.")
    retriever = DenseRetriever()
    retriever.index(DOCS)
    assert retriever.search("semantic retrieval", top_k=2)
