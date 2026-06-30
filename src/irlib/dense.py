"""Dense, semantic, and approximate nearest-neighbor retrieval.

This module exposes neural-style retrievers while keeping deterministic local
fallbacks for tests and offline experimentation. Heavy backends such as
sentence-transformers, hnswlib, FAISS, and scikit-learn are loaded lazily.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from irlib.core import BaseRetriever, Document, cosine_similarity, sparse_cosine, tokenize, top_k as rank_top_k
from irlib.models import TFIDFRetriever


class HashingEncoder:
    """Small deterministic text encoder used when no embedding model is given.

    Pseudocode:
        vector = zeros(dim)
        for token in text: vector[hash(token) % dim] += 1
        normalize vector

    Limitation: this is not semantic; it is a stable fallback for tests.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: str | Sequence[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        vectors = [self._encode_one(text) for text in batch]
        return vectors[0] if single else vectors

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for term in tokenize(text):
            digest = hashlib.md5(term.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class DenseRetriever(BaseRetriever):
    """Bi-encoder dense retrieval.

    Use for semantic search, RAG, recommendations, and multilingual retrieval.

    Pseudocode:
        encode all documents independently
        encode the query independently
        return documents with highest vector cosine similarity

    Limitation: independent encoders can miss exact constraints and fine-grained
    query-document interactions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        encoder: Any | None = None,
        dim: int = 128,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = encoder
        self.dim = dim
        self.doc_vectors: dict[int, list[float]] = {}

    def _get_encoder(self) -> Any:
        if self.encoder is not None:
            return self.encoder
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)
        except Exception:
            self.encoder = HashingEncoder(self.dim)
        return self.encoder

    def _encode(self, texts: str | Sequence[str]) -> list[float] | list[list[float]]:
        encoder = self._get_encoder()
        if hasattr(encoder, "encode"):
            encoded = encoder.encode(texts)
        elif callable(encoder):
            encoded = encoder(texts)
        else:
            raise TypeError("encoder must be callable or expose encode()")
        if isinstance(texts, str):
            return [float(value) for value in list(encoded)]
        return [[float(value) for value in list(row)] for row in list(encoded)]

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        texts = [doc.text for doc in self.documents]
        vectors = self._encode(texts)
        self.doc_vectors = {doc.id: vector for doc, vector in zip(self.documents, vectors)}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_vec = self._encode(query)
        scores = {doc_id: cosine_similarity(q_vec, vector) for doc_id, vector in self.doc_vectors.items()}
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class ExactKNNRetriever(DenseRetriever):
    """Exact vector k-nearest-neighbor retrieval.

    Pseudocode:
        encode query
        compare query with every document vector
        return exact top-k

    Limitation: full scans are slow for large collections.
    """


class LSHRetriever(DenseRetriever):
    """Locality Sensitive Hashing for approximate vector retrieval.

    Use for simple approximate nearest-neighbor experiments.

    Pseudocode:
        create random hyperplanes
        hash each vector by sign of dot(vector, plane)
        search only vectors in the same bucket, fallback to all if empty

    Limitation: recall depends heavily on the number of planes and buckets.
    """

    def __init__(self, num_planes: int = 12, seed: int = 13, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_planes = num_planes
        self.seed = seed
        self.planes: list[list[float]] = []
        self.buckets: dict[str, list[int]] = defaultdict(list)

    def _make_planes(self, dim: int) -> None:
        rng = random.Random(self.seed)
        self.planes = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(self.num_planes)]

    def _hash(self, vector: Sequence[float]) -> str:
        bits = []
        for plane in self.planes:
            dot = sum(a * b for a, b in zip(vector, plane))
            bits.append("1" if dot >= 0 else "0")
        return "".join(bits)

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        super().index(documents)
        dim = len(next(iter(self.doc_vectors.values()), []))
        self._make_planes(dim)
        self.buckets = defaultdict(list)
        for doc_id, vector in self.doc_vectors.items():
            self.buckets[self._hash(vector)].append(doc_id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_vec = self._encode(query)
        bucket = self.buckets.get(self._hash(q_vec), [])
        candidate_ids = bucket or list(self.doc_vectors)
        scores = {doc_id: cosine_similarity(q_vec, self.doc_vectors[doc_id]) for doc_id in candidate_ids}
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class HNSWRetriever(DenseRetriever):
    """HNSW-backed approximate nearest-neighbor retrieval with exact fallback.

    Use for production-like vector search when `hnswlib` is available.

    Pseudocode:
        if hnswlib exists: add document vectors to HNSW graph
        search graph for nearest labels
        otherwise perform exact dense search

    Limitation: HNSW uses more memory than compressed indexes.
    """

    def __init__(self, ef: int = 50, m: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ef = ef
        self.m = m
        self._index: Any | None = None

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        super().index(documents)
        try:
            import hnswlib
            import numpy as np

            ids = list(self.doc_vectors)
            vectors = np.array([self.doc_vectors[doc_id] for doc_id in ids], dtype="float32")
            index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
            index.init_index(max_elements=len(ids), ef_construction=max(self.ef, 100), M=self.m)
            index.add_items(vectors, ids)
            index.set_ef(self.ef)
            self._index = index
        except Exception:
            self._index = None

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._index is None:
            return super().search(query, top_k)
        import numpy as np

        q_vec = np.array([self._encode(query)], dtype="float32")
        labels, distances = self._index.knn_query(q_vec, k=min(top_k, len(self.doc_vectors)))
        return [(int(doc_id), float(1.0 - distance)) for doc_id, distance in zip(labels[0], distances[0])]


class IVFRetriever(DenseRetriever):
    """Inverted-file vector retrieval using FAISS when available.

    Use for larger vector collections where only a few clusters should be
    searched per query.

    Pseudocode:
        train centroids over document vectors
        assign vectors to inverted lists
        search the closest lists, then rerank their members

    Limitation: approximate recall depends on `nlist` and `nprobe`.
    """

    def __init__(self, nlist: int = 8, nprobe: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.nlist = nlist
        self.nprobe = nprobe
        self._faiss_index: Any | None = None
        self._ids: list[int] = []

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        super().index(documents)
        self._ids = list(self.doc_vectors)
        try:
            import faiss
            import numpy as np

            vectors = np.array([self.doc_vectors[doc_id] for doc_id in self._ids], dtype="float32")
            dim = vectors.shape[1]
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.nlist, max(1, len(self._ids)))
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.add(vectors)
            index.nprobe = min(self.nprobe, nlist)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._faiss_index is None:
            return super().search(query, top_k)
        import numpy as np

        q_vec = np.array([self._encode(query)], dtype="float32")
        scores, positions = self._faiss_index.search(q_vec, min(top_k, len(self._ids)))
        results = []
        for position, score in zip(positions[0], scores[0]):
            if position >= 0:
                results.append((self._ids[int(position)], float(score)))
        return results


class ProductQuantizationRetriever(DenseRetriever):
    """Compressed vector retrieval using simple scalar/product quantization.

    Use for memory-constrained vector experiments.

    Pseudocode:
        split vectors into subvectors
        quantize values into compact codes
        approximate similarity using reconstructed vectors

    Limitation: this compact reference favors clarity over FAISS-grade PQ.
    """

    def __init__(self, levels: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.levels = levels
        self.codes: dict[int, list[int]] = {}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        super().index(documents)
        self.codes = {doc_id: self._quantize(vector) for doc_id, vector in self.doc_vectors.items()}

    def _quantize(self, vector: Sequence[float]) -> list[int]:
        return [max(0, min(self.levels - 1, int(round((value + 1.0) * (self.levels - 1) / 2.0)))) for value in vector]

    def _dequantize(self, code: Sequence[int]) -> list[float]:
        if self.levels <= 1:
            return [0.0 for _ in code]
        return [(value * 2.0 / (self.levels - 1)) - 1.0 for value in code]

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_vec = self._encode(query)
        scores = {
            doc_id: cosine_similarity(q_vec, self._dequantize(code))
            for doc_id, code in self.codes.items()
        }
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class LSIRetriever(BaseRetriever):
    """Latent Semantic Indexing using TF-IDF plus truncated SVD.

    Use for small semantic baseline experiments.

    Pseudocode:
        build TF-IDF matrix
        project documents and query into low-rank SVD space
        score by cosine similarity

    Limitation: updates require recomputing the factorization.
    """

    def __init__(self, n_components: int = 100) -> None:
        super().__init__()
        self.n_components = n_components
        self._vectorizer: Any | None = None
        self._svd: Any | None = None
        self._doc_matrix: Any | None = None
        self._fallback = TFIDFRetriever()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer()
            matrix = self._vectorizer.fit_transform([doc.text for doc in self.documents])
            components = min(self.n_components, max(1, min(matrix.shape) - 1))
            self._svd = TruncatedSVD(n_components=components)
            self._doc_matrix = self._svd.fit_transform(matrix)
        except Exception:
            self._fallback.index(self.documents)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._vectorizer is None or self._svd is None or self._doc_matrix is None:
            return self._fallback.search(query, top_k)
        q_vec = self._svd.transform(self._vectorizer.transform([query]))[0]
        scores = {
            doc.id: cosine_similarity(q_vec, self._doc_matrix[i])
            for i, doc in enumerate(self.documents)
        }
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class TopicModelRetriever(BaseRetriever):
    """LDA topic-distribution retrieval.

    Use for thematic browsing and exploratory search.

    Pseudocode:
        fit a topic model over documents
        infer query topic distribution
        rank documents by topic-distribution similarity

    Limitation: topic models are usually weaker for precise ad hoc ranking.
    """

    def __init__(self, n_topics: int = 10, random_state: int = 13) -> None:
        super().__init__()
        self.n_topics = n_topics
        self.random_state = random_state
        self._vectorizer: Any | None = None
        self._lda: Any | None = None
        self._doc_topics: Any | None = None
        self._fallback = TFIDFRetriever()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer

            self._vectorizer = CountVectorizer()
            matrix = self._vectorizer.fit_transform([doc.text for doc in self.documents])
            topics = min(self.n_topics, max(1, len(self.documents)))
            self._lda = LatentDirichletAllocation(n_components=topics, random_state=self.random_state)
            self._doc_topics = self._lda.fit_transform(matrix)
        except Exception:
            self._fallback.index(self.documents)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._vectorizer is None or self._lda is None or self._doc_topics is None:
            return self._fallback.search(query, top_k)
        q_topic = self._lda.transform(self._vectorizer.transform([query]))[0]
        scores = {
            doc.id: cosine_similarity(q_topic, self._doc_topics[i])
            for i, doc in enumerate(self.documents)
        }
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class SparseNeuralRetriever(BaseRetriever):
    """Sparse neural retrieval interface with lexical fallback.

    Use for SPLADE/uniCOIL-style encoders that emit vocabulary weights while
    still using inverted-index-like scoring.

    Pseudocode:
        weights = sparse_encoder(text)
        index nonzero weights per document
        score = sparse dot/cosine(query_weights, doc_weights)

    Limitation: real quality depends on a trained sparse encoder.
    """

    def __init__(self, encoder: Callable[[str], Mapping[str, float]] | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.doc_vectors: dict[int, dict[str, float]] = {}

    def _encode(self, text: str) -> dict[str, float]:
        if self.encoder:
            return {str(term): float(value) for term, value in self.encoder(text).items()}
        tf = Counter(tokenize(text))
        return {term: float(count) for term, count in tf.items()}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.doc_vectors = {doc.id: self._encode(doc.text) for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_vec = self._encode(query)
        scores = {doc_id: sparse_cosine(q_vec, vector) for doc_id, vector in self.doc_vectors.items()}
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


__all__ = [
    "HashingEncoder",
    "DenseRetriever",
    "ExactKNNRetriever",
    "LSHRetriever",
    "HNSWRetriever",
    "IVFRetriever",
    "ProductQuantizationRetriever",
    "LSIRetriever",
    "TopicModelRetriever",
    "SparseNeuralRetriever",
]

