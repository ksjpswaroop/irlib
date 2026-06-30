"""Reranking and learning-to-rank algorithms.

Rerankers operate after a first-stage retriever has produced candidates. They
can use lexical overlap, cross-encoders, or learned ranking models.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from irlib.core import tokenize, top_k as rank_top_k


def _feature_vector(features: Mapping[str, float] | Sequence[float]) -> list[float]:
    if isinstance(features, Mapping):
        return [float(value) for _key, value in sorted(features.items())]
    return [float(value) for value in features]


def _dot(weights: Sequence[float], vector: Sequence[float]) -> float:
    return sum(w * x for w, x in zip(weights, vector))


class Reranker:
    """Cross-encoder-style candidate reranker.

    Use after BM25, dense, or hybrid retrieval to improve precision.

    Pseudocode:
        candidates = first_stage(query)
        for passage in candidates:
            score = model(query, passage)
        return candidates sorted by score

    Limitation: cross-encoders are too slow for full-corpus retrieval.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model: Any | None = None,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.scorer = scorer

    def _get_model(self) -> Any | None:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None
        return self.model

    def _fallback_score(self, query: str, passage: str) -> float:
        q_terms = set(tokenize(query))
        p_terms = set(tokenize(passage))
        if not q_terms or not p_terms:
            return 0.0
        return len(q_terms & p_terms) / len(q_terms | p_terms)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if self.scorer:
            return [float(self.scorer(query, passage)) for passage in passages]
        model = self._get_model()
        if model is not None:
            pairs = [(query, passage) for passage in passages]
            return [float(value) for value in model.predict(pairs)]
        return [self._fallback_score(query, passage) for passage in passages]

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        doc_ids: Sequence[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        ids = list(doc_ids) if doc_ids is not None else list(range(len(passages)))
        scores = dict(zip(ids, self.score(query, passages)))
        return rank_top_k(scores, top_k)


class PointwiseLTRRanker:
    """Pointwise learning-to-rank with a compact linear fallback.

    Use when each query-document pair has an absolute relevance label.

    Pseudocode:
        train model to predict relevance from features(q, d)
        rank candidates by predicted relevance

    Limitation: pointwise training ignores relative ordering unless labels
    encode that ordering.
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 100) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, features: Sequence[Mapping[str, float] | Sequence[float]], labels: Sequence[float]) -> "PointwiseLTRRanker":
        vectors = [_feature_vector(row) for row in features]
        width = max((len(vector) for vector in vectors), default=0)
        self.weights = [0.0] * width
        self.bias = 0.0
        for _ in range(self.epochs):
            for vector, label in zip(vectors, labels):
                padded = vector + [0.0] * (width - len(vector))
                error = (_dot(self.weights, padded) + self.bias) - float(label)
                for i, value in enumerate(padded):
                    self.weights[i] -= self.learning_rate * error * value
                self.bias -= self.learning_rate * error
        return self

    def predict(self, features: Sequence[Mapping[str, float] | Sequence[float]]) -> list[float]:
        width = len(self.weights)
        predictions = []
        for row in features:
            vector = _feature_vector(row)
            padded = vector + [0.0] * (width - len(vector))
            predictions.append(_dot(self.weights, padded) + self.bias)
        return predictions

    def rank(
        self,
        features: Sequence[Mapping[str, float] | Sequence[float]],
        doc_ids: Sequence[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        ids = list(doc_ids) if doc_ids is not None else list(range(len(features)))
        return rank_top_k(dict(zip(ids, self.predict(features))), top_k)


class PairwiseLTRRanker:
    """Pairwise learning-to-rank with a perceptron-style preference model.

    Use when training data says one document should rank above another.

    Pseudocode:
        for each preferred pair (good, bad):
            if score(good) <= score(bad): weights += good - bad

    Limitation: pair generation can be expensive and biased.
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 20) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []

    def fit(self, pairs: Sequence[tuple[Sequence[float], Sequence[float]]]) -> "PairwiseLTRRanker":
        width = max((max(len(good), len(bad)) for good, bad in pairs), default=0)
        self.weights = [0.0] * width
        for _ in range(self.epochs):
            for good, bad in pairs:
                good_vec = list(good) + [0.0] * (width - len(good))
                bad_vec = list(bad) + [0.0] * (width - len(bad))
                if _dot(self.weights, good_vec) <= _dot(self.weights, bad_vec):
                    for i in range(width):
                        self.weights[i] += self.learning_rate * (good_vec[i] - bad_vec[i])
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        width = len(self.weights)
        return [_dot(self.weights, list(row) + [0.0] * (width - len(row))) for row in features]

    def rank(self, features: Sequence[Sequence[float]], doc_ids: Sequence[int] | None = None, top_k: int = 10) -> list[tuple[int, float]]:
        ids = list(doc_ids) if doc_ids is not None else list(range(len(features)))
        return rank_top_k(dict(zip(ids, self.predict(features))), top_k)


class ListwiseLTRRanker(PointwiseLTRRanker):
    """Listwise learning-to-rank reference model.

    Use when labels are grouped by query result list.

    Pseudocode:
        flatten query lists into feature rows and list-relative labels
        train a scoring model
        rank each candidate list by score

    Limitation: this compact fallback does not implement a full neural listwise
    loss, but keeps the interface testable.
    """

    def fit_lists(
        self,
        feature_lists: Sequence[Sequence[Mapping[str, float] | Sequence[float]]],
        label_lists: Sequence[Sequence[float]],
    ) -> "ListwiseLTRRanker":
        features = [row for feature_list in feature_lists for row in feature_list]
        labels = [label for label_list in label_lists for label in label_list]
        return self.fit(features, labels)


class LambdaMARTRanker:
    """LambdaMART wrapper backed by LightGBM when available.

    Use for production-style ranking over tabular search features.

    Pseudocode:
        if lightgbm exists: train LGBMRanker with query group sizes
        else: use pointwise fallback
        rank candidates by model score

    Limitation: useful LambdaMART needs high-quality relevance labels and
    engineered features.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.model: Any | None = None
        self.fallback = PointwiseLTRRanker()

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[float],
        group: Sequence[int] | None = None,
    ) -> "LambdaMARTRanker":
        try:
            from lightgbm import LGBMRanker

            self.model = LGBMRanker(**self.params)
            self.model.fit(features, labels, group=group)
        except Exception:
            self.model = None
            self.fallback.fit(features, labels)
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if self.model is not None:
            return [float(value) for value in self.model.predict(features)]
        return self.fallback.predict(features)

    def rank(self, features: Sequence[Sequence[float]], doc_ids: Sequence[int] | None = None, top_k: int = 10) -> list[tuple[int, float]]:
        ids = list(doc_ids) if doc_ids is not None else list(range(len(features)))
        return rank_top_k(dict(zip(ids, self.predict(features))), top_k)


__all__ = [
    "Reranker",
    "PointwiseLTRRanker",
    "PairwiseLTRRanker",
    "ListwiseLTRRanker",
    "LambdaMARTRanker",
]

