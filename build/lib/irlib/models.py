"""Classical and probabilistic lexical retrieval models.

These algorithms use tokens, postings, counts, fields, and metadata rather than
large neural models. They are fast, deterministic, and useful as first-stage
retrievers or as transparent baselines.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter, defaultdict
from itertools import product
from typing import Any, Mapping, Sequence

from irlib.core import BaseRetriever, Document, sparse_cosine, tokenize, token_positions, top_k as rank_top_k


class InvertedIndexRetriever(BaseRetriever):
    """Retrieve documents through a term-to-document posting list.

    Use for fast lexical candidate generation.

    Pseudocode:
        index: for each document, add doc_id to postings[term]
        search: union postings for query terms, score by matched term count

    Limitation: it matches exact terms only unless paired with expansion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.postings: dict[str, set[int]] = defaultdict(set)

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.postings = defaultdict(set)
        for doc in self.documents:
            for term in set(tokenize(doc.text)):
                self.postings[term].add(doc.id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: Counter[int] = Counter()
        for term in set(tokenize(query)):
            for doc_id in self.postings.get(term, set()):
                scores[doc_id] += 1
        return top_k_scores(scores, top_k)


class BooleanRetriever(InvertedIndexRetriever):
    """Exact Boolean retrieval with AND, OR, NOT, and parentheses.

    Use for legal, compliance, and expert search where the query is a precise
    logical expression.

    Pseudocode:
        tokenize query operators and terms
        convert infix expression to reverse polish notation
        evaluate posting-list set operations
        score matched documents by matched query-term count

    Limitation: Boolean matching is brittle and does not infer semantics.
    """

    _QUERY_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[\w*?]+", re.IGNORECASE)
    _PRECEDENCE = {"OR": 1, "AND": 2, "NOT": 3}

    def _query_tokens(self, query: str) -> list[str]:
        raw = self._QUERY_RE.findall(query)
        tokens: list[str] = []
        previous_kind: str | None = None
        for token in raw:
            upper = token.upper()
            if upper in self._PRECEDENCE or token in {"(", ")"}:
                current = upper if upper in self._PRECEDENCE else token
            else:
                current = token.lower()
            current_kind = "op" if current in self._PRECEDENCE else current
            if previous_kind in {"term", ")"} and (current_kind == "term" or current == "(" or current == "NOT"):
                tokens.append("OR")
            tokens.append(current)
            if current in self._PRECEDENCE:
                previous_kind = "op"
            elif current == "(":
                previous_kind = "("
            elif current == ")":
                previous_kind = ")"
            else:
                previous_kind = "term"
        return tokens

    def _to_rpn(self, query: str) -> list[str]:
        output: list[str] = []
        ops: list[str] = []
        for token in self._query_tokens(query):
            if token == "(":
                ops.append(token)
            elif token == ")":
                while ops and ops[-1] != "(":
                    output.append(ops.pop())
                if ops and ops[-1] == "(":
                    ops.pop()
            elif token in self._PRECEDENCE:
                while ops and ops[-1] in self._PRECEDENCE and self._PRECEDENCE[ops[-1]] >= self._PRECEDENCE[token]:
                    output.append(ops.pop())
                ops.append(token)
            else:
                output.append(token)
        while ops:
            op = ops.pop()
            if op != "(":
                output.append(op)
        return output

    def _term_set(self, token: str) -> set[int]:
        if "*" in token or "?" in token:
            matched: set[int] = set()
            for term, postings in self.postings.items():
                if fnmatch.fnmatch(term, token):
                    matched.update(postings)
            return matched
        return set(self.postings.get(token, set()))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        universe = {doc.id for doc in self.documents}
        stack: list[set[int]] = []
        for token in self._to_rpn(query):
            if token == "NOT":
                operand = stack.pop() if stack else set()
                stack.append(universe - operand)
            elif token in {"AND", "OR"}:
                right = stack.pop() if stack else set()
                left = stack.pop() if stack else set()
                stack.append(left & right if token == "AND" else left | right)
            else:
                stack.append(self._term_set(token))
        matched = stack[-1] if stack else set()
        query_terms = [token for token in self._query_tokens(query) if token not in self._PRECEDENCE and token not in {"(", ")"}]
        scores = {
            doc.id: float(sum(1 for term in query_terms if doc.id in self._term_set(term)) or 1)
            for doc in self.documents
            if doc.id in matched
        }
        return rank_top_k(scores, top_k)


class TermFrequencyRetriever(BaseRetriever):
    """Rank by raw query-term frequency in each document.

    Use as a tiny baseline or for tests.

    Pseudocode:
        index: count terms in each document
        search: score[d] = sum(tf(term, d) for term in query)

    Limitation: long documents and common terms are over-rewarded.
    """

    def __init__(self) -> None:
        super().__init__()
        self.doc_tfs: dict[int, Counter[str]] = {}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.doc_tfs = {doc.id: Counter(tokenize(doc.text)) for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        scores = {
            doc_id: float(sum(tf.get(term, 0) for term in q_terms))
            for doc_id, tf in self.doc_tfs.items()
        }
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class TFIDFRetriever(BaseRetriever):
    """Vector-space TF-IDF retrieval with sparse cosine similarity.

    Use as an interpretable lexical baseline.

    Pseudocode:
        idf[term] = log((N + 1) / (df + 1)) + 1
        vector[d][term] = tf(term, d) * idf[term]
        score = cosine(vector(query), vector(document))

    Limitation: exact vocabulary matching is still required.
    """

    def __init__(self) -> None:
        super().__init__()
        self.idf: dict[str, float] = {}
        self.doc_vectors: dict[int, dict[str, float]] = {}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        dfs: Counter[str] = Counter()
        doc_tfs: dict[int, Counter[str]] = {}
        for doc in self.documents:
            tf = Counter(tokenize(doc.text))
            doc_tfs[doc.id] = tf
            dfs.update(tf.keys())
        n_docs = max(1, len(self.documents))
        self.idf = {term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in dfs.items()}
        self.doc_vectors = {
            doc_id: {term: count * self.idf.get(term, 0.0) for term, count in tf.items()}
            for doc_id, tf in doc_tfs.items()
        }

    def _query_vector(self, query: str) -> dict[str, float]:
        tf = Counter(tokenize(query))
        return {term: count * self.idf.get(term, 0.0) for term, count in tf.items()}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_vec = self._query_vector(query)
        scores = {doc_id: sparse_cosine(q_vec, vec) for doc_id, vec in self.doc_vectors.items()}
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class BM25Retriever(BaseRetriever):
    """Okapi BM25 probabilistic ranking.

    Use as the default sparse first-stage retriever for most text search and RAG
    systems.

    Pseudocode:
        index: store document term counts, lengths, and document frequencies
        search: sum idf(term) * saturated_tf(term, document)

    Limitation: BM25 cannot match paraphrases without expansion or hybrid search.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, tf_exponent: float = 1.0) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b
        self.tf_exponent = tf_exponent
        self.doc_tfs: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.df: Counter[str] = Counter()
        self.postings: dict[str, set[int]] = defaultdict(set)
        self.avgdl = 0.0

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.doc_tfs = {}
        self.doc_lengths = {}
        self.df = Counter()
        self.postings = defaultdict(set)
        for doc in self.documents:
            tf = Counter(tokenize(doc.text))
            self.doc_tfs[doc.id] = tf
            self.doc_lengths[doc.id] = sum(tf.values())
            for term in tf:
                self.df[term] += 1
                self.postings[term].add(doc.id)
        self.avgdl = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.df.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs else 0.0

    def _tf_weight(self, tf: float, doc_len: int) -> float:
        if self.tf_exponent != 1.0:
            tf = math.log1p(tf) ** self.tf_exponent
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
        return (tf * (self.k1 + 1)) / denominator if denominator else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self._idf(term)
            for doc_id in self.postings.get(term, set()):
                tf = self.doc_tfs[doc_id][term]
                scores[doc_id] += idf * self._tf_weight(tf, self.doc_lengths[doc_id])
        return rank_top_k(scores, top_k)


class BM25SublinearRetriever(BM25Retriever):
    """BM25 variant using sublinear term-frequency scaling."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__(k1=k1, b=b, tf_exponent=1.5)


class BM25PlusRetriever(BM25Retriever):
    """BM25+ adds a lower-bound delta to reduce long-document under-scoring."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 1.0) -> None:
        super().__init__(k1=k1, b=b)
        self.delta = delta

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self._idf(term)
            for doc_id in self.postings.get(term, set()):
                tf = self.doc_tfs[doc_id][term]
                scores[doc_id] += idf * (self._tf_weight(tf, self.doc_lengths[doc_id]) + self.delta)
        return rank_top_k(scores, top_k)


class BM25LRetriever(BM25Retriever):
    """BM25L uses adjusted term frequency to soften length normalization."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 0.5) -> None:
        super().__init__(k1=k1, b=b)
        self.delta = delta

    def _tf_weight(self, tf: float, doc_len: int) -> float:
        ctd = tf / (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
        numerator = (self.k1 + 1) * (ctd + self.delta)
        denominator = self.k1 + ctd + self.delta
        return numerator / denominator if denominator else 0.0


class BM25FRetriever(BaseRetriever):
    """Field-aware BM25 for title/body/tag style documents.

    Use when important fields should receive different weights.

    Pseudocode:
        for each field, count terms and field lengths
        combine field term frequencies with configured weights
        score with BM25 saturation over the combined field frequency

    Limitation: field weights need tuning for each collection.
    """

    def __init__(self, field_weights: Mapping[str, float] | None = None, k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__()
        self.field_weights = dict(field_weights or {"title": 2.0, "body": 1.0, "text": 1.0})
        self.k1 = k1
        self.b = b
        self.field_tfs: dict[int, dict[str, Counter[str]]] = {}
        self.field_lengths: dict[int, dict[str, int]] = {}
        self.avg_field_lengths: dict[str, float] = {}
        self.df: Counter[str] = Counter()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.field_tfs = {}
        self.field_lengths = {}
        field_length_totals: Counter[str] = Counter()
        field_counts: Counter[str] = Counter()
        self.df = Counter()
        for doc in self.documents:
            fields = dict(doc.fields) if doc.fields else {"text": doc.text}
            if "text" not in fields:
                fields["text"] = doc.text
            per_field: dict[str, Counter[str]] = {}
            seen_terms: set[str] = set()
            self.field_lengths[doc.id] = {}
            for field_name, field_text in fields.items():
                tf = Counter(tokenize(field_text))
                per_field[field_name] = tf
                length = sum(tf.values())
                self.field_lengths[doc.id][field_name] = length
                field_length_totals[field_name] += length
                field_counts[field_name] += 1
                seen_terms.update(tf.keys())
            self.field_tfs[doc.id] = per_field
            self.df.update(seen_terms)
        self.avg_field_lengths = {
            field: field_length_totals[field] / max(1, field_counts[field])
            for field in field_length_totals
        }

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for doc in self.documents:
            for term in tokenize(query):
                weighted_tf = 0.0
                for field_name, tf in self.field_tfs[doc.id].items():
                    weight = self.field_weights.get(field_name, 1.0)
                    length = self.field_lengths[doc.id].get(field_name, 0)
                    avg_len = self.avg_field_lengths.get(field_name, 1.0) or 1.0
                    normalized_tf = tf.get(term, 0) / (1 - self.b + self.b * length / avg_len)
                    weighted_tf += weight * normalized_tf
                denominator = self.k1 + weighted_tf
                if weighted_tf and denominator:
                    scores[doc.id] += self._idf(term) * ((self.k1 + 1) * weighted_tf / denominator)
        return rank_top_k(scores, top_k)


class LanguageModelRetriever(BaseRetriever):
    """Query likelihood retrieval with Dirichlet smoothing.

    Use as a probabilistic alternative to BM25.

    Pseudocode:
        estimate collection probability P(term | collection)
        score document by sum log((tf + mu * p_collection) / (doc_len + mu))

    Limitation: smoothing choices can dominate behavior on small collections.
    """

    def __init__(self, mu: float = 1500.0) -> None:
        super().__init__()
        self.mu = mu
        self.doc_tfs: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.collection_tf: Counter[str] = Counter()
        self.collection_len = 0

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.doc_tfs = {}
        self.doc_lengths = {}
        self.collection_tf = Counter()
        self.collection_len = 0
        for doc in self.documents:
            tf = Counter(tokenize(doc.text))
            self.doc_tfs[doc.id] = tf
            length = sum(tf.values())
            self.doc_lengths[doc.id] = length
            self.collection_tf.update(tf)
            self.collection_len += length

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        scores: dict[int, float] = {}
        for doc in self.documents:
            score = 0.0
            doc_len = self.doc_lengths.get(doc.id, 0)
            for term in q_terms:
                p_collection = self.collection_tf.get(term, 0) / (self.collection_len or 1)
                probability = (self.doc_tfs[doc.id].get(term, 0) + self.mu * p_collection) / (doc_len + self.mu)
                if probability > 0:
                    score += math.log(probability)
            if score != 0:
                scores[doc.id] = score
        return rank_top_k(scores, top_k)


class DFRRetriever(BM25Retriever):
    """Simplified Divergence-from-Randomness lexical retrieval.

    Use for classical IR experimentation.

    Pseudocode:
        expected_tf = collection_frequency(term) * doc_len / collection_len
        score += observed_tf * log((observed_tf + 1) / (expected_tf + 1))

    Limitation: this is a compact reference variant, not every DFR model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.collection_tf: Counter[str] = Counter()
        self.collection_len = 0

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        super().index(documents)
        self.collection_tf = Counter()
        for tf in self.doc_tfs.values():
            self.collection_tf.update(tf)
        self.collection_len = sum(self.collection_tf.values())

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            for doc_id in self.postings.get(term, set()):
                observed = self.doc_tfs[doc_id].get(term, 0)
                expected = self.collection_tf.get(term, 0) * self.doc_lengths[doc_id] / (self.collection_len or 1)
                scores[doc_id] += observed * math.log(1 + observed / (expected + 1e-9))
        return rank_top_k(scores, top_k)


class PhraseRetriever(BaseRetriever):
    """Retrieve documents containing exact ordered token phrases.

    Use for quote search, code search, and known-item search.

    Pseudocode:
        index token positions per document
        for each phrase, keep documents where positions are consecutive

    Limitation: exact phrase matching misses reordered or paraphrased text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.positions: dict[int, dict[str, list[int]]] = {}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        self.positions = {doc.id: token_positions(doc.text) for doc in self.documents}

    def _phrase_count(self, terms: list[str], doc_positions: dict[str, list[int]]) -> int:
        if not terms or any(term not in doc_positions for term in terms):
            return 0
        count = 0
        first_positions = doc_positions[terms[0]]
        following = [set(doc_positions[term]) for term in terms[1:]]
        for start in first_positions:
            if all(start + offset + 1 in positions for offset, positions in enumerate(following)):
                count += 1
        return count

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        quoted = re.findall(r'"([^"]+)"', query)
        phrases = quoted or [query]
        phrase_terms = [tokenize(phrase) for phrase in phrases]
        scores: dict[int, float] = {}
        for doc in self.documents:
            score = sum(self._phrase_count(terms, self.positions[doc.id]) for terms in phrase_terms)
            if score:
                scores[doc.id] = float(score)
        return rank_top_k(scores, top_k)


class ProximityRetriever(PhraseRetriever):
    """Reward documents where query terms occur close together.

    Use when word co-occurrence is important but exact phrase order is too
    strict.

    Pseudocode:
        find all positions of query terms
        compute the smallest token window covering all terms
        score = query_terms / (1 + window_size)

    Limitation: positional indexes are larger than plain postings.
    """

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        scores: dict[int, float] = {}
        for doc in self.documents:
            pos = self.positions[doc.id]
            if not q_terms or any(term not in pos for term in q_terms):
                continue
            best_window: int | None = None
            for combo in product(*(pos[term] for term in q_terms)):
                window = max(combo) - min(combo) + 1
                best_window = window if best_window is None else min(best_window, window)
            if best_window:
                scores[doc.id] = len(q_terms) / (1 + best_window)
        return rank_top_k(scores, top_k)


class FuzzyRetriever(InvertedIndexRetriever):
    """Retrieve terms close by edit distance or fuzzy ratio.

    Use for typo tolerance, OCR text, and names.

    Pseudocode:
        for each query term, compare with vocabulary
        keep vocabulary terms above threshold
        score documents by fuzzy similarity

    Limitation: fuzzy expansion can introduce false positives.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        self.threshold = threshold

    def _similarity(self, a: str, b: str) -> float:
        try:
            from rapidfuzz import fuzz

            return fuzz.ratio(a, b) / 100.0
        except Exception:
            from difflib import SequenceMatcher

            return SequenceMatcher(None, a, b).ratio()

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        vocabulary = list(self.postings.keys())
        for q_term in tokenize(query):
            for term in vocabulary:
                similarity = self._similarity(q_term, term)
                if similarity >= self.threshold:
                    for doc_id in self.postings[term]:
                        scores[doc_id] += similarity
        return rank_top_k(scores, top_k)


class CharNGramRetriever(BaseRetriever):
    """Character n-gram retrieval for partial and noisy text.

    Use for autocomplete, names, OCR, and languages where tokenization is hard.

    Pseudocode:
        index character n-grams for each document
        score by cosine overlap between query n-grams and document n-grams

    Limitation: short n-grams can be noisy and indexes can be larger.
    """

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = n
        self.doc_vectors: dict[int, Counter[str]] = {}

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        from irlib.core import char_ngrams

        self._set_documents(documents)
        self.doc_vectors = {doc.id: Counter(char_ngrams(doc.text, self.n)) for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        from irlib.core import char_ngrams

        q_vec = Counter(char_ngrams(query, self.n))
        scores = {doc_id: sparse_cosine(q_vec, vec) for doc_id, vec in self.doc_vectors.items()}
        return rank_top_k({doc_id: score for doc_id, score in scores.items() if score > 0}, top_k)


class WildcardPrefixRetriever(InvertedIndexRetriever):
    """Wildcard and prefix retrieval over the indexed vocabulary.

    Use for autocomplete, catalog codes, and symbol lookup.

    Pseudocode:
        expand wildcard or prefix query terms against the vocabulary
        union posting lists for expanded terms

    Limitation: broad wildcards can expand to many terms.
    """

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for raw_term in tokenize(query):
            pattern = raw_term if "*" in raw_term or "?" in raw_term else f"{raw_term}*"
            for term, postings in self.postings.items():
                if fnmatch.fnmatch(term, pattern):
                    for doc_id in postings:
                        scores[doc_id] += 1.0
        return rank_top_k(scores, top_k)


class FacetedRetriever(BaseRetriever):
    """Apply metadata filters after a base retrieval pass.

    Use for ecommerce, enterprise search, logs, and structured collections.

    Pseudocode:
        base_results = base.search(query)
        keep results where document metadata matches all filters

    Limitation: quality depends on complete and reliable metadata.
    """

    def __init__(self, base_retriever: BaseRetriever | None = None, filters: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.base_retriever = base_retriever or BM25Retriever()
        self.filters = dict(filters or {})

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.base_retriever.index(documents)
        self.documents = self.base_retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def _matches(self, doc: Document, filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = doc.metadata.get(key)
            if isinstance(expected, (set, list, tuple)):
                if actual not in expected:
                    return False
            elif callable(expected):
                if not expected(actual):
                    return False
            elif actual != expected:
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        active_filters = dict(self.filters)
        if filters:
            active_filters.update(filters)
        candidates = self.base_retriever.search(query, top_k=max(top_k * 10, top_k))
        scores = {
            doc_id: score
            for doc_id, score in candidates
            if self._matches(self.get_document(doc_id), active_filters)
        }
        return top_k_scores(scores, top_k)


def top_k_scores(scores: Mapping[int, float] | Counter[int], k: int) -> list[tuple[int, float]]:
    return rank_top_k({doc_id: float(score) for doc_id, score in scores.items() if score > 0}, k)


__all__ = [
    "InvertedIndexRetriever",
    "BooleanRetriever",
    "TermFrequencyRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "BM25SublinearRetriever",
    "BM25PlusRetriever",
    "BM25LRetriever",
    "BM25FRetriever",
    "LanguageModelRetriever",
    "DFRRetriever",
    "PhraseRetriever",
    "ProximityRetriever",
    "FuzzyRetriever",
    "CharNGramRetriever",
    "WildcardPrefixRetriever",
    "FacetedRetriever",
]
