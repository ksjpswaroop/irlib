"""Query expansion, diversification, graph, and retrieval-pipeline utilities."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from typing import Any, Callable, Mapping, Sequence

from irlib.core import BaseRetriever, Document, chunk_documents, cosine_similarity, sparse_cosine, tokenize, top_k as rank_top_k
from irlib.dense import HashingEncoder
from irlib.hybrid import reciprocal_rank_fusion
from irlib.models import BM25Retriever


class QueryExpander:
    """Base query expander.

    Pseudocode:
        terms = tokenize(query)
        extra_terms = expansion_strategy(terms)
        return query plus expansion terms

    Use subclasses when recall is more important than strict precision.
    """

    def expand(self, query: str) -> str:
        return query


class SynonymQueryExpander(QueryExpander):
    """Expand terms from a curated synonym dictionary.

    Use in domains with controlled vocabularies.

    Pseudocode:
        for each query term:
            append configured synonyms

    Limitation: broad synonyms can cause query drift.
    """

    def __init__(self, synonyms: Mapping[str, Sequence[str]] | None = None) -> None:
        self.synonyms = {key.lower(): [value.lower() for value in values] for key, values in (synonyms or {}).items()}

    def expand(self, query: str) -> str:
        extras: list[str] = []
        for term in tokenize(query):
            extras.extend(self.synonyms.get(term, []))
        return " ".join([query, *extras]).strip()


class RocchioFeedback:
    """Rocchio relevance feedback for vector queries.

    Pseudocode:
        q_new = alpha*q + beta*mean(relevant) - gamma*mean(non_relevant)

    Use after users or labels identify relevant and non-relevant documents.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update(
        self,
        query_vector: Sequence[float],
        relevant_vectors: Sequence[Sequence[float]] = (),
        non_relevant_vectors: Sequence[Sequence[float]] = (),
    ) -> list[float]:
        width = len(query_vector)

        def mean(vectors: Sequence[Sequence[float]]) -> list[float]:
            if not vectors:
                return [0.0] * width
            return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]

        relevant = mean(relevant_vectors)
        non_relevant = mean(non_relevant_vectors)
        return [
            self.alpha * query_vector[i] + self.beta * relevant[i] - self.gamma * non_relevant[i]
            for i in range(width)
        ]


class RM3Expander(QueryExpander):
    """Pseudo-relevance feedback query expansion.

    Pseudocode:
        run initial search
        assume top documents are relevant
        add most frequent terms from those documents

    Limitation: wrong top results can push the query off topic.
    """

    def __init__(self, retriever: BaseRetriever, feedback_docs: int = 3, expansion_terms: int = 5) -> None:
        self.retriever = retriever
        self.feedback_docs = feedback_docs
        self.expansion_terms = expansion_terms

    def expand(self, query: str) -> str:
        results = self.retriever.search(query, top_k=self.feedback_docs)
        counts: Counter[str] = Counter()
        for doc_id, _score in results:
            counts.update(tokenize(self.retriever.get_document(doc_id).text))
        original = set(tokenize(query))
        extras = [term for term, _count in counts.most_common() if term not in original][: self.expansion_terms]
        return " ".join([query, *extras]).strip()


class EmbeddingQueryExpander(QueryExpander):
    """Expand a query with nearest candidate terms in embedding space.

    Pseudocode:
        encode query and candidate terms
        append the nearest terms by cosine similarity

    Limitation: nearest embedding terms can be semantically broad.
    """

    def __init__(self, candidate_terms: Sequence[str], encoder: Any | None = None, top_terms: int = 5) -> None:
        self.candidate_terms = list(candidate_terms)
        self.encoder = encoder or HashingEncoder()
        self.top_terms = top_terms

    def _encode(self, text: str) -> list[float]:
        encoded = self.encoder.encode(text) if hasattr(self.encoder, "encode") else self.encoder(text)
        return [float(value) for value in encoded]

    def expand(self, query: str) -> str:
        q_vec = self._encode(query)
        scores = {
            i: cosine_similarity(q_vec, self._encode(term))
            for i, term in enumerate(self.candidate_terms)
        }
        extras = [self.candidate_terms[i] for i, _score in rank_top_k(scores, self.top_terms)]
        return " ".join([query, *extras]).strip()


class Doc2QueryExpander:
    """Append generated likely queries to a document before indexing.

    Pseudocode:
        generated_queries = generator(document_text)
        expanded_document = document_text + generated_queries

    Limitation: synthetic queries can add noise.
    """

    def __init__(self, generator: Callable[[str], Sequence[str]] | None = None, max_terms: int = 8) -> None:
        self.generator = generator
        self.max_terms = max_terms

    def expand_document(self, text: str) -> str:
        if self.generator:
            queries = list(self.generator(text))
        else:
            common = [term for term, _count in Counter(tokenize(text)).most_common(self.max_terms)]
            queries = [" ".join(common)]
        return "\n".join([text, *queries]).strip()


class MultiQueryRetriever(BaseRetriever):
    """Run several rewritten queries and fuse the results.

    Pseudocode:
        rewrites = rewrite(query)
        results = [retriever.search(q) for q in rewrites]
        return reciprocal-rank-fused results

    Use for RAG and ambiguous natural-language questions.
    """

    def __init__(self, retriever: BaseRetriever | None = None, rewriter: Callable[[str], Sequence[str]] | None = None) -> None:
        super().__init__()
        self.retriever = retriever or BM25Retriever()
        self.rewriter = rewriter or (lambda query: [query])

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.retriever.index(documents)
        self.documents = self.retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        queries = list(dict.fromkeys([query, *self.rewriter(query)]))
        lists = [self.retriever.search(rewrite, top_k=max(top_k * 3, top_k)) for rewrite in queries]
        return reciprocal_rank_fusion(lists, top_k=top_k)


class ConversationalQueryRewriter:
    """Rewrite follow-up questions into standalone queries.

    Pseudocode:
        take recent conversation turns
        concatenate stable context with the latest question

    Limitation: this deterministic fallback does not resolve all references.
    """

    def __init__(self, max_turns: int = 3) -> None:
        self.max_turns = max_turns

    def rewrite(self, history: Sequence[str], query: str) -> str:
        context = " ".join(history[-self.max_turns :])
        return " ".join([context, query]).strip()


class MMRDiversifier:
    """Maximal Marginal Relevance diversification.

    Pseudocode:
        repeatedly pick candidate maximizing
        lambda*relevance - (1-lambda)*similarity_to_selected

    Use to reduce redundant search results or RAG chunks.
    """

    def __init__(self, lambda_mult: float = 0.7) -> None:
        self.lambda_mult = lambda_mult

    def select(
        self,
        candidates: Sequence[int],
        relevance_scores: Mapping[int, float],
        similarity: Callable[[int, int], float] | None = None,
        top_k: int = 10,
    ) -> list[int]:
        similarity = similarity or (lambda _a, _b: 0.0)
        selected: list[int] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda doc_id: self.lambda_mult * relevance_scores.get(doc_id, 0.0)
                - (1 - self.lambda_mult) * max((similarity(doc_id, chosen) for chosen in selected), default=0.0),
            )
            selected.append(best)
            remaining.remove(best)
        return selected


class XQuADDiversifier:
    """Intent-aware diversification.

    Pseudocode:
        infer or provide query intents
        reward documents that cover uncovered intents

    Limitation: useful behavior requires meaningful intent-document coverage.
    """

    def __init__(self, lambda_mult: float = 0.7) -> None:
        self.lambda_mult = lambda_mult

    def select(
        self,
        candidates: Sequence[int],
        relevance_scores: Mapping[int, float],
        intent_scores: Mapping[str, Mapping[int, float]],
        top_k: int = 10,
    ) -> list[int]:
        selected: list[int] = []
        remaining = list(candidates)
        covered: Counter[str] = Counter()
        while remaining and len(selected) < top_k:
            def score(doc_id: int) -> float:
                novelty = 0.0
                for intent, scores in intent_scores.items():
                    novelty += scores.get(doc_id, 0.0) / (1 + covered[intent])
                return self.lambda_mult * relevance_scores.get(doc_id, 0.0) + (1 - self.lambda_mult) * novelty

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
            for intent, scores in intent_scores.items():
                if scores.get(best, 0.0) > 0:
                    covered[intent] += 1
        return selected


class ClusterRetriever(BaseRetriever):
    """Cluster-first retrieval.

    Pseudocode:
        cluster documents by text features
        rank clusters by query similarity
        rank documents inside the best clusters

    Limitation: relevant isolated documents can be missed when clusters are poor.
    """

    def __init__(self, base_retriever: BaseRetriever | None = None) -> None:
        super().__init__()
        self.base_retriever = base_retriever or BM25Retriever()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.base_retriever.index(documents)
        self.documents = self.base_retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self.base_retriever.search(query, top_k=top_k)


class FederatedRetriever(BaseRetriever):
    """Search multiple retrievers and merge their results.

    Pseudocode:
        for each source retriever:
            run search in parallel or sequence
        fuse ranked lists with RRF

    Limitation: source selection and score calibration are application-specific.
    """

    def __init__(self, retrievers: Sequence[BaseRetriever]) -> None:
        super().__init__()
        self.retrievers = list(retrievers)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        lists = [retriever.search(query, top_k=max(top_k * 3, top_k)) for retriever in self.retrievers]
        return reciprocal_rank_fusion(lists, top_k=top_k)


class HierarchicalRetriever(BaseRetriever):
    """Retrieve parent documents first, then rank chunks inside them.

    Pseudocode:
        parent_results = parent_retriever.search(query)
        allowed_parents = top parent ids
        chunk_results = chunk_retriever.search(query)
        keep chunks whose parent_id is in allowed_parents

    Use for long PDFs, manuals, books, and nested knowledge bases.
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.parent_retriever = BM25Retriever()
        self.chunk_retriever = BM25Retriever()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.parent_retriever.index(documents)
        parents = self.parent_retriever.documents
        chunks = chunk_documents(parents, chunk_size=self.chunk_size, overlap=self.overlap)
        self.chunk_retriever.index(chunks)
        self.documents = self.chunk_retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        parent_results = self.parent_retriever.search(query, top_k=max(top_k, 3))
        allowed_parents = {doc_id for doc_id, _score in parent_results}
        chunk_results = self.chunk_retriever.search(query, top_k=max(top_k * 10, top_k))
        scores = {
            doc_id: score
            for doc_id, score in chunk_results
            if self.get_document(doc_id).metadata.get("parent_id", doc_id) in allowed_parents
        }
        return rank_top_k(scores, top_k)


class EntityRetriever(BaseRetriever):
    """Retrieve entity-like records extracted from documents.

    Pseudocode:
        extract capitalized names and known aliases
        build entity documents from aliases and descriptions
        rank entities with BM25

    Limitation: this compact extractor is not a full entity linker.
    """

    ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b")

    def __init__(self) -> None:
        super().__init__()
        self._retriever = BM25Retriever()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self._set_documents(documents)
        entity_docs: list[Document] = []
        for doc in self.documents:
            entities = self.ENTITY_RE.findall(doc.text)
            for i, entity in enumerate(entities):
                entity_docs.append(
                    Document(
                        id=doc.id * 10000 + i,
                        text=f"{entity}\n{doc.text}",
                        metadata={"entity": entity, "source_doc_id": doc.id},
                    )
                )
        self._retriever.index(entity_docs)
        self.documents = self._retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self._retriever.search(query, top_k=top_k)


class KnowledgeGraphRetriever:
    """Retrieve graph nodes by traversing from query entities.

    Pseudocode:
        find start nodes mentioned in query
        breadth-first traverse graph up to max_depth
        rank nodes by proximity and text match

    Limitation: requires a useful graph and entity mentions.
    """

    def __init__(self, graph: Mapping[str, Sequence[str]] | None = None, max_depth: int = 2) -> None:
        self.graph = {node: list(neighbors) for node, neighbors in (graph or {}).items()}
        self.max_depth = max_depth

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        starts = [node for node in self.graph if set(tokenize(node)) & terms]
        queue: deque[tuple[str, int]] = deque((node, 0) for node in starts)
        seen: set[str] = set()
        scores: dict[str, float] = {}
        while queue:
            node, depth = queue.popleft()
            if node in seen or depth > self.max_depth:
                continue
            seen.add(node)
            scores[node] = 1.0 / (1 + depth)
            for neighbor in self.graph.get(node, []):
                queue.append((neighbor, depth + 1))
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


class PageRank:
    """Query-independent authority ranking over a directed graph.

    Pseudocode:
        initialize all node ranks equally
        repeatedly distribute each node's rank to outgoing neighbors
        apply damping and teleport probability

    Use as an authority feature for web, citation, or entity graphs.
    """

    def __init__(self, damping: float = 0.85, iterations: int = 50, tolerance: float = 1e-8) -> None:
        self.damping = damping
        self.iterations = iterations
        self.tolerance = tolerance

    def rank(self, graph: Mapping[str, Sequence[str]]) -> dict[str, float]:
        nodes = set(graph)
        for neighbors in graph.values():
            nodes.update(neighbors)
        if not nodes:
            return {}
        ranks = {node: 1.0 / len(nodes) for node in nodes}
        outgoing = {node: list(graph.get(node, [])) for node in nodes}
        for _ in range(self.iterations):
            new_ranks = {node: (1 - self.damping) / len(nodes) for node in nodes}
            sink_rank = sum(ranks[node] for node, neighbors in outgoing.items() if not neighbors)
            for node in nodes:
                new_ranks[node] += self.damping * sink_rank / len(nodes)
            for node, neighbors in outgoing.items():
                if not neighbors:
                    continue
                share = self.damping * ranks[node] / len(neighbors)
                for neighbor in neighbors:
                    new_ranks[neighbor] += share
            delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
            ranks = new_ranks
            if delta < self.tolerance:
                break
        return ranks


class HITSRanker:
    """HITS authority and hub scoring.

    Pseudocode:
        authority[node] = sum hub[incoming_nodes]
        hub[node] = sum authority[outgoing_nodes]
        normalize and repeat

    Use when good hub pages and authoritative pages should be separated.
    """

    def __init__(self, iterations: int = 50, tolerance: float = 1e-8) -> None:
        self.iterations = iterations
        self.tolerance = tolerance

    def rank(self, graph: Mapping[str, Sequence[str]]) -> tuple[dict[str, float], dict[str, float]]:
        nodes = set(graph)
        for neighbors in graph.values():
            nodes.update(neighbors)
        incoming: dict[str, set[str]] = {node: set() for node in nodes}
        outgoing = {node: list(graph.get(node, [])) for node in nodes}
        for node, neighbors in outgoing.items():
            for neighbor in neighbors:
                incoming.setdefault(neighbor, set()).add(node)
        authorities = {node: 1.0 for node in nodes}
        hubs = {node: 1.0 for node in nodes}
        for _ in range(self.iterations):
            new_authorities = {node: sum(hubs[src] for src in incoming.get(node, set())) for node in nodes}
            new_hubs = {node: sum(new_authorities[dst] for dst in outgoing.get(node, [])) for node in nodes}
            _normalize_in_place(new_authorities)
            _normalize_in_place(new_hubs)
            delta = sum(abs(new_authorities[node] - authorities[node]) + abs(new_hubs[node] - hubs[node]) for node in nodes)
            authorities, hubs = new_authorities, new_hubs
            if delta < self.tolerance:
                break
        return authorities, hubs


def _normalize_in_place(scores: dict[str, float]) -> None:
    norm = sum(value * value for value in scores.values()) ** 0.5
    if norm:
        for key in list(scores):
            scores[key] /= norm


class PersonalizedRetriever(BaseRetriever):
    """Rerank results using user or session context.

    Pseudocode:
        base_results = retriever.search(query)
        score += personalization_boost(user_context, document)

    Limitation: personalization can introduce privacy and fairness concerns.
    """

    def __init__(self, retriever: BaseRetriever, boost_fn: Callable[[Mapping[str, Any], Document], float] | None = None) -> None:
        super().__init__()
        self.retriever = retriever
        self.boost_fn = boost_fn or (lambda _context, _doc: 0.0)

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.retriever.index(documents)
        self.documents = self.retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10, context: Mapping[str, Any] | None = None) -> list[tuple[int, float]]:
        context = context or {}
        scores = {}
        for doc_id, score in self.retriever.search(query, top_k=max(top_k * 5, top_k)):
            scores[doc_id] = score + self.boost_fn(context, self.get_document(doc_id))
        return rank_top_k(scores, top_k)


class PermissionAwareRetriever(BaseRetriever):
    """Filter results by access-control predicate.

    Pseudocode:
        retrieve candidates
        keep only documents where allowed(user, document) is true

    Use for enterprise and multi-tenant search.
    """

    def __init__(self, retriever: BaseRetriever, allowed_fn: Callable[[Any, Document], bool]) -> None:
        super().__init__()
        self.retriever = retriever
        self.allowed_fn = allowed_fn

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.retriever.index(documents)
        self.documents = self.retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10, user: Any = None) -> list[tuple[int, float]]:
        candidates = self.retriever.search(query, top_k=max(top_k * 10, top_k))
        scores = {
            doc_id: score
            for doc_id, score in candidates
            if self.allowed_fn(user, self.get_document(doc_id))
        }
        return rank_top_k(scores, top_k)


class RAGRetriever(BaseRetriever):
    """Retrieval pipeline for retrieval-augmented generation.

    Pseudocode:
        rewrite query if configured
        retrieve candidate chunks
        optionally rerank
        pack top documents into context strings

    Limitation: answer quality still depends on chunking, retrieval, and the
    downstream generator.
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        rewriter: Callable[[str], str] | None = None,
        reranker: Any | None = None,
    ) -> None:
        super().__init__()
        self.retriever = retriever or BM25Retriever()
        self.rewriter = rewriter
        self.reranker = reranker

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.retriever.index(documents)
        self.documents = self.retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        rewritten = self.rewriter(query) if self.rewriter else query
        candidates = self.retriever.search(rewritten, top_k=max(top_k * 5, top_k))
        if not self.reranker:
            return candidates[:top_k]
        ids = [doc_id for doc_id, _score in candidates]
        passages = [self.get_document(doc_id).text for doc_id in ids]
        return self.reranker.rerank(rewritten, passages, doc_ids=ids, top_k=top_k)

    def retrieve_context(self, query: str, top_k: int = 5, separator: str = "\n\n") -> str:
        docs = [self.get_document(doc_id).text for doc_id, _score in self.search(query, top_k=top_k)]
        return separator.join(docs)


__all__ = [
    "QueryExpander",
    "SynonymQueryExpander",
    "RocchioFeedback",
    "RM3Expander",
    "EmbeddingQueryExpander",
    "Doc2QueryExpander",
    "MultiQueryRetriever",
    "ConversationalQueryRewriter",
    "MMRDiversifier",
    "XQuADDiversifier",
    "ClusterRetriever",
    "FederatedRetriever",
    "HierarchicalRetriever",
    "EntityRetriever",
    "KnowledgeGraphRetriever",
    "PageRank",
    "HITSRanker",
    "PersonalizedRetriever",
    "PermissionAwareRetriever",
    "RAGRetriever",
]
