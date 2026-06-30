"""Advanced IR helpers: late interaction, HyDE, and perturbation explanations."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from irlib.core import cosine_similarity, tokenize
from irlib.dense import HashingEncoder


class ColBERTInteraction:
    """ColBERT-style late interaction scoring.

    Pseudocode:
        encode query tokens independently
        encode document tokens independently
        score = sum over query tokens of max token similarity in document

    Use for higher-precision neural retrieval experiments.

    Limitation: real ColBERT quality requires a trained late-interaction model;
    the fallback encoder is deterministic but not truly semantic.
    """

    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2", encoder: Any | None = None) -> None:
        self.encoder_model = encoder_model
        self.encoder = encoder
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(encoder_model)
            except Exception:
                self.encoder = HashingEncoder()

    def _encode_tokens(self, tokens: Sequence[str]) -> list[list[float]]:
        if hasattr(self.encoder, "encode"):
            vectors = self.encoder.encode(list(tokens))
        else:
            vectors = self.encoder(list(tokens))
        return [[float(value) for value in vector] for vector in vectors]

    def compute_late_interaction(self, query: str, doc: str) -> float:
        q_tokens = tokenize(query)
        d_tokens = tokenize(doc)
        if not q_tokens or not d_tokens:
            return 0.0
        q_vectors = self._encode_tokens(q_tokens)
        d_vectors = self._encode_tokens(d_tokens)
        score = 0.0
        for q_vector in q_vectors:
            score += max(cosine_similarity(q_vector, d_vector) for d_vector in d_vectors)
        return score


class HydeExpander:
    """HyDE-style hypothetical document generation.

    Pseudocode:
        hypothetical = llm(query) or deterministic fallback text
        use hypothetical text as expanded retrieval query

    Use when short queries need more semantic context before dense retrieval.
    """

    def __init__(self, llm_client: Any | None = None, generator: Callable[[str], str] | None = None) -> None:
        self.llm = llm_client
        self.generator = generator

    def generate_hypothetical_answer(self, query: str) -> str:
        if self.generator:
            return self.generator(query)
        if not self.llm:
            return f"Hypothetical answer about {query}. Relevant evidence should discuss {query} in detail."
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Write a short hypothetical answer for retrieval: {query}"}],
        )
        return response.choices[0].message.content or ""

    def expand(self, query: str) -> str:
        return self.generate_hypothetical_answer(query)


class LIMEExplainer:
    """Perturbation-based retrieval explanation.

    Pseudocode:
        compute base score(query, document)
        remove each query term
        measure score drop for each removal

    Use to debug which query terms influence a retrieval score.
    """

    def __init__(self, retriever_fn: Callable[[str, str], float]) -> None:
        self.retriever = retriever_fn

    def explain(self, query: str, doc: str) -> dict[str, Any]:
        terms = tokenize(query)
        base = float(self.retriever(query, doc))
        impacts = {}
        for i, term in enumerate(terms):
            perturbed = " ".join(value for j, value in enumerate(terms) if j != i)
            impacts[term] = base - float(self.retriever(perturbed, doc))
        return {"query_terms": terms, "base_score": base, "impact": impacts}


__all__ = ["ColBERTInteraction", "HydeExpander", "LIMEExplainer"]

