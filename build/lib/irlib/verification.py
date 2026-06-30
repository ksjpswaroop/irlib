"""Evidence verification and abstention for retrieved documents.

This module distinguishes retrieval relevance from evidentiary support. A
retrieved document can be topically similar without proving the requested claim.

Pseudocode:
    split retrieved documents into sentence-like spans
    score each span for query coverage, claim coverage, entity/number overlap,
    contradiction cues, optional reranker score, and source prior
    label spans as supports, contradicts, mentions, or irrelevant
    aggregate span labels into a final verdict

The default implementation is deterministic and does not download models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from irlib.core import Document, normalize_documents, tokenize


SUPPORTS = "supports"
CONTRADICTS = "contradicts"
MENTIONS = "mentions"
IRRELEVANT = "irrelevant"

SUPPORTED = "supported"
PARTIALLY_SUPPORTED = "partially_supported"
CONTRADICTED = "contradicted"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "cannot",
    "can't",
    "doesn't",
    "dont",
    "don't",
    "isn't",
    "wasn't",
    "without",
    "false",
    "incorrect",
    "refute",
    "refutes",
    "contradict",
    "contradicts",
    "unsupported",
}


@dataclass()
class EvidenceSpan:
    """A scored text span from a retrieved document.

    Pseudocode:
        span = sentence-like text from a retrieved document
        label = supports | contradicts | mentions | irrelevant
        score = confidence for that label

    `start_char` and `end_char` are offsets within the source document when
    available.
    """

    doc_id: int
    text: str
    start_char: int | None
    end_char: int | None
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass()
class VerificationResult:
    """Final evidence-verification result.

    Pseudocode:
        aggregate labeled evidence spans
        choose supported, partially_supported, contradicted, or
        insufficient_evidence
        optionally attach calibrated support probability
        include confidence, answerability, evidence, and a plain reason
    """

    query: str
    verdict: str
    confidence: float
    answerable: bool
    evidence: list[EvidenceSpan]
    reason: str
    calibration: Any | None = None


@dataclass()
class _CandidateSpan:
    doc: Document
    text: str
    start_char: int | None
    end_char: int | None


class EvidenceVerifier:
    """Deterministic verifier for retrieved evidence.

    Use after retrieval/reranking to decide whether the available documents
    support a query or claim, partially address it, contradict it, or leave the
    system unable to conclude. Pass a `BayesianEvidenceCalibrator` to convert
    span-level evidence into calibrated support probabilities.

    Pseudocode:
        candidates = extract_spans(retrieved_documents)
        reranker_scores = optional_reranker(query_or_claim, candidates)
        for span in candidates:
            score support and contradiction features
            assign a span label
        calibration = optional_calibrator(evidence)
        aggregate labels into a final verdict

    Limitation: the default verifier is heuristic. It is designed to be safe and
    testable, not to replace an NLI model.
    """

    def __init__(
        self,
        *,
        reranker: Any | None = None,
        calibrator: Any | None = None,
        source_priors: Mapping[str, float] | None = None,
        max_spans_per_doc: int = 5,
        support_threshold: float = 0.62,
        partial_threshold: float = 0.28,
        contradiction_threshold: float = 0.55,
    ) -> None:
        self.reranker = reranker
        self.calibrator = calibrator
        self.source_priors = dict(source_priors or {})
        self.max_spans_per_doc = max_spans_per_doc
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold
        self.contradiction_threshold = contradiction_threshold

    def verify(
        self,
        query: str,
        documents: Sequence[str | Mapping[str, Any] | Document],
        *,
        claim: str | None = None,
        top_k: int = 5,
    ) -> VerificationResult:
        """Verify whether retrieved documents support the query or claim."""

        normalized = normalize_documents(documents)[:top_k]
        target = claim or query
        candidates = self._extract_candidate_spans(normalized)
        if not candidates:
            result = VerificationResult(
                query=query,
                verdict=INSUFFICIENT_EVIDENCE,
                confidence=1.0,
                answerable=False,
                evidence=[],
                reason="No retrieved text was available to verify the query.",
            )
            return self._apply_calibration(result, [])

        reranker_scores = self._reranker_scores(target, [candidate.text for candidate in candidates])
        evidence = [
            self._label_span(query=query, claim=target, candidate=candidate, reranker_score=reranker_scores[i])
            for i, candidate in enumerate(candidates)
        ]
        evidence.sort(key=lambda span: (span.label != CONTRADICTS, -span.score, span.doc_id))
        result = self._aggregate(query, evidence)
        return self._apply_calibration(result, evidence)

    def answer_or_abstain(
        self,
        query: str,
        documents: Sequence[str | Mapping[str, Any] | Document],
        *,
        claim: str | None = None,
        top_k: int = 5,
    ) -> VerificationResult:
        """Alias for callers that want explicit abstention semantics."""

        return self.verify(query, documents, claim=claim, top_k=top_k)

    def _extract_candidate_spans(self, documents: Sequence[Document]) -> list[_CandidateSpan]:
        candidates: list[_CandidateSpan] = []
        for doc in documents:
            spans = _sentence_spans(doc.text)
            for text, start, end in spans[: self.max_spans_per_doc]:
                if text.strip():
                    candidates.append(_CandidateSpan(doc=doc, text=text.strip(), start_char=start, end_char=end))
        return candidates

    def _reranker_scores(self, target: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        if self.reranker is None:
            return [0.5] * len(passages)
        try:
            if hasattr(self.reranker, "score"):
                raw = [float(value) for value in self.reranker.score(target, passages)]
            elif callable(self.reranker):
                raw = [float(self.reranker(target, passage)) for passage in passages]
            else:
                return [0.5] * len(passages)
        except Exception:
            return [0.5] * len(passages)
        return _minmax(raw)

    def _label_span(self, *, query: str, claim: str, candidate: _CandidateSpan, reranker_score: float) -> EvidenceSpan:
        query_terms = _content_terms(query)
        claim_terms = _content_terms(claim)
        span_terms = _content_terms(candidate.text)

        query_coverage = _coverage(query_terms, span_terms)
        claim_coverage = _coverage(claim_terms, span_terms)
        entity_overlap = _entity_number_overlap(f"{query} {claim}", candidate.text)
        source_prior = self._source_prior(candidate.doc)
        contradiction_score = self._contradiction_score(claim, candidate.text, max(query_coverage, claim_coverage))

        support_score = _clamp(
            0.36 * query_coverage
            + 0.28 * claim_coverage
            + 0.14 * entity_overlap
            + 0.12 * reranker_score
            + 0.10 * source_prior
        )

        if contradiction_score >= self.contradiction_threshold:
            label = CONTRADICTS
            score = contradiction_score
        elif support_score >= self.support_threshold and claim_coverage >= 0.55:
            label = SUPPORTS
            score = support_score
        elif max(query_coverage, claim_coverage, support_score) >= self.partial_threshold:
            label = MENTIONS
            score = max(query_coverage, claim_coverage, support_score)
        else:
            label = IRRELEVANT
            score = max(query_coverage, claim_coverage, support_score)

        metadata = dict(candidate.doc.metadata)
        metadata.update(
            {
                "query_coverage": query_coverage,
                "claim_coverage": claim_coverage,
                "entity_number_overlap": entity_overlap,
                "reranker_score": reranker_score,
                "source_prior": source_prior,
                "contradiction_score": contradiction_score,
            }
        )
        return EvidenceSpan(
            doc_id=candidate.doc.id,
            text=candidate.text,
            start_char=candidate.start_char,
            end_char=candidate.end_char,
            label=label,
            score=score,
            metadata=metadata,
        )

    def _source_prior(self, doc: Document) -> float:
        for key in ("source_trust", "trust", "prior"):
            value = doc.metadata.get(key)
            if isinstance(value, (int, float)):
                return _clamp(float(value))
        for key in ("source", "domain", "kind"):
            value = doc.metadata.get(key)
            if value in self.source_priors:
                return _clamp(float(self.source_priors[value]))
        return 0.5

    def _contradiction_score(self, claim: str, span: str, overlap: float) -> float:
        if overlap < 0.25:
            return 0.0
        claim_negated = _has_negation(claim)
        span_negated = _has_negation(span)
        explicit_cue = any(phrase in span.lower() for phrase in ("is false", "are false", "is incorrect", "not true"))
        if claim_negated != span_negated:
            return _clamp(0.35 + 0.65 * overlap)
        if explicit_cue and overlap >= 0.4:
            return _clamp(0.25 + 0.75 * overlap)
        return 0.0

    def _aggregate(self, query: str, evidence: list[EvidenceSpan]) -> VerificationResult:
        contradictions = [span for span in evidence if span.label == CONTRADICTS]
        supports = [span for span in evidence if span.label == SUPPORTS]
        mentions = [span for span in evidence if span.label == MENTIONS]

        best_contradiction = max((span.score for span in contradictions), default=0.0)
        best_support = max((span.score for span in supports), default=0.0)
        best_mention = max((span.score for span in mentions), default=0.0)

        if best_contradiction >= self.contradiction_threshold and best_contradiction >= best_support * 0.9:
            return VerificationResult(
                query=query,
                verdict=CONTRADICTED,
                confidence=_clamp(best_contradiction),
                answerable=False,
                evidence=_top_evidence(contradictions, supports, mentions),
                reason="The strongest retrieved evidence contradicts the claim or query framing.",
            )

        if best_support >= self.support_threshold:
            confidence = _clamp(best_support * (1.0 - 0.35 * best_contradiction))
            return VerificationResult(
                query=query,
                verdict=SUPPORTED,
                confidence=confidence,
                answerable=True,
                evidence=_top_evidence(supports, contradictions, mentions),
                reason="Retrieved evidence directly supports the query or claim.",
            )

        if best_mention >= self.partial_threshold:
            return VerificationResult(
                query=query,
                verdict=PARTIALLY_SUPPORTED,
                confidence=_clamp(best_mention),
                answerable=False,
                evidence=_top_evidence(mentions, supports, contradictions),
                reason="Retrieved evidence is related, but it does not fully establish the requested conclusion.",
            )

        best_any = max((span.score for span in evidence), default=0.0)
        return VerificationResult(
            query=query,
            verdict=INSUFFICIENT_EVIDENCE,
            confidence=_clamp(1.0 - best_any),
            answerable=False,
            evidence=_top_evidence(evidence),
            reason="Retrieved documents do not provide enough evidence to answer confidently.",
        )

    def _apply_calibration(self, result: VerificationResult, evidence: list[EvidenceSpan]) -> VerificationResult:
        if self.calibrator is None:
            return result
        try:
            calibration = self.calibrator.calibrate(evidence)
        except Exception:
            return result

        conclusion = getattr(calibration, "conclusion", "")
        probability = _clamp(getattr(calibration, "probability_supported", result.confidence))
        calibrated_confidence = _clamp(getattr(calibration, "confidence", result.confidence))

        verdict = result.verdict
        answerable = result.answerable
        confidence = calibrated_confidence
        if conclusion == "contradicted":
            verdict = CONTRADICTED
            answerable = False
            confidence = max(confidence, 1.0 - probability)
        elif conclusion in {"supported", "likely_supported"}:
            verdict = SUPPORTED
            answerable = True
            confidence = max(confidence, probability)
        elif conclusion == "likely_unsupported":
            verdict = INSUFFICIENT_EVIDENCE
            answerable = False
        elif conclusion == "uncertain":
            has_related_evidence = any(span.label in {SUPPORTS, MENTIONS, CONTRADICTS} for span in evidence)
            verdict = PARTIALLY_SUPPORTED if has_related_evidence else INSUFFICIENT_EVIDENCE
            answerable = False

        reason = result.reason
        calibration_reason = getattr(calibration, "reason", "")
        if calibration_reason:
            reason = f"{reason} Calibrated conclusion: {conclusion}. {calibration_reason}"

        return replace(
            result,
            verdict=verdict,
            confidence=confidence,
            answerable=answerable,
            reason=reason,
            calibration=calibration,
        )


def _sentence_spans(text: str) -> list[tuple[str, int | None, int | None]]:
    spans: list[tuple[str, int | None, int | None]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        span = match.group(0).strip()
        if span:
            spans.append((span, match.start(), match.end()))
    if not spans and text.strip():
        spans.append((text.strip(), 0, len(text)))
    return spans


def _content_terms(text: str) -> set[str]:
    return {_normalize_term(term) for term in tokenize(text) if term not in STOPWORDS and len(term) > 1}


def _normalize_term(term: str) -> str:
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        root = term[:-2]
        if root.endswith(("at", "ct", "it", "nt", "rt", "ss", "ch", "sh")):
            return root
        return root + "e" if not root.endswith("e") else root
    if len(term) > 4 and term.endswith("ses"):
        return term[:-1]
    if len(term) > 4 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _coverage(target_terms: set[str], span_terms: set[str]) -> float:
    if not target_terms:
        return 0.0
    return len(target_terms & span_terms) / len(target_terms)


def _entity_number_overlap(target: str, span: str) -> float:
    target_items = _entities_and_numbers(target)
    if not target_items:
        return 0.5
    span_items = _entities_and_numbers(span)
    return len(target_items & span_items) / len(target_items)


def _entities_and_numbers(text: str) -> set[str]:
    entities = {
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", text)
        if match.group(0).lower() not in STOPWORDS
    }
    numbers = {match.group(0) for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text)}
    return entities | numbers


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    terms = set(tokenize(lowered, lowercase=False))
    return bool(terms & NEGATION_TERMS) or any(term in lowered for term in ("does not", "do not", "is not", "are not"))


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _top_evidence(*groups: Sequence[EvidenceSpan], limit: int = 5) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for group in groups:
        spans.extend(group)
    deduped: dict[tuple[int, str], EvidenceSpan] = {}
    for span in spans:
        key = (span.doc_id, span.text)
        if key not in deduped or span.score > deduped[key].score:
            deduped[key] = span
    return sorted(deduped.values(), key=lambda span: (-span.score, span.doc_id, span.text))[:limit]


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "CONTRADICTED",
    "CONTRADICTS",
    "INSUFFICIENT_EVIDENCE",
    "IRRELEVANT",
    "MENTIONS",
    "PARTIALLY_SUPPORTED",
    "SUPPORTED",
    "SUPPORTS",
    "EvidenceSpan",
    "EvidenceVerifier",
    "VerificationResult",
]
