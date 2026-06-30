from irlib import (
    CONTRADICTED,
    CONTRADICTS,
    INSUFFICIENT_EVIDENCE,
    MENTIONS,
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    SUPPORTS,
    Document,
    EvidenceSpan,
    EvidenceVerifier,
    VerificationResult,
)


class FakeReranker:
    def score(self, query, passages):
        return [1.0 if "vitamin c deficiency" in passage.lower() else 0.0 for passage in passages]


def test_evidence_verifier_returns_supported_verdict():
    verifier = EvidenceVerifier()

    result = verifier.verify(
        "What causes scurvy?",
        ["Scurvy is caused by vitamin C deficiency. The condition improves with vitamin C."],
    )

    assert isinstance(result, VerificationResult)
    assert result.verdict == SUPPORTED
    assert result.answerable is True
    assert result.confidence > 0.6
    assert result.evidence[0].label == SUPPORTS
    assert "vitamin C deficiency" in result.evidence[0].text


def test_evidence_verifier_returns_partially_supported_verdict():
    verifier = EvidenceVerifier()

    result = verifier.verify(
        "What causes scurvy and how is it treated?",
        ["Scurvy is caused by vitamin C deficiency. The passage does not discuss treatment."],
    )

    assert result.verdict == PARTIALLY_SUPPORTED
    assert result.answerable is False
    assert result.evidence[0].label == MENTIONS
    assert "does not fully establish" in result.reason


def test_evidence_verifier_returns_contradicted_verdict_with_claim():
    verifier = EvidenceVerifier()

    result = verifier.verify(
        "Is scurvy treated with aspirin?",
        ["Scurvy is not treated with aspirin. It is treated with vitamin C."],
        claim="Scurvy is treated with aspirin.",
    )

    assert result.verdict == CONTRADICTED
    assert result.answerable is False
    assert result.confidence >= 0.8
    assert result.evidence[0].label == CONTRADICTS


def test_evidence_verifier_returns_insufficient_evidence_and_abstains():
    verifier = EvidenceVerifier()

    result = verifier.answer_or_abstain(
        "What causes scurvy?",
        ["Enterprise search systems rank documents by relevance."],
    )

    assert result.verdict == INSUFFICIENT_EVIDENCE
    assert result.answerable is False
    assert "do not provide enough evidence" in result.reason


def test_evidence_verifier_handles_empty_documents_as_insufficient():
    result = EvidenceVerifier().verify("What causes scurvy?", [])

    assert result.verdict == INSUFFICIENT_EVIDENCE
    assert result.confidence == 1.0
    assert result.evidence == []


def test_evidence_span_offsets_and_metadata_features_are_reported():
    doc = Document(
        id=42,
        text="Intro sentence. Scurvy is caused by vitamin C deficiency.",
        metadata={"source": "medical", "source_trust": 0.9},
    )
    verifier = EvidenceVerifier(reranker=FakeReranker(), source_priors={"medical": 0.8})

    result = verifier.verify("What causes scurvy?", [doc])

    assert isinstance(result.evidence[0], EvidenceSpan)
    assert result.evidence[0].doc_id == 42
    assert result.evidence[0].start_char is not None
    assert result.evidence[0].end_char is not None
    assert result.evidence[0].metadata["query_coverage"] > 0
    assert result.evidence[0].metadata["claim_coverage"] > 0
    assert result.evidence[0].metadata["reranker_score"] >= 0.0
    assert result.evidence[0].metadata["source_prior"] == 0.9
    assert result.evidence[0].metadata["contradiction_score"] == 0.0

