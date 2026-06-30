from irlib import (
    CONTRADICTED,
    CONTRADICTED_CONCLUSION,
    LIKELY_SUPPORTED,
    PARTIALLY_SUPPORTED,
    SUPPORTED_CONCLUSION,
    SUPPORTS,
    UNCERTAIN,
    BayesianEvidenceCalibrator,
    CalibrationResult,
    CalibrationSignals,
    EvidenceSpan,
    EvidenceVerifier,
)


def evidence_span(
    label: str,
    score: float,
    *,
    source_prior: float = 0.5,
    reranker_score: float = 0.5,
    query_coverage: float = 0.8,
    claim_coverage: float = 0.8,
) -> EvidenceSpan:
    return EvidenceSpan(
        doc_id=1,
        text="Scurvy is caused by vitamin C deficiency.",
        start_char=0,
        end_char=42,
        label=label,
        score=score,
        metadata={
            "source_prior": source_prior,
            "reranker_score": reranker_score,
            "query_coverage": query_coverage,
            "claim_coverage": claim_coverage,
        },
    )


def test_bayesian_calibrator_returns_high_support_probability():
    calibrator = BayesianEvidenceCalibrator()
    evidence = [
        evidence_span(SUPPORTS, 0.88, source_prior=0.9, reranker_score=0.9, query_coverage=0.95, claim_coverage=0.9),
        evidence_span(SUPPORTS, 0.82, source_prior=0.85, reranker_score=0.8, query_coverage=0.9, claim_coverage=0.9),
    ]

    result = calibrator.calibrate(evidence)

    assert isinstance(result, CalibrationResult)
    assert result.probability_supported > 0.8
    assert result.credible_interval[0] < result.probability_supported < result.credible_interval[1]
    assert result.conclusion == SUPPORTED_CONCLUSION


def test_bayesian_calibrator_detects_contradiction_dominated_evidence():
    calibrator = BayesianEvidenceCalibrator()
    contradiction = evidence_span(
        "contradicts",
        0.9,
        source_prior=0.9,
        reranker_score=0.9,
        query_coverage=0.95,
        claim_coverage=0.95,
    )

    result = calibrator.calibrate([contradiction])

    assert result.conclusion == CONTRADICTED_CONCLUSION
    assert result.probability_supported < 0.45


def test_bayesian_calibrator_returns_uncertain_for_sparse_related_evidence():
    calibrator = BayesianEvidenceCalibrator()
    mention = evidence_span(
        "mentions",
        0.42,
        source_prior=0.5,
        reranker_score=0.5,
        query_coverage=0.45,
        claim_coverage=0.4,
    )

    result = calibrator.calibrate([mention])

    assert result.conclusion == UNCERTAIN
    assert 0.35 < result.probability_supported < 0.65
    assert result.credible_interval[1] - result.credible_interval[0] > 0.5


def test_bayesian_calibrator_uses_source_priors():
    calibrator = BayesianEvidenceCalibrator()
    high_trust = calibrator.calibrate([evidence_span(SUPPORTS, 0.72, source_prior=0.9, reranker_score=0.7)])
    low_trust = calibrator.calibrate([evidence_span(SUPPORTS, 0.72, source_prior=0.2, reranker_score=0.7)])

    assert high_trust.probability_supported > low_trust.probability_supported


def test_contradiction_lowers_probability_even_with_high_overlap():
    calibrator = BayesianEvidenceCalibrator()
    support = calibrator.calibrate(
        [evidence_span(SUPPORTS, 0.76, query_coverage=0.95, claim_coverage=0.95, reranker_score=0.8)]
    )
    contradiction = calibrator.calibrate(
        [evidence_span("contradicts", 0.76, query_coverage=0.95, claim_coverage=0.95, reranker_score=0.8)]
    )

    assert contradiction.probability_supported < support.probability_supported
    assert contradiction.conclusion == CONTRADICTED_CONCLUSION


def test_calibration_signals_can_be_passed_directly_and_are_deterministic():
    calibrator = BayesianEvidenceCalibrator()
    signals = CalibrationSignals(
        prior_probability=0.55,
        best_support_score=0.7,
        supporting_span_count=1,
        evidence_count=1,
        mean_source_prior=0.8,
        mean_reranker_score=0.7,
        mean_query_coverage=0.8,
        mean_claim_coverage=0.75,
    )

    first = calibrator.calibrate(signals=signals)
    second = calibrator.calibrate(signals=signals)

    assert first == second
    assert first.conclusion in {SUPPORTED_CONCLUSION, LIKELY_SUPPORTED}


def test_evidence_verifier_can_attach_calibrated_uncertainty():
    verifier = EvidenceVerifier(calibrator=BayesianEvidenceCalibrator())

    result = verifier.verify(
        "What causes scurvy and how is it treated?",
        ["Scurvy is caused by vitamin C deficiency. The passage does not discuss treatment."],
    )

    assert result.verdict == PARTIALLY_SUPPORTED
    assert result.answerable is False
    assert result.calibration is not None
    assert result.calibration.conclusion == UNCERTAIN
    assert "Calibrated conclusion: uncertain" in result.reason


def test_evidence_verifier_can_attach_calibrated_contradiction():
    verifier = EvidenceVerifier(calibrator=BayesianEvidenceCalibrator())

    result = verifier.verify(
        "Is scurvy treated with aspirin?",
        ["Scurvy is not treated with aspirin. It is treated with vitamin C."],
        claim="Scurvy is treated with aspirin.",
    )

    assert result.verdict == CONTRADICTED
    assert result.answerable is False
    assert result.calibration is not None
    assert result.calibration.conclusion == CONTRADICTED_CONCLUSION
