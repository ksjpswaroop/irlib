"""Bayesian-style calibration for evidence verification.

This module turns labeled evidence spans into a calibrated probability that the
retrieved evidence supports the requested claim.

Pseudocode:
    extract support, contradiction, coverage, reranker, and source-prior signals
    combine them with a beta prior
    compute posterior mean as P(supported | evidence)
    approximate a credible interval from beta posterior variance
    map probability and contradiction strength to a calibrated conclusion

The implementation is deterministic and dependency-free. It is meant to make
abstention decisions less brittle, not to replace a trained NLI model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence


SUPPORTED_CONCLUSION = "supported"
LIKELY_SUPPORTED = "likely_supported"
UNCERTAIN = "uncertain"
LIKELY_UNSUPPORTED = "likely_unsupported"
CONTRADICTED_CONCLUSION = "contradicted"


@dataclass()
class CalibrationSignals:
    """Numerical evidence signals used by the calibrator.

    Use this dataclass when callers already computed retrieval/evidence
    features and want deterministic calibration without passing raw spans.

    Pseudocode:
        collect strongest support and contradiction scores
        count supporting, contradicting, and mentioning spans
        average source-prior, reranker, and coverage features
        pass the resulting signals into BayesianEvidenceCalibrator

    Limitation: these are summary features. They do not preserve full text or
    document-level dependencies between spans.
    """

    prior_probability: float = 0.5
    best_support_score: float = 0.0
    best_contradiction_score: float = 0.0
    supporting_span_count: int = 0
    contradiction_span_count: int = 0
    mention_span_count: int = 0
    evidence_count: int = 0
    mean_source_prior: float = 0.5
    mean_reranker_score: float = 0.5
    mean_query_coverage: float = 0.0
    mean_claim_coverage: float = 0.0

    @classmethod
    def from_evidence(
        cls,
        evidence: Sequence[Any],
        *,
        prior_probability: float = 0.5,
    ) -> "CalibrationSignals":
        """Build calibration signals from EvidenceSpan-like objects."""

        supports = [span for span in evidence if getattr(span, "label", "") == "supports"]
        contradictions = [span for span in evidence if getattr(span, "label", "") == "contradicts"]
        mentions = [span for span in evidence if getattr(span, "label", "") == "mentions"]

        return cls(
            prior_probability=_clamp(prior_probability),
            best_support_score=max((_score(span) for span in supports), default=0.0),
            best_contradiction_score=max((_score(span) for span in contradictions), default=0.0),
            supporting_span_count=len(supports),
            contradiction_span_count=len(contradictions),
            mention_span_count=len(mentions),
            evidence_count=len(evidence),
            mean_source_prior=_mean_metadata(evidence, "source_prior", default=0.5),
            mean_reranker_score=_mean_metadata(evidence, "reranker_score", default=0.5),
            mean_query_coverage=_mean_metadata(evidence, "query_coverage", default=0.0),
            mean_claim_coverage=_mean_metadata(evidence, "claim_coverage", default=0.0),
        )


@dataclass()
class CalibrationResult:
    """Calibrated support probability and conclusion.

    Use this result when the caller needs a probability, interval, and readable
    explanation before deciding whether to answer or abstain.

    Pseudocode:
        probability_supported = posterior_alpha / posterior_total
        credible_interval = normal_approximation(beta_posterior)
        confidence = probability distance from 0.5 adjusted by interval width
        conclusion = supported | likely_supported | uncertain |
                     likely_unsupported | contradicted

    Limitation: the interval is an approximation for deterministic calibration;
    it should be treated as a ranking and abstention aid, not a formal audit.
    """

    probability_supported: float
    credible_interval: tuple[float, float]
    confidence: float
    conclusion: str
    reason: str
    signals: CalibrationSignals


class BayesianEvidenceCalibrator:
    """Calibrate evidence support with a small beta-posterior model.

    Use this after `EvidenceVerifier` has labeled spans. It is useful when the
    system should distinguish strong support from topical but inconclusive
    evidence.

    Pseudocode:
        signals = CalibrationSignals.from_evidence(evidence)
        alpha = prior * prior_strength + weighted_support_evidence
        beta = (1-prior) * prior_strength + weighted_contradiction_evidence
        probability = alpha / (alpha + beta)
        interval = approximate_beta_interval(alpha, beta)
        conclusion = threshold(probability, interval, contradiction_strength)

    Limitation: weights are transparent heuristics. They should be learned or
    tuned against labeled verification data once enough traces exist.
    """

    def __init__(
        self,
        *,
        prior_probability: float = 0.5,
        prior_strength: float = 2.0,
        supported_threshold: float = 0.78,
        likely_supported_threshold: float = 0.64,
        likely_unsupported_threshold: float = 0.35,
        contradiction_threshold: float = 0.55,
        z_score: float = 1.96,
    ) -> None:
        self.prior_probability = _clamp(prior_probability)
        self.prior_strength = max(0.1, float(prior_strength))
        self.supported_threshold = _clamp(supported_threshold)
        self.likely_supported_threshold = _clamp(likely_supported_threshold)
        self.likely_unsupported_threshold = _clamp(likely_unsupported_threshold)
        self.contradiction_threshold = _clamp(contradiction_threshold)
        self.z_score = max(0.0, float(z_score))

    def calibrate(
        self,
        evidence: Sequence[Any] | None = None,
        *,
        signals: CalibrationSignals | None = None,
        prior_probability: float | None = None,
    ) -> CalibrationResult:
        """Return calibrated support probability for spans or prebuilt signals."""

        if signals is None:
            signals = CalibrationSignals.from_evidence(
                list(evidence or []),
                prior_probability=self.prior_probability if prior_probability is None else prior_probability,
            )
        elif prior_probability is not None:
            signals = CalibrationSignals(
                prior_probability=_clamp(prior_probability),
                best_support_score=signals.best_support_score,
                best_contradiction_score=signals.best_contradiction_score,
                supporting_span_count=signals.supporting_span_count,
                contradiction_span_count=signals.contradiction_span_count,
                mention_span_count=signals.mention_span_count,
                evidence_count=signals.evidence_count,
                mean_source_prior=signals.mean_source_prior,
                mean_reranker_score=signals.mean_reranker_score,
                mean_query_coverage=signals.mean_query_coverage,
                mean_claim_coverage=signals.mean_claim_coverage,
            )

        alpha, beta = self._posterior_parameters(signals)
        probability = alpha / (alpha + beta)
        interval = _credible_interval(alpha, beta, self.z_score)
        conclusion = self._conclusion(signals, probability, interval)
        confidence = self._confidence(probability, interval)
        return CalibrationResult(
            probability_supported=_clamp(probability),
            credible_interval=interval,
            confidence=confidence,
            conclusion=conclusion,
            reason=self._reason(conclusion, probability, interval, signals),
            signals=signals,
        )

    def _posterior_parameters(self, signals: CalibrationSignals) -> tuple[float, float]:
        prior = _clamp(signals.prior_probability)
        coverage = _clamp((signals.mean_query_coverage + signals.mean_claim_coverage) / 2.0)

        support_evidence = (
            signals.supporting_span_count * (0.75 + signals.best_support_score)
            + max(0.0, coverage - 0.35) * 1.4
            + max(0.0, signals.mean_source_prior - 0.5) * 1.2
            + max(0.0, signals.mean_reranker_score - 0.5) * 0.7
        )
        contradiction_evidence = (
            signals.contradiction_span_count * (0.9 + signals.best_contradiction_score) * 1.5
            + signals.best_contradiction_score * 1.6
            + max(0.0, 0.5 - signals.mean_source_prior) * 1.1
            + max(0.0, 0.5 - signals.mean_reranker_score) * 0.5
        )

        if signals.evidence_count == 0:
            contradiction_evidence += 0.2
        if signals.supporting_span_count == 0 and signals.contradiction_span_count == 0:
            contradiction_evidence += max(0.0, 0.45 - coverage) * 0.8

        alpha = prior * self.prior_strength + support_evidence
        beta = (1.0 - prior) * self.prior_strength + contradiction_evidence
        return max(alpha, 0.01), max(beta, 0.01)

    def _conclusion(
        self,
        signals: CalibrationSignals,
        probability: float,
        interval: tuple[float, float],
    ) -> str:
        lower, upper = interval
        if (
            signals.contradiction_span_count > 0
            and signals.best_contradiction_score >= self.contradiction_threshold
            and probability <= 0.45
        ):
            return CONTRADICTED_CONCLUSION
        if probability >= self.supported_threshold and lower >= 0.55:
            return SUPPORTED_CONCLUSION
        if probability >= self.likely_supported_threshold:
            return LIKELY_SUPPORTED
        if probability <= self.likely_unsupported_threshold and upper <= 0.55:
            return LIKELY_UNSUPPORTED
        return UNCERTAIN

    def _confidence(self, probability: float, interval: tuple[float, float]) -> float:
        width = max(0.0, interval[1] - interval[0])
        distance_from_uncertain = abs(probability - 0.5) * 2.0
        return _clamp(0.55 * distance_from_uncertain + 0.45 * (1.0 - width))

    def _reason(
        self,
        conclusion: str,
        probability: float,
        interval: tuple[float, float],
        signals: CalibrationSignals,
    ) -> str:
        interval_text = f"{interval[0]:.2f}, {interval[1]:.2f}"
        base = f"P(supported | evidence)={probability:.2f}; credible interval=[{interval_text}]."
        if conclusion == CONTRADICTED_CONCLUSION:
            return f"{base} Contradiction evidence dominates the support signal."
        if conclusion == SUPPORTED_CONCLUSION:
            return f"{base} Support evidence is strong and the lower interval bound is above uncertainty."
        if conclusion == LIKELY_SUPPORTED:
            return f"{base} Support evidence is positive, but uncertainty remains."
        if conclusion == LIKELY_UNSUPPORTED:
            return f"{base} Available evidence is weak or low-trust."
        if signals.mention_span_count or signals.evidence_count:
            return f"{base} Evidence is related, but calibration cannot conclude support."
        return f"{base} No evidence was available for calibration."


def _mean_metadata(evidence: Sequence[Any], key: str, *, default: float) -> float:
    values: list[float] = []
    for span in evidence:
        metadata = getattr(span, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                values.append(_clamp(float(value)))
    if not values:
        return default
    return sum(values) / len(values)


def _score(span: Any) -> float:
    value = getattr(span, "score", 0.0)
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    return 0.0


def _credible_interval(alpha: float, beta: float, z_score: float) -> tuple[float, float]:
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / ((total**2) * (total + 1.0))
    radius = z_score * sqrt(max(0.0, variance))
    return _clamp(mean - radius), _clamp(mean + radius)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "CONTRADICTED_CONCLUSION",
    "LIKELY_SUPPORTED",
    "LIKELY_UNSUPPORTED",
    "SUPPORTED_CONCLUSION",
    "UNCERTAIN",
    "BayesianEvidenceCalibrator",
    "CalibrationResult",
    "CalibrationSignals",
]
