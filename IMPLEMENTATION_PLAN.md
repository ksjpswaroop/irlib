# irlib Implementation Plan

This roadmap extends `irlib` from a retrieval library into a retrieval, evidence-verification, and continuous-learning system.

The guiding principle is: the moat is not the base model; the moat is the learning loop. The system should capture retrieval traces, human corrections, verification outcomes, benchmark results, and preference data so every use can improve the next run.

References:

- Nadella learning-loop article attached in this workspace: `Microsoft CEO Satya Nadella argues that an organization's true competitive advantage (moat) in the AI era is no longer t.md`
- Forbes source URL: https://www.forbes.com/sites/sandycarter/2026/06/14/microsoft-ceo-nadella-argues-learning-loops-beat-picking-a-model/
- Qwen small model candidate: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen base model candidate: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base

## Current Status

- Phase 1 is complete: evidence verification and abstention are implemented in `src/irlib/verification.py`.
- Public Phase 1 exports are available from `irlib`: `EvidenceSpan`, `VerificationResult`, and `EvidenceVerifier`.
- Phase 2A is complete: Bayesian evidence calibration is implemented in `src/irlib/calibration.py`.
- Public Phase 2A exports are available from `irlib`: `CalibrationSignals`, `CalibrationResult`, and `BayesianEvidenceCalibrator`.
- `EvidenceVerifier` accepts an optional calibrator and keeps threshold-based behavior when no calibrator is supplied.
- Default verification and calibration are deterministic and do not download models.
- Verification status from the latest local run:
  - `.venv312/bin/python -m pytest -q` -> `31 passed, 3 skipped`
  - `.venv312/bin/python -m pytest -q -m slow` -> `2 passed, 2 skipped, 30 deselected`
  - `IRLIB_RUN_BENCHMARKS=1 .venv312/bin/python -m pytest -q -m benchmark` -> `1 passed, 33 deselected`

## Next Goal To Implement

Next goal: **Phase 2B: Bayesian Fusion Retriever**.

This is the strongest next step because evidence support is now calibrated. The retrieval layer can use that calibrated probability, plus sparse/dense/reranker scores and metadata priors, to produce posterior-style document relevance scores while preserving the existing `search(query, top_k)->list[tuple[int, float]]` API.

Implement Phase 2B before trace logging and bandits because the learning loop needs a stable policy surface: a fusion retriever whose weights, priors, and calibration outputs can be logged, evaluated, and improved later.

Deliverables:

- Add `src/irlib/bayesian.py`.
- Add `BayesianFusionRetriever`.
- Accept a list of retrievers or named retriever configs.
- Preserve the existing retriever contract:
  - `index(documents)`
  - `search(query, top_k=10) -> list[tuple[int, float]]`
  - `get_document(doc_id)`
- Combine signals:
  - BM25/TF-IDF/dense/hybrid scores
  - Reciprocal rank positions
  - source trust
  - freshness
  - authority/PageRank-style prior when metadata exists
  - user/context affinity when supplied
  - optional calibrated evidence probability from `BayesianEvidenceCalibrator`
- Return scores as calibrated posterior-style confidence values in `[0, 1]`.
- Add deterministic tests for:
  - API compatibility
  - fusion of two deterministic retrievers
  - source-prior influence
  - calibrated evidence influence
  - stable behavior when retrievers return disjoint result sets
- Update `README.md` with a Bayesian fusion example.
- Update benchmark wiring so `bayesian-fusion` can be evaluated later without changing the CLI shape.

Acceptance criteria:

- Existing retriever APIs remain stable.
- Default tests require no external model downloads or network access.
- `BayesianFusionRetriever` produces deterministic rankings for deterministic inputs.
- Scores are bounded to `[0, 1]` and documented as posterior-style confidence, not formal probability proof.
- Fusion can run without an evidence calibrator, but can consume one when supplied.

## Phase 2A: Bayesian Evidence Calibration

Status: implemented in `src/irlib/calibration.py`.

Goal: turn evidence-verification signals into `P(supported | evidence)` and a calibrated conclusion.

Implemented public types:

- `CalibrationSignals`
- `CalibrationResult`
- `BayesianEvidenceCalibrator`

Implemented behavior:

- Public types:
  - `CalibrationSignals`
  - `CalibrationResult`
  - `BayesianEvidenceCalibrator`
- Evidence-verification features used as signals:
  - best support score
  - best contradiction score
  - supporting span count
  - contradiction span count
  - mean source prior
  - mean reranker score
  - query/claim coverage
  - optional prior probability
- Calibration returns:
  - `probability_supported`
  - `credible_interval`
  - `confidence`
  - `conclusion`: `supported`, `likely_supported`, `uncertain`, `likely_unsupported`, or `contradicted`
  - `reason`
- Calibration is integrated into `EvidenceVerifier` behind an optional constructor argument so existing behavior remains stable:
  - `EvidenceVerifier(calibrator=BayesianEvidenceCalibrator())`
  - no calibrator keeps current threshold behavior
- Deterministic unit tests cover:
  - high-support evidence
  - contradiction-dominated evidence
  - sparse/uncertain evidence
  - source-prior influence
  - stable behavior with no model downloads
- `README.md` includes a short calibrated-verification example.

Acceptance criteria:

- [done] Existing public retrieval and verification APIs remain backward compatible.
- [done] Default tests require no external model downloads or network access.
- [done] Calibrated verification can return "uncertain" when evidence is topically related but not strong enough.
- [done] Tests demonstrate that contradiction evidence lowers support probability even when lexical overlap is high.
- [done] The implementation produces deterministic probabilities for deterministic inputs.

## Implementation Backlog

Marked for implementation:

- [done] Evidence verification harness with abstention
- [done] `EvidenceSpan`, `VerificationResult`, and `EvidenceVerifier`
- [done] `BayesianEvidenceCalibrator`
- [next] `BayesianFusionRetriever`
- [planned] Benchmark uncertainty and credible interval reporting
- [planned] Optional NLI/cross-encoder evidence verifier
- [planned] Retrieval trace logging and feedback capture
- [planned] Offline training dataset builder
- [planned] Contextual-bandit retrieval policy learner
- [planned] Qwen/Qwen3.5-0.8B verifier/query-rewriter adapter
- [planned] Distillation and periodic retraining harness
- [planned] Optional DPO/RL fine-tuning after supervised and bandit loops are stable

## Strong Recommendation

Do not start with full RL or continuous fine-tuning. Start with a disciplined learning loop:

1. [done] Evidence verification and abstention harness
2. [done] Bayesian calibrator for support confidence
3. [next] Bayesian fusion retriever
4. [planned] Feedback trace logging
5. [planned] Offline training dataset builder
6. [planned] Bandit retrieval-policy learner
7. [planned] Qwen/Qwen3.5-0.8B verifier/query-rewriter adapter
8. [planned] Distillation plus periodic retraining
9. [planned] Optional DPO/RL fine-tuning

This order gives `irlib` a measurable learning loop without letting a small model train on unverified outputs.

## Phase 1: Evidence Verification and Abstention

Status: implemented in `src/irlib/verification.py`.

Goal: distinguish "retrieved something similar" from "retrieved enough evidence to support the answer."

Add public types:

```python
EvidenceSpan(
    doc_id: int,
    text: str,
    start_char: int | None,
    end_char: int | None,
    label: str,  # supports, contradicts, mentions, irrelevant
    score: float,
    metadata: dict,
)

VerificationResult(
    query: str,
    verdict: str,  # supported, partially_supported, contradicted, insufficient_evidence
    confidence: float,
    answerable: bool,
    evidence: list[EvidenceSpan],
    reason: str,
)
```

Add `EvidenceVerifier`:

- Inputs: query, optional draft answer or claim, retrieved documents.
- Extract candidate spans from top documents.
- Score spans with deterministic heuristics first:
  - query coverage
  - entity/number overlap
  - negation and contradiction cues
  - reranker score if available
  - source prior if metadata exists
- Return a verdict:
  - `supported`: strong supporting spans, no strong contradiction
  - `partially_supported`: relevant evidence exists but does not fully answer
  - `contradicted`: contradiction signal dominates
  - `insufficient_evidence`: documents are weak, missing, or only tangential

Acceptance criteria:

- The verifier can abstain.
- Tests cover supported, partially supported, contradicted, and insufficient evidence cases.
- No model download is required for default tests.

## Phase 2: Bayesian Confidence and Fusion

Status: split into Phase 2A and Phase 2B. Phase 2A calibration is complete; Phase 2B fusion is next.

Goal: convert retrieval and evidence signals into calibrated probabilities.

Add `BayesianFusionRetriever`:

- Combines BM25, TF-IDF, dense, hybrid, reranker, metadata priors, and feedback priors.
- Outputs the standard `list[tuple[int, float]]`, where score is interpreted as posterior relevance confidence.
- Supports simple priors:
  - source trust
  - freshness
  - authority/PageRank
  - user/context affinity

`BayesianEvidenceCalibrator`:

- Inputs: retrieval signals, evidence signals, contradiction count, source priors.
- Output: `P(supported | evidence)`.
- Status: implemented with deterministic beta-posterior style calibration. Later iterations can learn or tune weights per retriever/source/query type.

Acceptance criteria:

- Existing retriever API remains stable.
- Calibration has deterministic tests.
- The final verification verdict uses calibrated confidence rather than raw thresholds only.

## Phase 3: Benchmark Uncertainty

Goal: avoid overclaiming from small benchmark samples.

Add benchmark uncertainty reporting:

- Bootstrap intervals for NDCG@k, Recall@k, MAP@k, MRR@k.
- Pairwise probability that retriever A beats retriever B on sampled queries.
- Report language that can say "cannot conclude" when intervals overlap heavily.

Output fields:

```json
{
  "ndcg@10": 0.6247,
  "ndcg@10_ci95": [0.54, 0.70],
  "p_beats_baseline": 0.91,
  "conclusion": "bm25 likely beats hybrid-hash on this sample"
}
```

Acceptance criteria:

- Benchmark reports include uncertainty by default.
- Tests use tiny deterministic metric arrays.
- README explains that benchmark point estimates are not definitive.

## Phase 4: Trace Logging and Feedback Capture

Goal: capture the loop data that compounds over time.

Add `RetrievalTrace`:

```python
RetrievalTrace(
    trace_id: str,
    query: str,
    retriever_config: dict,
    retrieved: list[dict],
    evidence_result: VerificationResult | None,
    latency_ms: float,
    timestamp: str,
)
```

Add `FeedbackRecord`:

```python
FeedbackRecord(
    trace_id: str,
    feedback_type: str,  # accepted, rejected, clicked, corrected, better_doc, unsafe, unsupported
    value: dict,
    source: str,         # user, benchmark, verifier, teacher
    timestamp: str,
)
```

Storage:

- Default local JSONL under `.cache/irlib-learning-loop/`.
- Path overridable with `IRLIB_LEARNING_LOOP_DIR`.
- Never log raw secrets.
- Support redaction hooks before writing traces.

Acceptance criteria:

- Trace logging is opt-in.
- Tests verify no traces are written unless enabled.
- Feedback can be converted into training examples.

## Phase 5: Offline Dataset Builder

Goal: turn traces and feedback into clean training/evaluation data.

Build datasets:

- Positive query-document pairs from accepted/clicked/supported evidence.
- Hard negatives from high-ranked rejected docs.
- Preference pairs: better doc/span over worse doc/span.
- Abstention examples: query plus retrieved docs where verdict was insufficient evidence.
- Contradiction examples where verifier or user flagged contradiction.

Outputs:

- `data/training/retriever_pairs.jsonl`
- `data/training/reranker_preferences.jsonl`
- `data/training/verifier_examples.jsonl`
- `data/training/query_rewrite_preferences.jsonl`

Acceptance criteria:

- Dataset builder is deterministic.
- Every training example includes provenance.
- Low-confidence self-generated examples are excluded by default.

## Phase 6: Bandit Retrieval Policy

Goal: learn which retrieval strategy works best per query type.

Start with contextual bandits, not full RL.

Actions:

- `bm25`
- `tfidf`
- `dense`
- `hybrid`
- `hybrid+rerank`
- `multi-query`
- `hyde`
- `retrieve-more`
- `abstain`

Context features:

- query length
- lexical specificity
- entity count
- score margin
- retriever agreement
- prior source/domain
- historical success for similar queries

Reward:

- positive: user accepted, relevant click, supported evidence, benchmark hit
- negative: unsupported answer, contradiction missed, user rejected, excessive latency, excessive token cost

Acceptance criteria:

- Initial policy can run in shadow mode.
- Policy decisions are logged with reward outcomes.
- Promotion requires benchmark and offline evaluation improvement.

## Phase 7: Qwen Verifier and Query-Rewriter Adapter

Goal: use a small local model as an optional helper, not as the retrieval moat.

Default candidate:

- `Qwen/Qwen3.5-0.8B`

Use cases:

- query rewriting
- evidence span classification
- support/contradiction explanation
- synthetic hard negative generation
- teacher-model distillation target

Interfaces:

```python
QwenVerifierAdapter(model_name="Qwen/Qwen3.5-0.8B")
QwenQueryRewriter(model_name="Qwen/Qwen3.5-0.8B")
```

Guardrails:

- Optional dependency path only.
- No model downloads in normal tests.
- Adapters accept mock model clients for deterministic testing.
- Model-generated labels must be marked as synthetic unless human or benchmark-confirmed.

Acceptance criteria:

- Unit tests use fake local model outputs.
- Real model smoke test is opt-in with `IRLIB_RUN_MODEL_TESTS=1`.
- Verifier output must feed the same `VerificationResult` schema.

## Phase 8: Distillation and Periodic Retraining

Goal: improve local adapters from verified traces and teacher labels.

Training modes:

- Supervised fine-tuning from human/benchmark labels.
- Preference training for query rewrites and verdicts.
- Distillation from a stronger teacher model.
- Optional DPO/RL only after stable supervised datasets exist.

Retraining policy:

- Never train directly on unverified model outputs.
- Train only on:
  - human-confirmed examples
  - benchmark-labeled examples
  - high-confidence verifier examples
  - teacher labels that pass evaluation gates

Promotion gate:

- Must improve benchmark metrics or abstention quality.
- Must not increase false-supported rate.
- Must pass latency budget.
- Must write model card metadata with training data provenance.

Acceptance criteria:

- Training jobs are explicit commands, not automatic background mutation.
- New model artifacts are versioned.
- Rollback path exists.

## Target Harness API

```python
harness = LearningLoopHarness(
    retriever=BayesianFusionRetriever(...),
    verifier=EvidenceVerifier(...),
    policy=BanditRetrievalPolicy(...),
    trace_store=JsonlTraceStore(...),
)

result = harness.answer_or_abstain(query, documents)
harness.record_feedback(result.trace_id, feedback)
harness.build_training_data()
harness.evaluate()
harness.train_if_ready()
```

Result behavior:

- If evidence is strong: return supported result with citations.
- If evidence is mixed: return partial or contradicted result.
- If evidence is weak: say that no conclusion can be drawn from the provided documents.

## Evaluation Metrics

Retrieval:

- NDCG@k
- Recall@k
- MAP@k
- MRR@k
- latency

Verification:

- supported precision
- insufficient-evidence precision
- false-supported rate
- contradiction detection rate
- abstention accuracy

Learning loop:

- improvement over previous model/policy
- probability of improvement with credible interval
- feedback conversion rate
- data quality rejection rate
- rollback frequency

## Non-Goals For The First Pass

- No always-on continuous fine-tuning.
- No hidden training on private data.
- No generated-answer benchmark before retrieval and evidence verification are stable.
- No model-specific architecture lock-in.
- No promotion of a trained adapter without evaluation gates.
