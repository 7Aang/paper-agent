# Evaluation protocol

## Dataset states

- `auto-generated`: may be used to create candidates, never presented as human gold.
- `verified=false`: a frozen diagnostic label that has not received independent human review.
- `verified=true`: requires a recorded human review. The current benchmark contains zero such questions.
- `frozen`: content hash is stored in `benchmark/manifest.json`; CI fails if content changes without an explicit new manifest.

The current 60-question set is bound by hash to a 50-paper corpus. It contains 50 answerable retrieval diagnostics and 10 constructed unanswerable questions. The labels comprise 24 retained seed questions and 26 arXiv-abstract-derived questions; none has independent human verification. It is useful for regression and ablation, but it is not a public benchmark and does not measure generalization to unseen papers.

## Retrieval levels

Paper-level evaluation deduplicates retrieved chunks by `paper_id`. Chunk/page-level evaluation requires explicit relevant chunk or page labels. These labels are currently incomplete, so the main report only claims paper-level metrics.

## Citation levels

Deterministic checks verify that every returned claim includes paper ID, page, evidence ID, and an exact quote that appears on that page. They establish provenance, not semantic entailment. Semantic citation correctness remains `NOT RUN` until fixed human labels or a calibrated, fully recorded LLM judge run is available.

## External models

OpenAI-compatible LLM runs, pretrained sentence embeddings, and Cross-Encoder reranking are separate optional experiments. A missing key or model produces `REQUIRES API KEY` or `NOT RUN`; the report does not substitute estimated values.
