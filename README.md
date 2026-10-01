# RWD Data Asset & Methodology Copilot

## Problem

## Target User

## Example Questions

## Data Source

## V0.1 Scope

## Out of Scope

## Acceptance Criteria

Lexical Baseline v0.1

Corpus:
- 6 source documents
- 77 indexed semantic chunks

Evaluation:
- 10 fixed analyst questions
- Hit@5 = 70%

True misses:
- Q1
- Q7
- Q9

Observed weakness:
Lexical retrieval performs well for explicit terminology and
exact field/table references, but struggles when user wording
requires semantic mapping to field names or methodological
limitations.

### Retrieval Experiment v0.1

Lexical retrieval:
- Hit@5: 70%

Vector retrieval:
- Model: all-MiniLM-L6-v2
- 384-dimensional embeddings
- Hit@5: 80%

Comparison:
- Improved: Q7
- Regressions: none
- Persistent failures: Q1, Q9

Observation:
Vector retrieval improved semantic mapping from natural-language
analyst terminology to a dataset field (`admission_type`) without
regressing on the questions already handled by lexical retrieval.


# Retrieval Experiment

Lexical:
Hit@5 = 70%

Vector:
Hit@5 = 80%
Improved Q7 with no regressions.

RRF Hybrid:
Hit@5 = 80%
No improvement over vector retrieval.

Failure diagnostics:
Q1:
- lexical gold rank = 8
- vector gold rank = 11

Q9:
- lexical gold rank = 17
- vector gold rank > 20

Interpretation:
The remaining failures are primarily ranking problems rather than
missing evidence in the corpus. Broader candidate retrieval followed
by reranking is the next experiment.

### Retrieval Architecture Experiment

Lexical retrieval:
- Hit@5: 70%

Vector retrieval:
- Hit@5: 80%
- Improved Q7
- No regressions

RRF hybrid retrieval:
- Hit@5: 80%
- No improvement over vector retrieval

Two-stage retrieval with CrossEncoder reranking:
- Candidate Recall: 100%
- Hit@5: 90%
- Improved Q1
- No regressions

Persistent hard case:
- Q9 remained outside Top 5
- Gold evidence was reranked to rank 14

Decision:
Use lexical + vector candidate retrieval followed by
CrossEncoder reranking as the v0.1 retrieval architecture.


## Retrieval Benchmark v1

Evaluation set: 10 analyst questions

- Lexical Hit@5: 80%
- Vector Hit@5: 90%
- RRF Hybrid Hit@5: 90%
- Two-stage retrieval + CrossEncoder reranking Hit@5: 100%

Final retrieval architecture:
Lexical Top 20 + Vector Top 20
→ union/deduplicate
→ CrossEncoder reranking
→ Top 5 evidence

## Generation Benchmark v1
Deterministic evaluation
────────────────────────
Schema validity             8 / 8
Source containment          8 / 8
Quantitative hallucination  0
Abstention behavior         2 / 2

Semantic evaluation
────────────────────────
Overall pass                8 / 8
Correctness                 2.00 / 2
Groundedness                2.00 / 2
Completeness                2.00 / 2
Source faithfulness         2.00 / 2
Abstention                  2.00 / 2
