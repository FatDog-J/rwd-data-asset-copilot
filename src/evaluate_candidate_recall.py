from pathlib import Path

from sentence_transformers import SentenceTransformer

from evaluate_retrieval import evidence_term_hit, load_eval_questions, source_hit
from retrieve import (
    build_index,
    load_processed_chunks as load_lexical_chunks,
    search as lexical_search,
)
from vector_retrieve import (
    MODEL_NAME,
    build_vector_index,
    load_processed_chunks as load_vector_chunks,
    vector_search,
)


def chunk_key(result):
    return result["source"], result["section"], result["text"]


def combine_candidates(lexical_results, vector_results):
    candidates = {}

    for retriever_name, results in (
        ("lexical", lexical_results),
        ("vector", vector_results),
    ):
        for result in results:
            key = chunk_key(result)
            if key not in candidates:
                candidates[key] = {
                    "source": result["source"],
                    "section": result["section"],
                    "text": result["text"],
                    "retrievers": set(),
                }
            candidates[key]["retrievers"].add(retriever_name)

    return list(candidates.values())


def is_relevant_candidate(question, candidate):
    return (
        source_hit([candidate], question["expected_sources"], "any")
        and evidence_term_hit([candidate], question["expected_sections"])
    )


def has_expected_sources(candidates, question):
    return source_hit(
        candidates,
        question["expected_sources"],
        question["source_match"],
    )


def relevant_retrievers(candidates, question):
    retrievers = set()

    for candidate in candidates:
        if is_relevant_candidate(question, candidate):
            retrievers.update(candidate["retrievers"])

    return sorted(retrievers)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)

    lexical_chunks = load_lexical_chunks()
    lexical_index = build_index(lexical_chunks)

    vector_chunks = load_vector_chunks()
    model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(vector_chunks, model)

    candidate_hits = 0

    for question in questions:
        lexical_results = lexical_search(question["question"], lexical_index, top_k=20)
        vector_results = vector_search(question["question"], vector_index, model, top_k=20)
        candidates = combine_candidates(lexical_results, vector_results)

        source_match_pass = has_expected_sources(candidates, question)
        evidence_term_match_pass = evidence_term_hit(
            candidates,
            question["expected_sections"],
        )
        candidate_hit = source_match_pass and evidence_term_match_pass
        candidate_hits += int(candidate_hit)

        retrievers = relevant_retrievers(candidates, question)
        retriever_label = ", ".join(retrievers) if retrievers else "N/A"

        print(f"id: {question['id']}")
        print(f"candidate_result: {'HIT' if candidate_hit else 'MISS'}")
        print(f"candidate_pool_size: {len(candidates)}")
        print(f"relevant_evidence_retrievers: {retriever_label}")
        print()

    total_questions = len(questions)
    candidate_recall = candidate_hits / total_questions if total_questions else 0

    print("Summary:")
    print(f"total_questions: {total_questions}")
    print(f"candidate_hits: {candidate_hits}")
    print(f"Candidate Recall: {candidate_recall:.2%}")
