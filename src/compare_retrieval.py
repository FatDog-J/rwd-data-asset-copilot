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


def evaluate_result(question, results):
    source_match_pass = source_hit(
        results,
        question["expected_sources"],
        question["source_match"],
    )
    evidence_term_match_pass = evidence_term_hit(
        results,
        question["expected_sections"],
    )

    return source_match_pass and evidence_term_match_pass


def result_label(is_hit):
    return "HIT" if is_hit else "MISS"


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)

    lexical_chunks = load_lexical_chunks()
    lexical_index = build_index(lexical_chunks)

    vector_chunks = load_vector_chunks()
    vector_model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(vector_chunks, vector_model)

    improved = []
    unchanged_hits = []
    unchanged_misses = []
    regressions = []

    for question in questions:
        lexical_results = lexical_search(question["question"], lexical_index, top_k=5)
        vector_results = vector_search(
            question["question"],
            vector_index,
            vector_model,
            top_k=5,
        )

        lexical_hit = evaluate_result(question, lexical_results)
        vector_hit = evaluate_result(question, vector_results)

        if not lexical_hit and vector_hit:
            improved.append(question["id"])
        elif lexical_hit and vector_hit:
            unchanged_hits.append(question["id"])
        elif not lexical_hit and not vector_hit:
            unchanged_misses.append(question["id"])
        else:
            regressions.append(question["id"])

        print(f"id: {question['id']}")
        print(f"category: {question['category']}")
        print(f"lexical: {result_label(lexical_hit)}")
        print(f"vector: {result_label(vector_hit)}")
        print()

    print("Summary:")
    print(f"improved_questions: {improved}")
    print(f"unchanged_hits: {unchanged_hits}")
    print(f"unchanged_misses: {unchanged_misses}")
    print(f"regressions: {regressions}")
