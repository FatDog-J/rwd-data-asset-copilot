from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from evaluate_retrieval import (
    evaluate_question,
    load_eval_questions,
)
from hybrid_retrieve import hybrid_search
from retrieve import build_index
from vector_retrieve import (
    MODEL_NAME,
    build_vector_index,
    load_processed_chunks,
    vector_search,
)


def evaluate_results(question, results):
    return evaluate_question(question, results)


def print_question_report(question, results, evaluation):
    print(f"id: {question['id']}")
    print(f"category: {question['category']}")
    print(f"result: {'HIT' if evaluation['is_hit'] else 'MISS'}")
    print(f"source_match_pass: {evaluation['source_match_pass']}")
    print(f"evidence_term_match_pass: {evaluation['evidence_term_match_pass']}")
    first_rank = evaluation["first_relevant_rank"]
    print(f"first_relevant_rank: {first_rank if first_rank is not None else 'N/A'}")
    print(f"expected_sources: {question.get('expected_sources', [])}")
    print(f"acceptable_evidence: {question.get('acceptable_evidence', [])}")
    print("retrieved_top_5:")

    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['source']} | {result['section']}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)
    chunks = load_processed_chunks()

    lexical_index = build_index(chunks)
    model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(chunks, model)

    total_hits = 0
    category_results = defaultdict(lambda: {"hits": 0, "total": 0})
    improved_questions = []
    unchanged_hits = []
    unchanged_misses = []
    regressions = []

    for question in questions:
        hybrid_results = hybrid_search(
            question["question"],
            lexical_index,
            vector_index,
            model,
            top_k=5,
        )
        vector_results = vector_search(question["question"], vector_index, model, top_k=5)

        hybrid_evaluation = evaluate_results(question, hybrid_results)
        vector_evaluation = evaluate_results(question, vector_results)

        hybrid_hit = hybrid_evaluation["is_hit"]
        vector_hit = vector_evaluation["is_hit"]

        total_hits += int(hybrid_hit)

        category = question["category"]
        category_results[category]["total"] += 1
        category_results[category]["hits"] += int(hybrid_hit)

        if not vector_hit and hybrid_hit:
            improved_questions.append(question["id"])
        elif vector_hit and hybrid_hit:
            unchanged_hits.append(question["id"])
        elif not vector_hit and not hybrid_hit:
            unchanged_misses.append(question["id"])
        else:
            regressions.append(question["id"])

        print_question_report(question, hybrid_results, hybrid_evaluation)
        print()

    total_questions = len(questions)
    hit_at_5 = total_hits / total_questions if total_questions else 0

    print("Summary:")
    print(f"total_questions: {total_questions}")
    print(f"total_hits: {total_hits}")
    print(f"Hit@5: {hit_at_5:.2%}")
    print("results_by_category:")

    for category, result in sorted(category_results.items()):
        category_hit_at_5 = result["hits"] / result["total"]
        print(
            f"- {category}: "
            f"{result['hits']}/{result['total']} "
            f"({category_hit_at_5:.2%})"
        )

    print("vector_to_hybrid_comparison:")
    print(f"improved_questions: {improved_questions}")
    print(f"unchanged_hits: {unchanged_hits}")
    print(f"unchanged_misses: {unchanged_misses}")
    print(f"regressions: {regressions}")
