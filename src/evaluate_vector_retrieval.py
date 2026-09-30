from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from evaluate_retrieval import (
    evaluate_question,
    load_eval_questions,
)
from vector_retrieve import (
    MODEL_NAME,
    build_vector_index,
    load_processed_chunks,
    vector_search,
)


def print_question_report(question, results, is_hit, first_rank):
    print(f"id: {question['id']}")
    print(f"category: {question['category']}")
    print(f"result: {'HIT' if is_hit else 'MISS'}")
    print(f"first_relevant_rank: {first_rank if first_rank is not None else 'N/A'}")
    print("retrieved_top_5:")

    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['source']} | {result['section']}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)
    chunks = load_processed_chunks()
    model = SentenceTransformer(MODEL_NAME)
    index = build_vector_index(chunks, model)

    total_hits = 0
    category_results = defaultdict(lambda: {"hits": 0, "total": 0})

    for question in questions:
        results = vector_search(question["question"], index, model, top_k=5)
        evaluation = evaluate_question(question, results)
        is_hit = evaluation["is_hit"]
        first_rank = evaluation["first_relevant_rank"]

        total_hits += int(is_hit)

        category = question["category"]
        category_results[category]["total"] += 1
        category_results[category]["hits"] += int(is_hit)

        print_question_report(question, results, is_hit, first_rank)
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
