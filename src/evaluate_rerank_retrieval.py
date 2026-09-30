from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from evaluate_retrieval import (
    evaluate_question,
    first_relevant_rank_for_question,
    load_eval_questions,
)
from rerank_retrieve import RERANKER_MODEL_NAME, rerank_search
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
    first_rank = evaluation["first_relevant_rank"]
    print(f"first_relevant_rank: {first_rank if first_rank is not None else 'N/A'}")
    print("retrieved_top_5:")

    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['source']} | {result['section']}")


def find_expected_evidence_rank(question, results):
    if "acceptable_evidence" in question:
        rank = first_relevant_rank_for_question(question, results)
        if rank is None:
            return None, None
        return rank, results[rank - 1]

    expected_sources = set(question["expected_sources"])
    expected_terms = [term.lower() for term in question["expected_sections"]]

    for rank, result in enumerate(results, start=1):
        source_pass = result["source"] in expected_sources
        if not expected_terms:
            evidence_pass = True
        else:
            haystack = f"{result['section']}\n{result['text']}".lower()
            evidence_pass = any(term in haystack for term in expected_terms)

        if source_pass and evidence_pass:
            return rank, result

    return None, None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)
    chunks = load_processed_chunks()

    lexical_index = build_index(chunks)
    embedding_model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(chunks, embedding_model)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    total_hits = 0
    category_results = defaultdict(lambda: {"hits": 0, "total": 0})
    improved_questions = []
    unchanged_hits = []
    unchanged_misses = []
    regressions = []
    q9_full_results = None

    for question in questions:
        rerank_results = rerank_search(
            question["question"],
            lexical_index,
            vector_index,
            embedding_model,
            reranker,
            top_k=5,
            candidate_k=20,
        )
        vector_results = vector_search(
            question["question"],
            vector_index,
            embedding_model,
            top_k=5,
        )

        rerank_evaluation = evaluate_results(question, rerank_results)
        vector_evaluation = evaluate_results(question, vector_results)

        rerank_hit = rerank_evaluation["is_hit"]
        vector_hit = vector_evaluation["is_hit"]

        total_hits += int(rerank_hit)

        category = question["category"]
        category_results[category]["total"] += 1
        category_results[category]["hits"] += int(rerank_hit)

        if not vector_hit and rerank_hit:
            improved_questions.append(question["id"])
        elif vector_hit and rerank_hit:
            unchanged_hits.append(question["id"])
        elif not vector_hit and not rerank_hit:
            unchanged_misses.append(question["id"])
        else:
            regressions.append(question["id"])

        if question["id"] == "Q9":
            q9_full_results = rerank_search(
                question["question"],
                lexical_index,
                vector_index,
                embedding_model,
                reranker,
                top_k=40,
                candidate_k=20,
            )

        print_question_report(question, rerank_results, rerank_evaluation)
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

    print("vector_to_rerank_comparison:")
    print(f"improved_questions: {improved_questions}")
    print(f"unchanged_hits: {unchanged_hits}")
    print(f"unchanged_misses: {unchanged_misses}")
    print(f"regressions: {regressions}")

    q9_question = next(question for question in questions if question["id"] == "Q9")
    q9_rank, q9_result = find_expected_evidence_rank(q9_question, q9_full_results)
    print("q9_full_candidate_reranked_position:")
    if q9_rank is None:
        print("NOT FOUND IN FULL RERANKED CANDIDATE POOL")
    else:
        print(f"rank: {q9_rank}")
        print(f"source: {q9_result['source']}")
        print(f"section: {q9_result['section']}")
