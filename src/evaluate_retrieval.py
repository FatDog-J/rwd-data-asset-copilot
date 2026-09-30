import json
from collections import defaultdict
from pathlib import Path

from retrieve import build_index, load_processed_chunks, search


def load_eval_questions(eval_path):
    return json.loads(Path(eval_path).read_text(encoding="utf-8"))


def source_hit(results, expected_sources, source_match):
    retrieved_sources = {result["source"] for result in results}
    expected_sources = set(expected_sources)

    if source_match == "all":
        return expected_sources.issubset(retrieved_sources)
    if source_match == "any":
        return bool(expected_sources & retrieved_sources)

    raise ValueError(f"Unknown source_match value: {source_match}")


def evidence_term_hit(results, expected_sections):
    if not expected_sections:
        return True

    expected_terms = [term.lower() for term in expected_sections]

    return any(
        term in f"{result['section']}\n{result['text']}".lower()
        for result in results
        for term in expected_terms
    )


def acceptable_evidence_hit(results, acceptable_evidence, evidence_match):
    evidence_matches = []

    for evidence in acceptable_evidence:
        expected_source = evidence["source"]
        expected_section = evidence["section"].lower()
        evidence_matches.append(
            any(
                result["source"] == expected_source
                and expected_section in result["section"].lower()
                for result in results
            )
        )

    if evidence_match == "all":
        return all(evidence_matches)
    if evidence_match == "any":
        return any(evidence_matches)

    raise ValueError(f"Unknown evidence_match value: {evidence_match}")


def acceptable_source_hit(results, acceptable_evidence, evidence_match):
    retrieved_sources = {result["source"] for result in results}
    expected_sources = {evidence["source"] for evidence in acceptable_evidence}

    if evidence_match == "all":
        return expected_sources.issubset(retrieved_sources)
    if evidence_match == "any":
        return bool(expected_sources & retrieved_sources)

    raise ValueError(f"Unknown evidence_match value: {evidence_match}")


def first_acceptable_evidence_rank(results, acceptable_evidence):
    for rank, result in enumerate(results, start=1):
        for evidence in acceptable_evidence:
            if (
                result["source"] == evidence["source"]
                and evidence["section"].lower() in result["section"].lower()
            ):
                return rank

    return None


def first_relevant_rank(results, expected_sources, expected_sections):
    expected_sources = set(expected_sources)
    expected_terms = [term.lower() for term in expected_sections]

    for rank, result in enumerate(results, start=1):
        has_expected_source = result["source"] in expected_sources
        has_expected_evidence = any(
            term in f"{result['section']}\n{result['text']}".lower()
            for term in expected_terms
        )

        if has_expected_source and has_expected_evidence:
            return rank

    return None


def first_relevant_rank_for_question(question, results):
    if "acceptable_evidence" in question:
        return first_acceptable_evidence_rank(results, question["acceptable_evidence"])

    return first_relevant_rank(
        results,
        question["expected_sources"],
        question["expected_sections"],
    )


def evaluate_question(question, results):
    if "acceptable_evidence" in question:
        evidence_match = question.get("evidence_match", "any")
        source_match_pass = acceptable_source_hit(
            results,
            question["acceptable_evidence"],
            evidence_match,
        )
        evidence_term_match_pass = acceptable_evidence_hit(
            results,
            question["acceptable_evidence"],
            evidence_match,
        )

        return {
            "source_match_pass": source_match_pass,
            "evidence_term_match_pass": evidence_term_match_pass,
            "is_hit": source_match_pass and evidence_term_match_pass,
            "first_relevant_rank": first_relevant_rank_for_question(question, results),
        }

    source_match_pass = source_hit(
        results,
        question["expected_sources"],
        question["source_match"],
    )
    evidence_term_match_pass = evidence_term_hit(
        results,
        question["expected_sections"],
    )

    return {
        "source_match_pass": source_match_pass,
        "evidence_term_match_pass": evidence_term_match_pass,
        "is_hit": source_match_pass and evidence_term_match_pass,
        "first_relevant_rank": first_relevant_rank_for_question(question, results),
    }


def matched_sources(results, expected_sources):
    retrieved_sources = {result["source"] for result in results}
    return [
        source
        for source in expected_sources
        if source in retrieved_sources
    ]


def matched_evidence_terms(results, expected_sections):
    matched_terms = []

    for term in expected_sections:
        term_lower = term.lower()
        if any(
            term_lower in f"{result['section']}\n{result['text']}".lower()
            for result in results
        ):
            matched_terms.append(term)

    return matched_terms


def print_question_report(
    question,
    results,
    evaluation,
):
    print(f"id: {question['id']}")
    print(f"category: {question['category']}")
    print(f"source_match_pass: {evaluation['source_match_pass']}")
    print(f"evidence_term_match_pass: {evaluation['evidence_term_match_pass']}")
    print(f"final_result: {'HIT' if evaluation['is_hit'] else 'MISS'}")
    first_rank = evaluation["first_relevant_rank"]
    print(f"first_relevant_rank: {first_rank if first_rank is not None else 'N/A'}")
    print(f"expected_sources: {question.get('expected_sources', [])}")
    print(f"retrieved_sources: {[result['source'] for result in results]}")
    print(f"matched_sources: {matched_sources(results, question.get('expected_sources', []))}")
    print(f"acceptable_evidence: {question.get('acceptable_evidence', [])}")
    print(f"expected_evidence_terms: {question.get('expected_sections', [])}")
    if not question.get("expected_sections", []) and "acceptable_evidence" not in question:
        print("evidence_note: no expected evidence terms configured")
    print(
        "matched_evidence_terms: "
        f"{matched_evidence_terms(results, question.get('expected_sections', []))}"
    )
    print("retrieved_top_5:")

    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['source']} | {result['section']}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    questions = load_eval_questions(eval_path)
    chunks = load_processed_chunks()
    index = build_index(chunks)

    total_hits = 0
    category_results = defaultdict(lambda: {"hits": 0, "total": 0})

    for question in questions:
        results = search(question["question"], index, top_k=5)
        evaluation = evaluate_question(question, results)

        if evaluation["is_hit"]:
            total_hits += 1

        category = question["category"]
        category_results[category]["total"] += 1
        category_results[category]["hits"] += int(evaluation["is_hit"])

        print_question_report(
            question,
            results,
            evaluation,
        )
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
