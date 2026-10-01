import json
import re
from pathlib import Path

from models import RWDAnswer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = PROJECT_ROOT / "evals" / "generation_outputs.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "generation_eval_results.json"


PERCENT_RE = re.compile(
    r"\b(?:approximately|about|around|roughly)?\s*\d+(?:\.\d+)?\s*(?:%|percent)\b",
    re.IGNORECASE,
)
FRACTION_RESULT_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine)-"
    r"(?:third|quarter|fifth|sixth|seventh|eighth|ninth|tenth)s?\b",
    re.IGNORECASE,
)
ABSTENTION_TERMS = [
    "does not provide",
    "does not include",
    "is not provided",
    "are not provided",
    "no percentage is provided",
    "no statistics are provided",
    "no aggregate statistics are provided",
    "not provided",
    "not available in the retrieved evidence",
    "not available in the documentation",
    "insufficient evidence",
    "cannot be quantified",
    "cannot be determined",
    "cannot calculate",
    "cannot be calculated",
    "available evidence is insufficient",
    "would require patient-level analysis",
    "would require analysis of the underlying data",
    "requires patient-level analysis",
    "aggregated study results would be needed",
]


def load_generation_outputs():
    return json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))["cases"]


def normalize_text(text):
    return text.lower().replace("_", " ")


def answer_search_text(answer):
    parts = [
        answer.answer,
        " ".join(answer.relevant_tables_or_fields),
        " ".join(answer.limitations),
    ]
    return normalize_text(" ".join(parts))


def required_point_diagnostics(answer, required_points):
    search_text = answer_search_text(answer)
    diagnostics = []

    for point in required_points:
        normalized_point = normalize_text(point)
        diagnostics.append(
            {
                "required_point": point,
                "literal_match": normalized_point in search_text,
            }
        )

    return diagnostics


def has_forbidden_quantitative_claim(answer):
    text = " ".join([answer.answer, " ".join(answer.limitations)])
    return bool(PERCENT_RE.search(text) or FRACTION_RESULT_RE.search(text))


def abstention_behavior(answer, answerable):
    if answerable:
        return "N/A"

    text = normalize_text(" ".join([answer.answer, " ".join(answer.limitations)]))
    if any(term in text for term in ABSTENTION_TERMS):
        return "PASS"
    if "not" in text and any(term in text for term in ["evidence", "documentation", "available"]):
        return "UNCERTAIN"
    return "FAIL"


def sources_within_context(answer, evidence):
    context_sources = {result["source"] for result in evidence}
    invalid_sources = [
        source
        for source in answer.sources
        if source not in context_sources
    ]
    return len(invalid_sources) == 0, invalid_sources


def print_case_report(result):
    print(f"ID: {result['id']}")
    print(f"Category: {result['category']}")
    print(f"Question: {result['question']}")
    print()
    print("Retrieved evidence:")
    for item in result["retrieved_evidence"]:
        print(f"- {item['source']} | {item['section']}")
    print()
    print("Generated RWDAnswer:")
    answer = result["answer"]
    print(f"- answer: {answer['answer']}")
    print(f"- relevant_tables_or_fields: {answer['relevant_tables_or_fields']}")
    print(f"- limitations: {answer['limitations']}")
    print(f"- sources: {answer['sources']}")
    print()
    print("Deterministic checks:")
    checks = result["checks"]
    print(f"- schema_valid: {checks['schema_valid']}")
    print(f"- sources_within_context: {checks['sources_within_context']}")
    print(f"- invalid_sources: {checks['invalid_sources']}")
    print(f"- forbidden_quantitative_claim: {checks['forbidden_quantitative_claim']}")
    print(f"- abstention_behavior: {checks['abstention_behavior']}")
    print(f"- required_point_diagnostics: {checks['required_point_diagnostics']}")
    print()


def validate_answer(answer_data):
    try:
        return RWDAnswer.model_validate(answer_data), True
    except Exception:
        return RWDAnswer(
            answer="",
            relevant_tables_or_fields=[],
            limitations=[],
            sources=[],
        ), False


def evaluate_case(frozen_case):
    evidence = frozen_case["retrieved_evidence"]
    answer, schema_valid = validate_answer(frozen_case["generated_answer"])
    source_pass, invalid_sources = sources_within_context(answer, evidence)
    quantitative_claim = has_forbidden_quantitative_claim(answer)
    abstention = abstention_behavior(answer, frozen_case["answerable"])
    required_diagnostics = required_point_diagnostics(
        answer,
        frozen_case["required_points"],
    )

    return {
        "id": frozen_case["id"],
        "category": frozen_case["category"],
        "question": frozen_case["question"],
        "answerable": frozen_case["answerable"],
        "retrieved_evidence": evidence,
        "answer": answer.model_dump(),
        "checks": {
            "schema_valid": schema_valid,
            "sources_within_context": source_pass,
            "invalid_sources": invalid_sources,
            "forbidden_quantitative_claim": quantitative_claim,
            "abstention_behavior": abstention,
            "required_point_diagnostics": required_diagnostics,
        },
    }


def summarize(results):
    total_cases = len(results)
    schema_valid_count = sum(result["checks"]["schema_valid"] for result in results)
    source_pass_count = sum(
        result["checks"]["sources_within_context"]
        for result in results
    )
    quantitative_claim_count = sum(
        result["checks"]["forbidden_quantitative_claim"]
        for result in results
    )
    abstention_results = [
        result
        for result in results
        if result["answerable"] is False
    ]
    abstention_pass_count = sum(
        result["checks"]["abstention_behavior"] == "PASS"
        for result in abstention_results
    )

    return {
        "total_cases": total_cases,
        "schema_valid_count": schema_valid_count,
        "source_containment_pass_count": source_pass_count,
        "quantitative_hallucination_count": quantitative_claim_count,
        "abstention_pass_count": abstention_pass_count,
        "abstention_cases": len(abstention_results),
        "cases_with_invalid_sources": [
            result["id"]
            for result in results
            if result["checks"]["invalid_sources"]
        ],
        "cases_with_suspicious_quantitative_claims": [
            result["id"]
            for result in results
            if result["checks"]["forbidden_quantitative_claim"]
        ],
        "cases_with_abstention_failure_or_uncertainty": [
            result["id"]
            for result in abstention_results
            if result["checks"]["abstention_behavior"] != "PASS"
        ],
    }


if __name__ == "__main__":
    cases = load_generation_outputs()
    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)
        print_case_report(result)

    summary = summarize(results)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "results": results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("SUMMARY")
    print(f"total_cases: {summary['total_cases']}")
    print(f"schema_valid_count: {summary['schema_valid_count']}")
    print(f"source_containment_pass_count: {summary['source_containment_pass_count']}")
    print(
        "quantitative_hallucination_count: "
        f"{summary['quantitative_hallucination_count']}"
    )
    print(
        "abstention_pass_count / abstention_cases: "
        f"{summary['abstention_pass_count']} / {summary['abstention_cases']}"
    )
    print(f"cases_with_invalid_sources: {summary['cases_with_invalid_sources']}")
    print(
        "cases_with_suspicious_quantitative_claims: "
        f"{summary['cases_with_suspicious_quantitative_claims']}"
    )
    print(
        "cases_with_abstention_failure_or_uncertainty: "
        f"{summary['cases_with_abstention_failure_or_uncertainty']}"
    )
