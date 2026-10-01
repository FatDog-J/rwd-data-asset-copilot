import argparse
import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from models import RWDAnswer
from rag import (
    OPENAI_MODEL,
    _load_openai_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = PROJECT_ROOT / "evals" / "generation_outputs.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "generation_judge_results.json"
JUDGE_MODEL = OPENAI_MODEL


class GenerationJudgeResult(BaseModel):
    correctness: int
    groundedness: int
    completeness: int
    source_faithfulness: int
    abstention: int | None
    pass_overall: bool
    issues: list[str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def load_generation_outputs(input_path):
    return json.loads(input_path.read_text(encoding="utf-8"))["cases"]


def build_judge_prompt(case, evidence, answer):
    evidence_blocks = []
    for result in evidence:
        evidence_blocks.append(
            "\n".join(
                [
                    f"[SOURCE: {result['source']}]",
                    f"[SECTION: {result['section']}]",
                    "",
                    result["text"],
                ]
            )
        )

    judge_input = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "required_points": case["required_points"],
        "forbidden_behavior": case["forbidden_behavior"],
        "category": case["category"],
        "retrieved_evidence": "\n\n---\n\n".join(evidence_blocks),
        "generated_rwd_answer": answer.model_dump(),
    }

    return f"""
You are a semantic evaluator for a grounded RWD / Healthcare Intelligence copilot.

Use only the supplied retrieved evidence.
Do not use external knowledge about MIMIC-IV.
Do not reward verbosity.
Do not require exact wording from required_points.
Treat required_points as semantic expectations.
Distinguish "documentation does not provide X" from "the dataset itself cannot support X".
Be conservative when deciding whether a limitation is explicitly documented.
Prefer identifying a specific issue over vague criticism.

Evaluate the generated RWDAnswer against the retrieved evidence and evaluation specification.

Scoring:
- correctness, groundedness, completeness, source_faithfulness use 0, 1, or 2.
- abstention is null when answerable is true; otherwise use 0, 1, or 2.

CORRECTNESS:
2 = The core answer is correct and directly answers the question.
1 = The core answer is substantially correct but contains a minor error, ambiguity, or imprecision that does not overturn the main conclusion.
0 = The core answer is wrong, misleading, or fails to answer the question.

GROUNDEDNESS:
Evaluate all substantive claims in answer, relevant_tables_or_fields, and limitations.
2 = All substantive claims are explicitly supported by retrieved evidence. No unsupported inference is presented as fact.
1 = The core answer is supported, but there is a minor unsupported expansion, overstatement, or inferred statement.
0 = There is a major unsupported claim, fabricated fact, or conclusion that the retrieved evidence does not support.
Absence of information in the retrieved documentation must not automatically be treated as a documented dataset limitation.
"The documentation does not give a percentage" may be valid about available evidence.
But an inferred absence such as "the dataset lacks the capability to..." is not grounded unless evidence explicitly states it.

COMPLETENESS:
Judge against required_points semantically, not by literal string matching.
Required points are semantic concepts, not independent sentences that must each be repeated separately.
Multiple required points may be satisfied by the same phrase or sentence.
A concept counts as covered if it is clearly expressed anywhere in answer, relevant_tables_or_fields, or limitations, when appropriate to the question.
Do not penalize concise answers merely because they combine concepts.
Example: if required_points include "charttime" and "emar", then "`charttime` in the `emar` table records medication administration time" satisfies both points.
Continue to judge semantically rather than by literal matching.
2 = All important required points are covered.
1 = The answer captures the main conclusion but misses one meaningful required point or supporting element.
0 = Important required information is substantially missing.

SOURCE_FAITHFULNESS:
2 = Listed sources are a minimal or reasonable set of retrieved sources that genuinely support the final answer.
1 = Sources support the answer overall, but include an unnecessary source or omit a useful supporting source without making the answer misleading.
0 = A cited source does not support the answer, or the answer materially relies on evidence from a source that is not cited.
Do not penalize the model merely for not citing every retrieved source.

ABSTENTION:
If answerable is true, return null.
If answerable is false:
2 = The answer clearly states that the requested specific result cannot be determined or quantified from the retrieved evidence, does not invent the unsupported result, and appropriately explains what additional analysis or data would be needed when relevant.
1 = The answer avoids fabrication and substantially abstains, but the limitation of the available evidence or next analytical requirement is not fully explained.
0 = The answer invents, estimates, or asserts the unsupported requested result.

FORBIDDEN BEHAVIOR:
Use forbidden_behavior as additional constraints.
If a forbidden behavior occurs, identify it explicitly in issues and reduce the relevant score.
Do not mechanically set all dimensions to zero for one violation.

PASS_OVERALL:
Set pass_overall to true only if correctness >= 1, groundedness >= 1,
completeness >= 1, source_faithfulness >= 1, and, for answerable == false,
abstention >= 1. Otherwise false.

Evaluation input:
{json.dumps(judge_input, indent=2)}
""".strip()


def call_judge(prompt):
    _load_openai_environment()
    client = OpenAI()
    response = client.responses.parse(
        model=JUDGE_MODEL,
        input=prompt,
        text_format=GenerationJudgeResult,
    )

    return response.output_parsed


def validate_answer(answer_data):
    return RWDAnswer.model_validate(answer_data)


def evaluate_case(frozen_case):
    evidence = frozen_case["retrieved_evidence"]
    answer = validate_answer(frozen_case["generated_answer"])
    judge_prompt = build_judge_prompt(frozen_case, evidence, answer)
    judge_result = call_judge(judge_prompt)

    return {
        "id": frozen_case["id"],
        "category": frozen_case["category"],
        "question": frozen_case["question"],
        "answerable": frozen_case["answerable"],
        "retrieved_evidence": evidence,
        "generated_answer": answer.model_dump(),
        "judge_result": judge_result.model_dump(),
    }


def print_case_report(result):
    judge = result["judge_result"]
    answer = result["generated_answer"]

    print(f"ID: {result['id']}")
    print(f"Category: {result['category']}")
    print(f"Question: {result['question']}")
    print()
    print("Generated answer summary")
    print(f"- answer: {answer['answer']}")
    print(f"- relevant_tables_or_fields: {answer['relevant_tables_or_fields']}")
    print(f"- limitations: {answer['limitations']}")
    print(f"- sources: {answer['sources']}")
    print()
    print("Judge scores:")
    print(f"- correctness: {judge['correctness']}")
    print(f"- groundedness: {judge['groundedness']}")
    print(f"- completeness: {judge['completeness']}")
    print(f"- source_faithfulness: {judge['source_faithfulness']}")
    print(f"- abstention: {judge['abstention']}")
    print(f"- pass_overall: {judge['pass_overall']}")
    print()
    print("Issues:")
    if judge["issues"]:
        for issue in judge["issues"]:
            print(f"- {issue}")
    else:
        print("- none")
    print()


def average(values):
    return sum(values) / len(values) if values else 0


def summarize(results):
    judge_results = [result["judge_result"] for result in results]
    abstention_scores = [
        judge["abstention"]
        for judge in judge_results
        if judge["abstention"] is not None
    ]

    return {
        "total_cases": len(results),
        "overall_pass_count": sum(judge["pass_overall"] for judge in judge_results),
        "average_correctness": average([judge["correctness"] for judge in judge_results]),
        "average_groundedness": average([judge["groundedness"] for judge in judge_results]),
        "average_completeness": average([judge["completeness"] for judge in judge_results]),
        "average_source_faithfulness": average(
            [judge["source_faithfulness"] for judge in judge_results]
        ),
        "average_abstention": average(abstention_scores),
        "failed_cases": [
            result["id"]
            for result in results
            if not result["judge_result"]["pass_overall"]
        ],
        "cases_with_groundedness_below_2": [
            result["id"]
            for result in results
            if result["judge_result"]["groundedness"] < 2
        ],
        "cases_with_completeness_below_2": [
            result["id"]
            for result in results
            if result["judge_result"]["completeness"] < 2
        ],
        "cases_with_source_faithfulness_below_2": [
            result["id"]
            for result in results
            if result["judge_result"]["source_faithfulness"] < 2
        ],
        "abstention_cases_below_2": [
            result["id"]
            for result in results
            if result["judge_result"]["abstention"] is not None
            and result["judge_result"]["abstention"] < 2
        ],
    }


def print_summary(summary):
    print("SUMMARY")
    print(f"total_cases: {summary['total_cases']}")
    print(f"overall_pass_count: {summary['overall_pass_count']}")
    print(f"average_correctness: {summary['average_correctness']:.2f}")
    print(f"average_groundedness: {summary['average_groundedness']:.2f}")
    print(f"average_completeness: {summary['average_completeness']:.2f}")
    print(
        "average_source_faithfulness: "
        f"{summary['average_source_faithfulness']:.2f}"
    )
    print(f"average_abstention: {summary['average_abstention']:.2f}")
    print(f"failed_cases: {summary['failed_cases']}")
    print(
        "cases_with_groundedness_below_2: "
        f"{summary['cases_with_groundedness_below_2']}"
    )
    print(
        "cases_with_completeness_below_2: "
        f"{summary['cases_with_completeness_below_2']}"
    )
    print(
        "cases_with_source_faithfulness_below_2: "
        f"{summary['cases_with_source_faithfulness_below_2']}"
    )
    print(f"abstention_cases_below_2: {summary['abstention_cases_below_2']}")


if __name__ == "__main__":
    args = parse_args()
    cases = load_generation_outputs(args.input)
    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)
        print_case_report(result)

    summary = summarize(results)
    args.output.write_text(
        json.dumps(
            {
                "results": results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
