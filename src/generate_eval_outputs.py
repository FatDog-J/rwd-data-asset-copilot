import json
from pathlib import Path

from rag import (
    _context_sources,
    build_context,
    build_prompt,
    call_llm,
    retrieve_evidence,
    warn_for_unknown_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "evals" / "generation_eval.json"
OUTPUTS_PATH = PROJECT_ROOT / "evals" / "generation_outputs.json"


def load_generation_eval():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def generate_case_output(case):
    evidence = retrieve_evidence(case["question"], top_k=5)
    context = build_context(evidence)
    prompt = build_prompt(case["question"], context)
    answer = call_llm(prompt)
    warn_for_unknown_sources(answer, _context_sources(evidence))

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answerable": case["answerable"],
        "required_points": case["required_points"],
        "forbidden_behavior": case["forbidden_behavior"],
        "retrieved_evidence": [
            {
                "source": result["source"],
                "section": result["section"],
                "text": result["text"],
            }
            for result in evidence
        ],
        "generated_answer": answer.model_dump(),
    }


if __name__ == "__main__":
    cases = load_generation_eval()
    outputs = []

    for case in cases:
        output = generate_case_output(case)
        outputs.append(output)
        print(f"generated: {output['id']}")

    OUTPUTS_PATH.write_text(
        json.dumps(
            {
                "cases": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved: {OUTPUTS_PATH}")
    print(f"total_cases: {len(outputs)}")
