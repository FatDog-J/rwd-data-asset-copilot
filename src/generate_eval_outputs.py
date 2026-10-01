import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=EVAL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUTS_PATH)
    return parser.parse_args()


def load_generation_eval(spec_path):
    return json.loads(spec_path.read_text(encoding="utf-8"))


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
    args = parse_args()
    cases = load_generation_eval(args.spec)
    outputs = []

    for case in cases:
        output = generate_case_output(case)
        outputs.append(output)
        print(f"generated: {output['id']}")

    args.output.write_text(
        json.dumps(
            {
                "cases": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved: {args.output}")
    print(f"total_cases: {len(outputs)}")
