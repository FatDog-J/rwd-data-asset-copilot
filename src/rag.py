import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from models import RWDAnswer
from rerank_retrieve import RERANKER_MODEL_NAME, rerank_search
from retrieve import build_index, load_processed_chunks
from vector_retrieve import MODEL_NAME as EMBEDDING_MODEL_NAME, build_vector_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
OPENAI_MODEL = "gpt-4.1-mini"
TOP_K_EVIDENCE = 5
CANDIDATE_K = 20


def build_context(results):
    context_blocks = []

    for result in results:
        context_blocks.append(
            "\n".join(
                [
                    f"[SOURCE: {result['source']}]",
                    f"[SECTION: {result['section']}]",
                    "",
                    result["text"],
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def build_prompt(question, context):
    return f"""
You are an analyst-facing RWD Data Asset & Methodology Copilot for Healthcare Intelligence work.

Instructions:
- Answer only from the provided evidence.
- Do not use outside knowledge.
- Do not invent tables, fields, definitions, methodology, or limitations.
- If the evidence is insufficient, explicitly say that the available evidence is insufficient to answer the question.
- Only list source filenames that actually appear in the provided context.
- Distinguish explicitly documented limitations from inference.
- Do not turn an implication into a documented fact.
- For the limitations field:
  - Include only limitations explicitly stated in the retrieved evidence.
  - Do not create limitations based on missing information.
  - Statements such as "the documentation does not specify..." must not be treated as limitations unless the source explicitly states this.
  - If no relevant documented limitation is present, return [].
- For relevant_tables_or_fields:
  - Include only tables or fields directly relevant to the question.
  - Include only tables or fields explicitly supported by the retrieved evidence.
  - Include only tables or fields necessary to directly answer the question.
  - Do not add related modules or fields simply because they appear elsewhere in the context.
  - Do not expand generic wording such as "relevant modules" into specific modules unless the retrieved evidence explicitly identifies those modules for the task being asked.
  - For methodology questions where no specific table or field is required, return [].
- For sources:
  - Include only sources that directly support claims used in the final answer.
  - Do not list a source merely because it appears in the retrieved context.
  - Use the minimum sufficient set of sources.
- Do not infer that a link, field, mechanism, capability, or method is absent unless the retrieved evidence explicitly states that it is absent.
- Prefer omission over unsupported inference.
- Keep the answer concise and useful for an RWD / Healthcare Intelligence analyst.

Analyst question:
{question}

Retrieved evidence:
{context}
""".strip()


def _load_openai_environment():
    load_dotenv(ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Expected it in the project-root "
            f".env file at {ENV_PATH}, or in the current environment."
        )


def call_llm(prompt):
    _load_openai_environment()
    client = OpenAI()

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=prompt,
        text_format=RWDAnswer,
    )

    return response.output_parsed


@lru_cache(maxsize=1)
def _retrieval_components():
    chunks = load_processed_chunks()
    lexical_index = build_index(chunks)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_index = build_vector_index(chunks, embedding_model)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return lexical_index, vector_index, embedding_model, reranker


def retrieve_evidence(question, top_k=TOP_K_EVIDENCE):
    lexical_index, vector_index, embedding_model, reranker = _retrieval_components()

    return rerank_search(
        question,
        lexical_index,
        vector_index,
        embedding_model,
        reranker,
        top_k=top_k,
        candidate_k=CANDIDATE_K,
    )


def _context_sources(results):
    return {result["source"] for result in results}


def warn_for_unknown_sources(answer, context_sources):
    unknown_sources = [
        source
        for source in answer.sources
        if source not in context_sources
    ]

    if unknown_sources:
        print(f"WARNING: answer cited sources not present in retrieved context: {unknown_sources}")


def rag_answer(question):
    evidence = retrieve_evidence(question, top_k=TOP_K_EVIDENCE)
    context = build_context(evidence)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt)
    warn_for_unknown_sources(answer, _context_sources(evidence))

    return answer


def _print_answer(answer):
    print(f"- answer: {answer.answer}")
    print(f"- relevant_tables_or_fields: {answer.relevant_tables_or_fields}")
    print(f"- limitations: {answer.limitations}")
    print(f"- sources: {answer.sources}")


if __name__ == "__main__":
    test_questions = [
        # "Which field tells me when a medication was administered?",
        # (
        #     "Which tables or fields can help distinguish a prescribed medication "
        #     "from a medication that was actually administered?"
        # ),
        # "What steps does the documentation suggest for longitudinal analysis?",
        # "Are medication administration records available consistently across the entire MIMIC-IV study period?",
        "What proportion of patients lack eMAR records in each year of the MIMIC-IV study period?",
    ]

    for question in test_questions:
        evidence = retrieve_evidence(question, top_k=TOP_K_EVIDENCE)
        context = build_context(evidence)
        prompt = build_prompt(question, context)
        answer = call_llm(prompt)
        warn_for_unknown_sources(answer, _context_sources(evidence))

        print("QUESTION:")
        print(question)
        print()
        print("RETRIEVED EVIDENCE:")
        for result in evidence:
            print(f"- source: {result['source']}")
            print(f"  section: {result['section']}")
        print()
        print("STRUCTURED ANSWER:")
        _print_answer(answer)
        print()
