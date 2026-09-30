from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieve import (
    build_index,
    load_processed_chunks,
    search as lexical_search,
)
from vector_retrieve import (
    MODEL_NAME,
    build_vector_index,
    load_eval_questions,
    vector_search,
)


def reciprocal_rank_fusion(lexical_results, vector_results, rrf_k=60):
    fused_results = {}

    for results in (lexical_results, vector_results):
        for rank, result in enumerate(results, start=1):
            chunk_key = (
                result["source"],
                result["section"],
                result["text"],
            )
            contribution = 1 / (rrf_k + rank)

            if chunk_key not in fused_results:
                fused_results[chunk_key] = {
                    "source": result["source"],
                    "section": result["section"],
                    "text": result["text"],
                    "hybrid_score": 0.0,
                }

            fused_results[chunk_key]["hybrid_score"] += contribution

    return sorted(
        fused_results.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )


def hybrid_search(
    query,
    lexical_index,
    vector_index,
    model,
    top_k=5,
    candidate_k=10,
):
    lexical_results = lexical_search(query, lexical_index, top_k=candidate_k)
    vector_results = vector_search(query, vector_index, model, top_k=candidate_k)
    fused_results = reciprocal_rank_fusion(lexical_results, vector_results)

    return fused_results[:top_k]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    chunks = load_processed_chunks()
    lexical_index = build_index(chunks)

    model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(chunks, model)

    eval_questions = load_eval_questions(eval_path)

    for eval_item in eval_questions:
        print(f"\n{eval_item['id']}: {eval_item['question']}")
        results = hybrid_search(
            eval_item["question"],
            lexical_index,
            vector_index,
            model,
            top_k=5,
            candidate_k=10,
        )

        for rank, result in enumerate(results, start=1):
            print(f"Rank: {rank}")
            print(f"source={result['source']}")
            print(f"section={result['section']}")
            print(f"hybrid_score={result['hybrid_score']:.6f}")
