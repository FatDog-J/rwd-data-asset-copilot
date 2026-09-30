from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieve import (
    build_index,
    load_processed_chunks,
    search as lexical_search,
)
from vector_retrieve import MODEL_NAME, build_vector_index, load_eval_questions, vector_search


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def chunk_key(result):
    return result["source"], result["section"], result["text"]


def combine_candidates(lexical_results, vector_results):
    candidates = {}

    for result in lexical_results + vector_results:
        key = chunk_key(result)
        if key not in candidates:
            candidates[key] = {
                "source": result["source"],
                "section": result["section"],
                "text": result["text"],
            }

    return list(candidates.values())


def candidate_text(candidate):
    return f"{candidate['section']}\n{candidate['text']}"


def rerank_search(
    query,
    lexical_index,
    vector_index,
    embedding_model,
    reranker,
    top_k=5,
    candidate_k=20,
):
    lexical_results = lexical_search(query, lexical_index, top_k=candidate_k)
    vector_results = vector_search(query, vector_index, embedding_model, top_k=candidate_k)
    candidates = combine_candidates(lexical_results, vector_results)

    pairs = [
        [query, candidate_text(candidate)]
        for candidate in candidates
    ]
    scores = reranker.predict(pairs)

    reranked_results = []
    for candidate, score in zip(candidates, scores):
        reranked_results.append(
            {
                "source": candidate["source"],
                "section": candidate["section"],
                "text": candidate["text"],
                "reranker_score": float(score),
            }
        )

    return sorted(
        reranked_results,
        key=lambda result: result["reranker_score"],
        reverse=True,
    )[:top_k]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    chunks = load_processed_chunks()
    lexical_index = build_index(chunks)

    embedding_model = SentenceTransformer(MODEL_NAME)
    vector_index = build_vector_index(chunks, embedding_model)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    eval_questions = load_eval_questions(eval_path)
    questions_by_id = {
        question["id"]: question["question"]
        for question in eval_questions
    }

    for question_id in ("Q1", "Q9"):
        query = questions_by_id[question_id]
        print(f"\n{question_id}: {query}")
        results = rerank_search(
            query,
            lexical_index,
            vector_index,
            embedding_model,
            reranker,
            top_k=5,
            candidate_k=20,
        )

        for rank, result in enumerate(results, start=1):
            print(f"Rank: {rank}")
            print(f"source={result['source']}")
            print(f"section={result['section']}")
            print(f"reranker_score={result['reranker_score']:.4f}")
