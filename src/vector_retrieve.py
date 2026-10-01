import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import (
    chunk_document,
    clean_document,
    load_markdown_files,
    resolve_includes,
)
from retrieve import filter_indexable_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def load_processed_chunks():
    project_root = Path(__file__).resolve().parents[1]
    corpus_path = project_root / "data" / "raw" / "mimic_docs"
    include_path = project_root / "data" / "support" / "mimic_includes"

    documents = load_markdown_files(corpus_path)
    resolved_documents = [
        resolve_includes(document, include_path)
        for document in documents
    ]
    cleaned_documents = [clean_document(document) for document in resolved_documents]

    return [
        chunk
        for document in cleaned_documents
        for chunk in chunk_document(document)
    ]


def chunk_to_search_text(chunk):
    return f"{chunk['section']}\n{chunk['text']}"


def build_vector_index(chunks, model):
    indexed_chunks = filter_indexable_chunks(chunks)
    search_texts = [chunk_to_search_text(chunk) for chunk in indexed_chunks]
    embeddings = model.encode(
        search_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return {
        "chunks": indexed_chunks,
        "embeddings": np.asarray(embeddings),
    }


def vector_search(query, index, model, top_k=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = index["embeddings"] @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for result_index in top_indices:
        chunk = index["chunks"][result_index]
        results.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": float(scores[result_index]),
            }
        )

    return results


def load_eval_questions(eval_path):
    return json.loads(Path(eval_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "evals" / "retrieval_eval.json"

    chunks = load_processed_chunks()
    model = SentenceTransformer(MODEL_NAME)
    index = build_vector_index(chunks, model)
    eval_questions = load_eval_questions(eval_path)

    print(f"Model: {MODEL_NAME}")
    print(f"Indexed chunks: {len(index['chunks'])}")

    for eval_item in eval_questions:
        print(f"\n{eval_item['id']}: {eval_item['question']}")
        results = vector_search(eval_item["question"], index, model, top_k=5)

        for rank, result in enumerate(results, start=1):
            print(f"Rank: {rank}")
            print(f"source={result['source']}")
            print(f"section={result['section']}")
            print(f"score={result['score']:.4f}")
