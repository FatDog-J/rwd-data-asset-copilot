from pathlib import Path

import minsearch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ingest import (
    chunk_document,
    clean_document,
    load_markdown_files,
    resolve_includes,
)


TEXT_FIELDS = ["section", "text"]
KEYWORD_FIELDS = ["source", "section"]
BOOSTS = {"section": 2.0, "text": 1.0}


def _safe_print_text(text):
    return text.encode("ascii", errors="replace").decode("ascii")


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


def _is_link_section(section):
    section = section.strip()
    return section == "Links to" or section.endswith("> Links to")


def filter_indexable_chunks(chunks):
    return [
        {
            "source": chunk["source"],
            "section": chunk["section"],
            "text": chunk["text"],
        }
        for chunk in chunks
        if not _is_link_section(chunk["section"])
    ]


def build_index(chunks):
    searchable_chunks = filter_indexable_chunks(chunks)

    index = minsearch.Index(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
    )
    index.fit(searchable_chunks)

    return index


def _score_result(query, index, result, boost_dict):
    result_id = result["_id"]
    score = 0.0

    for field in index.text_fields:
        query_vector = index.vectorizers[field].transform([query])
        field_scores = cosine_similarity(
            query_vector,
            index.text_matrices[field],
        ).flatten()
        score += field_scores[result_id] * boost_dict.get(field, 1.0)

    return float(score)


def search(query, index, top_k=5):
    results = index.search(
        query,
        boost_dict=BOOSTS,
        num_results=top_k,
        output_ids=True,
    )

    scored_results = []
    for result in results:
        scored_result = {
            "source": result["source"],
            "section": result["section"],
            "text": result["text"],
            "score": _score_result(query, index, result, BOOSTS),
        }
        scored_results.append(scored_result)

    return scored_results


if __name__ == "__main__":
    questions = [
        "Which fields indicate when a patient was admitted to and discharged from the hospital?",
        "Which tables or fields can help distinguish a prescribed medication from a medication that was actually administered?",
        "Which field tells me when a medication was administered?",
    ]

    chunks = load_processed_chunks()
    index = build_index(chunks)

    print(f"Total processed chunks: {len(chunks)}")
    print(f"Indexed chunks: {len(index.docs)}")

    for question in questions:
        print(f"\nQuestion: {question}")
        results = search(question, index, top_k=5)

        for rank, result in enumerate(results, start=1):
            preview = _safe_print_text(result["text"][:200].replace("\n", " "))
            print(f"Rank: {rank}")
            print(f"source={result['source']}")
            print(f"section={_safe_print_text(result['section'])}")
            print(f"score={result['score']:.4f}")
            print(f"preview={preview}")
