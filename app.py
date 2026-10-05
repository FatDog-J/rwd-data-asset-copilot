import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag import (  # noqa: E402
    TOP_K_EVIDENCE,
    _context_sources,
    build_context,
    build_prompt,
    call_llm,
    retrieve_evidence,
    warn_for_unknown_sources,
)


def _clean_items(items):
    return [item for item in items if isinstance(item, str) and item.strip()]


def _answer_question(question):
    evidence = retrieve_evidence(question, top_k=TOP_K_EVIDENCE)
    context = build_context(evidence)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt)
    warn_for_unknown_sources(answer, _context_sources(evidence))

    return answer, evidence


def _display_string_list(items, empty_message):
    cleaned_items = _clean_items(items)

    if not cleaned_items:
        st.write(empty_message)
        return

    for item in cleaned_items:
        st.write(f"- {item}")


def _display_evidence(evidence):
    with st.expander("View retrieved evidence"):
        for rank, result in enumerate(evidence, start=1):
            if rank > 1:
                st.divider()
            st.markdown(f"**Rank {rank}**")
            st.write(f"Source: {result['source']}")
            st.write(f"Section: {result['section']}")
            st.code(result["text"], language=None, wrap_lines=True)


def main():
    st.set_page_config(page_title="RWD Data Asset & Methodology Copilot")
    st.title("RWD Data Asset & Methodology Copilot")
    st.write(
        "Ask questions about data definitions, tables and fields, methodology, "
        "relationships, coverage, and documented limitations."
    )

    question = st.text_area("Question", height=120)
    ask_clicked = st.button("Ask")

    if not ask_clicked:
        return

    cleaned_question = question.strip()
    if not cleaned_question:
        st.error("Please enter a question before asking.")
        return

    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            answer, evidence = _answer_question(cleaned_question)
    except Exception as exc:
        message = str(exc)
        if "OPENAI_API_KEY" in message:
            st.error(
                "The OpenAI API key is not configured. Add it to the project .env "
                "file or environment before asking a question."
            )
        else:
            st.error(
                "Sorry, the copilot could not complete this request. Check the "
                "local setup and try again."
            )
        return

    st.subheader("Answer")
    st.write(answer.answer)

    st.subheader("Relevant tables / fields")
    _display_string_list(
        answer.relevant_tables_or_fields,
        "None specifically identified.",
    )

    st.subheader("Additional documented limitations")
    _display_string_list(
        answer.limitations,
        "No additional documented limitation was identified in the retrieved evidence.",
    )

    st.subheader("Sources")
    _display_string_list(answer.sources, "No sources were returned.")

    _display_evidence(evidence)


if __name__ == "__main__":
    main()
