import sys
import time
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
from observability import (  # noqa: E402
    FEEDBACK_LOG_PATH,
    append_jsonl_event,
    build_feedback_event,
    build_runtime_event,
    create_request_id,
)


RESULT_STATE_KEYS = (
    "request_id",
    "answer",
    "evidence",
    "feedback_submitted",
    "submitted_rating",
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


def _append_runtime_event_safely(event):
    try:
        append_jsonl_event(event)
    except Exception:
        pass


def _clear_result_state(state):
    for key in RESULT_STATE_KEYS:
        state.pop(key, None)


def _store_successful_result(state, request_id, answer, evidence):
    state["request_id"] = request_id
    state["answer"] = answer
    state["evidence"] = evidence
    state["feedback_submitted"] = False
    state["submitted_rating"] = None


def _has_stored_result(state):
    return (
        state.get("request_id") is not None
        and state.get("answer") is not None
        and state.get("evidence") is not None
    )


def _submit_feedback(state, rating, append_event=append_jsonl_event):
    if state.get("feedback_submitted"):
        return True

    event = build_feedback_event(state["request_id"], rating)
    append_event(event, FEEDBACK_LOG_PATH)
    state["feedback_submitted"] = True
    state["submitted_rating"] = rating

    return True


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


def _display_feedback_controls():
    st.write("Was this answer helpful?")

    if st.session_state.get("feedback_submitted"):
        st.info("Thanks for your feedback.")
        return

    col_helpful, col_not_helpful = st.columns(2)
    with col_helpful:
        helpful_clicked = st.button("Helpful")
    with col_not_helpful:
        not_helpful_clicked = st.button("Not helpful")

    rating = None
    if helpful_clicked:
        rating = "helpful"
    elif not_helpful_clicked:
        rating = "not_helpful"

    if rating is None:
        return

    try:
        _submit_feedback(st.session_state, rating)
    except Exception:
        st.error("Feedback could not be saved.")
        return

    st.info("Thanks for your feedback.")


def _display_result(answer, evidence):
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
    _display_feedback_controls()


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
        if _has_stored_result(st.session_state):
            _display_result(
                st.session_state["answer"],
                st.session_state["evidence"],
            )
        return

    cleaned_question = question.strip()
    if not cleaned_question:
        st.error("Please enter a question before asking.")
        return

    _clear_result_state(st.session_state)
    request_id = create_request_id()
    start_time = time.monotonic()

    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            answer, evidence = _answer_question(cleaned_question)
    except Exception as exc:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        event = build_runtime_event(
            request_id=request_id,
            status="error",
            latency_ms=latency_ms,
            question=cleaned_question,
            error=exc,
        )
        _append_runtime_event_safely(event)

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

    latency_ms = int((time.monotonic() - start_time) * 1000)
    event = build_runtime_event(
        request_id=request_id,
        status="success",
        latency_ms=latency_ms,
        question=cleaned_question,
        evidence=evidence,
        answer=answer,
    )
    _append_runtime_event_safely(event)
    _store_successful_result(st.session_state, request_id, answer, evidence)

    _display_result(answer, evidence)


if __name__ == "__main__":
    main()
