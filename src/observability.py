import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LOG_PATH = PROJECT_ROOT / "logs" / "runtime.jsonl"
FEEDBACK_LOG_PATH = PROJECT_ROOT / "logs" / "feedback.jsonl"
RUNTIME_EVENT_FIELDS = (
    "request_id",
    "timestamp",
    "status",
    "latency_ms",
    "question_length",
    "retrieved_count",
    "retrieved_sources",
    "answer_has_limitations",
    "error_type",
)
FEEDBACK_EVENT_FIELDS = (
    "request_id",
    "timestamp",
    "rating",
)
SUPPORTED_FEEDBACK_RATINGS = {"helpful", "not_helpful"}


def create_request_id():
    return str(uuid.uuid4())


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _unique_sources(evidence):
    sources = {
        item.get("source")
        for item in evidence
        if isinstance(item, dict) and item.get("source")
    }
    return sorted(sources)


def _has_limitations(answer):
    if answer is None:
        return False

    limitations = getattr(answer, "limitations", [])
    return any(
        isinstance(item, str) and item.strip()
        for item in limitations
    )


def build_runtime_event(
    request_id,
    status,
    latency_ms,
    question,
    evidence=None,
    answer=None,
    error=None,
):
    evidence = evidence or []
    event = {
        "request_id": request_id,
        "timestamp": _utc_timestamp(),
        "status": status,
        "latency_ms": int(latency_ms),
        "question_length": len(question),
        "retrieved_count": len(evidence),
        "retrieved_sources": _unique_sources(evidence),
        "answer_has_limitations": _has_limitations(answer) if status == "success" else False,
        "error_type": None if error is None else error.__class__.__name__,
    }

    return {field: event[field] for field in RUNTIME_EVENT_FIELDS}


def build_feedback_event(request_id, rating):
    if rating not in SUPPORTED_FEEDBACK_RATINGS:
        raise ValueError(f"Unsupported feedback rating: {rating}")

    event = {
        "request_id": request_id,
        "timestamp": _utc_timestamp(),
        "rating": rating,
    }

    return {field: event[field] for field in FEEDBACK_EVENT_FIELDS}


def append_jsonl_event(event, path=RUNTIME_LOG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, ensure_ascii=True) + "\n")
