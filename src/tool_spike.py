import argparse
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo" / "prescriptions_demo.csv"
OPENAI_MODEL = "gpt-4.1-mini"
MAX_TOOL_ITERATIONS = 3

ROUTING_INSTRUCTIONS = """
You are testing tool routing for an experimental RWD Data Asset & Methodology Copilot spike.

Use search_rwd_docs for questions about documentation, field meaning, methodology, definitions, or documented limitations.
Use run_aggregate when the user asks for a count or percentage that requires calculation from the demo records.
Do not answer quantitative demo-data questions from memory.
Do not call tools that are unnecessary.
Never substitute a supported table, field, value, or metric for a different one requested by the user.
A tool call must remain semantically faithful to the user's requested operation.
If no available tool supports the requested operation, acknowledge the capability limitation instead of calling a tool with different arguments.
Documentation lookup must not be used as a substitute for a quantitative result that requires unavailable record-level data.
The demo dataset is synthetic.
""".strip()

TOOLS = [
    {
        "type": "function",
        "name": "search_rwd_docs",
        "description": (
            "Retrieve the top 5 evidence snippets from the frozen RWD documentation "
            "retrieval pipeline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The documentation question or search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_aggregate",
        "description": (
            "Deterministically calculate a count or percentage ONLY for the "
            "synthetic prescriptions_demo.drug_type field, using the supported "
            "values MAIN, BASE, or ADDITIVE. Do not use this tool for other "
            "fields or values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "enum": ["prescriptions_demo"],
                },
                "field": {
                    "type": "string",
                    "enum": ["drug_type"],
                },
                "value": {
                    "type": "string",
                    "enum": ["MAIN", "BASE", "ADDITIVE"],
                },
                "metric": {
                    "type": "string",
                    "enum": ["count", "percentage"],
                },
            },
            "required": ["table", "field", "value", "metric"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

MANUAL_TESTS = [
    {
        "id": "T1",
        "question": "What does drug_type mean?",
        "required_tools": ["search_rwd_docs"],
        "expected_search_terms": ["drug_type"],
    },
    {
        "id": "T2",
        "question": "What percentage of prescription records have drug_type = MAIN?",
        "required_tools": ["run_aggregate"],
    },
    {
        "id": "T3",
        "question": "What caution does the documentation give about organ donor accounts?",
        "required_tools": ["search_rwd_docs"],
        "expected_search_terms": ["organ", "donor"],
    },
    {
        "id": "T4",
        "question": "How many prescription records in the demo data have drug_type = BASE?",
        "required_tools": ["run_aggregate"],
    },
    {
        "id": "T5",
        "question": (
            "What does drug_type mean, and what percentage of the demo prescription "
            "records are MAIN?"
        ),
        "required_tools": ["search_rwd_docs", "run_aggregate"],
        "expected_search_terms": ["drug_type"],
    },
]

UNSUPPORTED_CAPABILITY_TESTS = [
    {
        "id": "T6",
        "question": "What percentage of prescription records in the demo data use route = IV?",
        "requested_operation": {
            "table": "prescriptions_demo",
            "field": "route",
            "value": "IV",
            "metric": "percentage",
        },
    },
]

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "does",
    "for",
    "give",
    "have",
    "how",
    "in",
    "is",
    "many",
    "of",
    "records",
    "the",
    "what",
}

EXPECTED_AGGREGATE_ARGUMENTS = {
    "T2": {
        "table": "prescriptions_demo",
        "field": "drug_type",
        "value": "MAIN",
        "metric": "percentage",
    },
    "T4": {
        "table": "prescriptions_demo",
        "field": "drug_type",
        "value": "BASE",
        "metric": "count",
    },
    "T5": {
        "table": "prescriptions_demo",
        "field": "drug_type",
        "value": "MAIN",
        "metric": "percentage",
    },
}


def search_rwd_docs(query):
    evidence = retrieve_evidence(query, top_k=5)

    return [
        {
            "source": result["source"],
            "section": result["section"],
            "text": result["text"],
        }
        for result in evidence
    ]


def _load_demo_rows():
    with DEMO_DATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if reader.fieldnames != ["drug_type"]:
        raise ValueError(
            f"Expected {DEMO_DATA_PATH} to contain exactly one column: drug_type"
        )

    return rows


def run_aggregate(table, field, value, metric):
    if table != "prescriptions_demo":
        raise ValueError(f"Unsupported table: {table}")
    if field != "drug_type":
        raise ValueError(f"Unsupported field: {field}")
    if metric not in {"count", "percentage"}:
        raise ValueError(f"Unsupported metric: {metric}")

    rows = _load_demo_rows()
    values = [row[field] for row in rows]
    present_values = set(values)

    if value not in present_values:
        raise ValueError(f"Unsupported value for {field}: {value}")

    matching_rows = sum(1 for row_value in values if row_value == value)
    total_rows = len(rows)

    if metric == "count":
        result = matching_rows
    else:
        result = matching_rows / total_rows * 100

    return {
        "table": table,
        "field": field,
        "value": value,
        "metric": metric,
        "result": result,
        "matching_rows": matching_rows,
        "total_rows": total_rows,
        "synthetic_dataset": True,
    }


AVAILABLE_TOOLS = {
    "search_rwd_docs": search_rwd_docs,
    "run_aggregate": run_aggregate,
}


def _load_openai_environment():
    load_dotenv(ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Expected it in the project-root "
            f".env file at {ENV_PATH}, or in the current environment."
        )


def _response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    text_parts = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) in {"output_text", "text"}:
                    text_parts.append(getattr(content, "text", ""))

    return "\n".join(part for part in text_parts if part).strip()


def _function_calls(response):
    return [
        item
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "function_call"
    ]


def _parse_arguments(raw_arguments):
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return json.loads(raw_arguments or "{}")


def answer_with_tools(question, client=None, verbose=True):
    _load_openai_environment()
    client = client or OpenAI()

    initial_input = [
        {
            "role": "developer",
            "content": ROUTING_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    tool_events = []

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=initial_input,
        tools=TOOLS,
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        calls = _function_calls(response)
        if not calls:
            final_answer = _response_text(response)
            if verbose:
                _print_run_log(question, tool_events, final_answer)
            return {
                "question": question,
                "tool_events": tool_events,
                "final_answer": final_answer,
            }

        function_outputs = []

        for call in calls:
            tool_name = call.name
            tool_arguments = _parse_arguments(call.arguments)

            if tool_name not in AVAILABLE_TOOLS:
                tool_result = {"error": f"Unsupported tool: {tool_name}"}
            else:
                try:
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_arguments)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

            tool_events.append(
                {
                    "selected_tool": tool_name,
                    "tool_arguments": tool_arguments,
                    "tool_result": tool_result,
                }
            )
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(tool_result, ensure_ascii=True),
                }
            )

        response = client.responses.create(
            model=OPENAI_MODEL,
            previous_response_id=response.id,
            input=function_outputs,
            tools=TOOLS,
        )

    final_answer = (
        "Tool-call iteration limit reached before the model produced a final answer."
    )
    if verbose:
        _print_run_log(question, tool_events, final_answer)
    return {
        "question": question,
        "tool_events": tool_events,
        "final_answer": final_answer,
    }


def _print_run_log(question, tool_events, final_answer):
    print(f"user_question: {question}")
    for event in tool_events:
        print(f"selected_tool: {event['selected_tool']}")
        print(f"tool_arguments: {json.dumps(event['tool_arguments'], ensure_ascii=True)}")
        print(f"tool_result: {json.dumps(event['tool_result'], ensure_ascii=True)}")
    print(f"final_answer: {final_answer}")


def _tool_call_trace(tool_events):
    return [
        {
            "tool": event["selected_tool"],
            "arguments": event["tool_arguments"],
            "result": event["tool_result"],
        }
        for event in tool_events
    ]


def _content_tokens(text):
    normalized = "".join(
        character.lower() if character.isalnum() or character == "_" else " "
        for character in text
    )
    return {
        token
        for token in normalized.split()
        if token and token not in STOPWORDS
    }


def _search_arguments_valid(test, arguments):
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return False

    query_tokens = _content_tokens(query)
    expected_terms = set(test.get("expected_search_terms", []))

    if expected_terms:
        return all(
            term in query_tokens
            or _content_tokens(term.replace("_", " ")).issubset(query_tokens)
            for term in expected_terms
        )

    question_tokens = _content_tokens(test["question"])
    return bool(question_tokens & query_tokens)


def _aggregate_arguments_valid(test_id, arguments):
    return arguments == EXPECTED_AGGREGATE_ARGUMENTS.get(test_id)


def _call_arguments_valid(test, call):
    if call["tool"] == "run_aggregate":
        return _aggregate_arguments_valid(test["id"], call["arguments"])

    if call["tool"] == "search_rwd_docs":
        return _search_arguments_valid(test, call["arguments"])

    return False


def _evaluate_trace(test, trace):
    required_tools = _required_tools(test)
    matched_indexes = []

    for required_tool in required_tools:
        matching_index = next(
            (
                index
                for index, call in enumerate(trace)
                if index not in matched_indexes and call["tool"] == required_tool
            ),
            None,
        )
        if matching_index is not None:
            matched_indexes.append(matching_index)

    required_tools_present = len(matched_indexes) == len(required_tools)
    required_calls = [trace[index] for index in matched_indexes]

    if not required_tools_present:
        arguments_valid = False
    else:
        arguments_valid = all(
            _call_arguments_valid(test, call)
            for call in required_calls
        )

    unnecessary_tool_calls = [
        call
        for index, call in enumerate(trace)
        if index not in matched_indexes
    ]
    routing_pass = (
        required_tools_present
        and arguments_valid
        and not unnecessary_tool_calls
    )

    return {
        "required_tools_present": required_tools_present,
        "arguments_valid": arguments_valid,
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "routing_pass": routing_pass,
    }


def _tool_error_observed(trace):
    return any(
        isinstance(call["result"], dict)
        and call["result"].get("error")
        for call in trace
    )


def _unsupported_capability_acknowledged(final_answer):
    answer = final_answer.lower()
    has_limitation_language = any(
        phrase in answer
        for phrase in (
            "cannot",
            "can't",
            "does not support",
            "do not support",
            "not supported",
            "unable",
            "not available",
        )
    )
    mentions_requested_field = "route" in answer and "iv" in answer
    mentions_demo_scope = any(
        term in answer
        for term in ("demo", "data", "dataset", "aggregation", "capability", "tool")
    )

    return has_limitation_language and mentions_requested_field and mentions_demo_scope


def _unsupported_quantitative_result_invented(final_answer):
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*%", final_answer))


def _tool_schema(tool_name):
    return next((tool for tool in TOOLS if tool["name"] == tool_name), None)


def _schema_valid_tool_call(call):
    schema = _tool_schema(call["tool"])
    if schema is None:
        return False

    parameters = schema["parameters"]
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        return False

    required = set(parameters.get("required", []))
    properties = parameters.get("properties", {})
    argument_keys = set(arguments)

    if not required.issubset(argument_keys):
        return False

    if not parameters.get("additionalProperties", True):
        if argument_keys - set(properties):
            return False

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if property_schema is None:
            continue
        if property_schema.get("type") == "string" and not isinstance(value, str):
            return False
        if "enum" in property_schema and value not in property_schema["enum"]:
            return False

    return True


def _aggregate_request_aligned(call, requested_operation):
    arguments = call["arguments"]
    return all(
        arguments.get(key) == requested_value
        for key, requested_value in requested_operation.items()
    )


def _search_request_aligned(call, requested_operation):
    query = call["arguments"].get("query", "")
    if not isinstance(query, str) or not query.strip():
        return False

    query_tokens = _content_tokens(query)
    requested_terms = [
        value
        for value in requested_operation.values()
        if isinstance(value, str)
    ]

    return any(
        term.lower() in query.lower()
        or _content_tokens(term.replace("_", " ")).issubset(query_tokens)
        for term in requested_terms
    )


def _request_aligned_tool_call(test, call):
    requested_operation = test.get("requested_operation", {})

    if call["tool"] == "run_aggregate":
        return _aggregate_request_aligned(call, requested_operation)

    if call["tool"] == "search_rwd_docs":
        return _search_request_aligned(call, requested_operation)

    return False


def _annotate_unsupported_tool_calls(test, trace):
    return [
        {
            **call,
            "schema_valid": _schema_valid_tool_call(call),
            "request_aligned": _request_aligned_tool_call(test, call),
        }
        for call in trace
    ]


def _redundant_tool_calls(trace):
    seen_calls = set()
    redundant_calls = []

    for call in trace:
        call_key = json.dumps(
            {
                "tool": call["tool"],
                "arguments": call["arguments"],
            },
            sort_keys=True,
        )
        if call_key in seen_calls:
            redundant_calls.append(call)
        else:
            seen_calls.add(call_key)

    return redundant_calls


def _evaluate_unsupported_capability(test, final_answer, trace):
    annotated_trace = _annotate_unsupported_tool_calls(test, trace)
    request_misaligned_tool_calls = [
        call
        for call in annotated_trace
        if not call["request_aligned"]
    ]
    redundant_tool_calls = _redundant_tool_calls(annotated_trace)
    unnecessary_tool_calls = []
    for call in request_misaligned_tool_calls + redundant_tool_calls:
        if call not in unnecessary_tool_calls:
            unnecessary_tool_calls.append(call)

    tool_error_observed = _tool_error_observed(trace)
    unsupported_acknowledged = _unsupported_capability_acknowledged(final_answer)
    quantitative_result_invented = _unsupported_quantitative_result_invented(
        final_answer
    )
    graceful_failure_pass = (
        not quantitative_result_invented
        and unsupported_acknowledged
        and not request_misaligned_tool_calls
        and not redundant_tool_calls
    )

    return {
        "annotated_trace": annotated_trace,
        "tool_error_observed": tool_error_observed,
        "request_misaligned_tool_calls": request_misaligned_tool_calls,
        "redundant_tool_calls": redundant_tool_calls,
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "unsupported_capability_acknowledged": unsupported_acknowledged,
        "unsupported_quantitative_result_invented": quantitative_result_invented,
        "graceful_failure_pass": graceful_failure_pass,
    }


def _required_tools(test):
    if "required_tools" in test:
        return test["required_tools"]
    if "expected_tool" in test:
        return [test["expected_tool"]]
    return []


def run_manual_tests():
    tool_selection_passed = 0
    argument_validity_passed = 0
    clean_routing_passed = 0
    total_unnecessary_tool_calls = 0
    graceful_failure_pass_count = 0
    request_misaligned_call_count = 0

    for test in MANUAL_TESTS:
        result = answer_with_tools(test["question"], verbose=False)
        trace = _tool_call_trace(result["tool_events"])
        evaluation = _evaluate_trace(test, trace)

        tool_selection_passed += int(evaluation["required_tools_present"])
        argument_validity_passed += int(evaluation["arguments_valid"])
        clean_routing_passed += int(evaluation["routing_pass"])
        total_unnecessary_tool_calls += len(evaluation["unnecessary_tool_calls"])

        print(f"ID: {test['id']}")
        print(f"Question: {test['question']}")
        print(f"Required tools: {json.dumps(_required_tools(test), ensure_ascii=True)}")
        print(
            "Tool-call trace: "
            f"{json.dumps(trace, ensure_ascii=True)}"
        )
        print(f"Required tools present: {evaluation['required_tools_present']}")
        print(f"Arguments valid: {evaluation['arguments_valid']}")
        print(
            "Unnecessary tool calls: "
            f"{json.dumps(evaluation['unnecessary_tool_calls'], ensure_ascii=True)}"
        )
        print(f"ROUTING {'PASS' if evaluation['routing_pass'] else 'FAIL'}")
        print(f"Final answer: {result['final_answer']}")
        print()

    print(f"tool_selection_accuracy: {tool_selection_passed} / {len(MANUAL_TESTS)}")
    print(f"argument_validity: {argument_validity_passed} / {len(MANUAL_TESTS)}")
    print(f"clean_routing_accuracy: {clean_routing_passed} / {len(MANUAL_TESTS)}")
    print(f"total_unnecessary_tool_calls: {total_unnecessary_tool_calls}")

    for test in UNSUPPORTED_CAPABILITY_TESTS:
        result = answer_with_tools(test["question"], verbose=False)
        trace = _tool_call_trace(result["tool_events"])
        evaluation = _evaluate_unsupported_capability(
            test,
            result["final_answer"],
            trace,
        )

        graceful_failure_pass_count += int(evaluation["graceful_failure_pass"])
        request_misaligned_call_count += len(
            evaluation["request_misaligned_tool_calls"]
        )
        print()
        print(f"ID: {test['id']}")
        print(f"Question: {test['question']}")
        print(
            "Tool-call trace: "
            f"{json.dumps(trace, ensure_ascii=True)}"
        )
        for index, call in enumerate(evaluation["annotated_trace"], start=1):
            print(f"Tool call {index} schema valid: {call['schema_valid']}")
            print(f"Tool call {index} request aligned: {call['request_aligned']}")
        print(f"Tool error observed: {evaluation['tool_error_observed']}")
        print(
            "Request-misaligned tool calls: "
            f"{json.dumps(evaluation['request_misaligned_tool_calls'], ensure_ascii=True)}"
        )
        print(
            "Redundant tool calls: "
            f"{json.dumps(evaluation['redundant_tool_calls'], ensure_ascii=True)}"
        )
        print(
            "Unnecessary tool calls: "
            f"{json.dumps(evaluation['unnecessary_tool_calls'], ensure_ascii=True)}"
        )
        print(
            "Unsupported capability acknowledged: "
            f"{evaluation['unsupported_capability_acknowledged']}"
        )
        print(
            "Unsupported quantitative result invented: "
            f"{evaluation['unsupported_quantitative_result_invented']}"
        )
        print(
            "GRACEFUL FAILURE "
            f"{'PASS' if evaluation['graceful_failure_pass'] else 'FAIL'}"
        )
        print(f"Final answer: {result['final_answer']}")

    print(f"unsupported_capability_cases: {len(UNSUPPORTED_CAPABILITY_TESTS)}")
    print(
        "graceful_failure_pass_count: "
        f"{graceful_failure_pass_count} / {len(UNSUPPORTED_CAPABILITY_TESTS)}"
    )
    print(f"request_misaligned_call_count: {request_misaligned_call_count}")


def summarize_demo_data():
    rows = _load_demo_rows()
    counts = {}
    for row in rows:
        counts[row["drug_type"]] = counts.get(row["drug_type"], 0) + 1
    return len(rows), counts


def main():
    parser = argparse.ArgumentParser(
        description="Experimental Day 16 tool-calling spike."
    )
    parser.add_argument(
        "--question",
        help="Ask one question through the tool-calling loop.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the manual routing tests. This calls the OpenAI API.",
    )
    args = parser.parse_args()

    if args.run_tests:
        run_manual_tests()
    elif args.question:
        answer_with_tools(args.question)
    else:
        row_count, counts = summarize_demo_data()
        print(f"demo_rows: {row_count}")
        print(f"demo_counts: {json.dumps(counts, sort_keys=True)}")
        print("No OpenAI call made. Use --question or --run-tests to run the spike.")


if __name__ == "__main__":
    main()
