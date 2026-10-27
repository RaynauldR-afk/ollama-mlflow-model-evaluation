
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 300


def check_response(
    answer: str,
    check_type: str,
    expected_answer: Any = None,
) -> dict:
    """Run deterministic checks on a model response."""

    result = {
        "check_type": check_type,
        "passed": False,
        "details": "",
    }

    if not isinstance(answer, str) or not answer.strip():
        result["details"] = "The model returned an empty response."
        return result

    cleaned = answer.strip()

    if check_type == "exact_match":
        passed = cleaned == str(expected_answer).strip()
        result.update(
            passed=passed,
            details="Exact answer matched." if passed
            else "Response did not match the expected answer.",
        )

    elif check_type == "non_empty_response":
        result.update(
            passed=True,
            details="Response is non-empty; factual correctness is not established.",
        )

    elif check_type == "json_array":
        try:
            parsed = json.loads(cleaned)
            passed = isinstance(parsed, list)
            result.update(
                passed=passed,
                details="Valid JSON array." if passed
                else "Valid JSON, but the top-level value is not an array.",
            )
        except json.JSONDecodeError as exc:
            result["details"] = f"Invalid JSON: {exc.msg}"

    elif check_type == "json_object":
        try:
            parsed = json.loads(cleaned)
            passed = (
                isinstance(parsed, dict)
                and parsed == expected_answer
            )
            result.update(
                passed=passed,
                details="JSON object matches expected structure and values."
                if passed
                else "JSON object does not match expected keys and values.",
            )
        except json.JSONDecodeError as exc:
            result["details"] = f"Invalid JSON: {exc.msg}"

    else:
        result["details"] = f"Unsupported check type: {check_type}"

    return result


def query_ollama(
    model: str,
    prompt: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Call the local Ollama API and record response metrics."""

    started = time.perf_counter()
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().percent

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        answer = payload.get("response", "")
        elapsed = time.perf_counter() - started
        eval_count = payload.get("eval_count", 0) or 0
        eval_duration = payload.get("eval_duration", 0) or 0

        # Ollama reports eval_duration in nanoseconds.
        tokens_per_second = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_duration > 0
            else 0.0
        )

        return {
            "model": model,
            "success": bool(answer.strip()),
            "answer": answer,
            "latency_seconds": round(elapsed, 4),
            "prompt_tokens": payload.get("prompt_eval_count", 0) or 0,
            "response_tokens": eval_count,
            "tokens_per_second": round(tokens_per_second, 4),
            "cpu_percent_before": cpu_before,
            "memory_percent_before": memory_before,
            "error": None if answer.strip() else "Empty model response",
        }

    except (requests.RequestException, ValueError) as exc:
        return {
            "model": model,
            "success": False,
            "answer": "",
            "latency_seconds": round(time.perf_counter() - started, 4),
            "prompt_tokens": 0,
            "response_tokens": 0,
            "tokens_per_second": 0.0,
            "cpu_percent_before": cpu_before,
            "memory_percent_before": memory_before,
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_case(model: str, test_case: dict) -> dict:
    """Run one test case and combine inference and validation results."""

    inference = query_ollama(model, test_case["prompt"])

    if inference["success"]:
        check = check_response(
            inference["answer"],
            test_case["check_type"],
            test_case.get("expected_answer"),
        )
    else:
        check = {
            "check_type": test_case["check_type"],
            "passed": False,
            "details": "Validation skipped because inference failed.",
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_id": test_case["id"],
        "category": test_case["category"],
        "model": model,
        "success": inference["success"],
        "passed": check["passed"],
        "check": check,
        "inference": inference,
    }


def save_results(results: list[dict], output_path: str | Path) -> Path:
    """Save structured evaluation results as UTF-8 JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
