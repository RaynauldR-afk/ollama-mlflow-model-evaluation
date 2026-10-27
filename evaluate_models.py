import json
import time
import threading
from pathlib import Path
from statistics import mean

import mlflow
import psutil
import requests
from mlflow import log_metric, log_param, log_text

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = [
    "llama3.2:latest",
    "qwen2.5:3b",
    "gemma2:2b",
]

OUTPUT_FILE = Path("evaluation_summary.json")
BASELINE_FILE = Path("baseline_results.json")
EXPERIMENT_NAME = "Ollama Three-Model Evaluation"
TIMEOUT_SECONDS = 300

# Each case targets one or more evaluation categories.
TESTS = [
    {
        "id": "functional-001",
        "category": "functional_correctness",
        "prompt": (
            "Write a Python function named is_prime(n) that returns "
            "True for prime numbers and False otherwise. Include code."
        ),
        "expected": ["def ", "return"],
        "check": "code",
    },
    {
        "id": "functional-002",
        "category": "functional_correctness",
        "prompt": (
            "A university has 200 students. If 15% are absent, "
            "how many are present? Explain the calculation."
        ),
        "expected": ["170"],
        "check": "number",
    },
    {
        "id": "quality-001",
        "category": "output_quality",
        "prompt": (
            "Explain retrieval-augmented generation (RAG) in simple "
            "terms. Include how retrieval works and one practical example."
        ),
        "expected": ["retriev", "generat", "example"],
        "check": "keywords",
    },
    {
        "id": "regression-001",
        "category": "regression_testing",
        "prompt": (
            "In Python, what is the result of 12 * 12? "
            "Give the answer and a short explanation."
        ),
        "expected": ["144"],
        "check": "number",
    },
    {
        "id": "hallucination-001",
        "category": "hallucination_checks",
        "prompt": (
            "How many days are in a standard week? Answer with the "
            "number and one short sentence."
        ),
        "expected": ["7", "week"],
        "check": "number",
    },
    {
        "id": "robustness-001",
        "category": "robustness",
        "prompt": (
            'Return a JSON object with exactly two keys: "status" '
            'with value "ok", and "count" with value 3. '
            "Return JSON only, with no Markdown."
        ),
        "expected": ['"status"', '"count"', '"ok"', "3"],
        "check": "json",
    },
    {
        "id": "latency-001",
        "category": "latency_throughput",
        "prompt": (
            "Explain in two short sentences what a Python dictionary is."
        ),
        "expected": ["key", "value"],
        "check": "keywords",
    },
]

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------

class ResourceMonitor:
    """Samples system CPU and memory while a request is running."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.cpu_samples = []
        self.memory_samples = []

    def _sample(self):
        while not self.stop_event.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.memory_samples.append(
                psutil.virtual_memory().percent
            )
            self.stop_event.wait(0.5)

    def start(self):
        # Prime the CPU measurement before sampling.
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def results(self):
        return {
            "avg_system_cpu_percent": round(
                mean(self.cpu_samples), 2
            ) if self.cpu_samples else 0.0,
            "peak_sampled_system_cpu_percent": round(
                max(self.cpu_samples), 2
            ) if self.cpu_samples else 0.0,
            "avg_system_memory_percent": round(
                mean(self.memory_samples), 2
            ) if self.memory_samples else round(
                psutil.virtual_memory().percent, 2
            ),
        }


# ---------------------------------------------------------
# Ollama request and response validation
# ---------------------------------------------------------

def query_ollama(model, prompt):
    monitor = ResourceMonitor()
    start = time.perf_counter()
    monitor.start()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    finally:
        monitor.stop()

    elapsed = time.perf_counter() - start
    answer = data.get("response", "")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty or invalid response.")

    completion_tokens = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)

    # Prefer Ollama's generation duration when available.
    tokens_per_second = None
    if eval_duration_ns and completion_tokens:
        tokens_per_second = round(
            completion_tokens / (eval_duration_ns / 1_000_000_000),
            3,
        )

    return {
        "answer": answer,
        "latency_seconds": round(elapsed, 3),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
        "resource": monitor.results(),
    }


# ---------------------------------------------------------
# Evaluation scoring
# ---------------------------------------------------------

def evaluate_answer(answer, test):
    """Return heuristic scores; these are not proof of correctness."""
    text = answer.lower()
    expected = test["expected"]
    check = test["check"]
    checks = {}

    checks["non_empty"] = bool(answer.strip())
    checks["expected_keywords"] = all(
        keyword.lower() in text for keyword in expected
    )

    if check == "code":
        checks["contains_function"] = "def is_prime" in text
        checks["contains_return"] = "return" in text

    elif check == "number":
        # Checks for the expected answer as a standalone number.
        import re

        expected_number = expected[0]
        checks["contains_expected_number"] = bool(
            re.search(
                rf"(?<!\d){re.escape(expected_number)}(?!\d)",
                answer,
            )
        )

    elif check == "json":
        try:
            parsed = json.loads(answer)
            checks["valid_json"] = isinstance(parsed, dict)
            checks["json_values_match"] = (
                isinstance(parsed, dict)
                and parsed.get("status") == "ok"
                and parsed.get("count") == 3
                and set(parsed.keys()) == {"status", "count"}
            )
        except (json.JSONDecodeError, TypeError):
            checks["valid_json"] = False
            checks["json_values_match"] = False

    passed = sum(checks.values())
    score = round(passed / len(checks), 3) if checks else 0.0

    return {
        "score": score,
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
    }


# ---------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------

def compare_baseline(current):
    """
    First successful run creates a baseline.
    Later runs compare average keyword/check score and latency.
    """
    if not BASELINE_FILE.exists():
        BASELINE_FILE.write_text(
            json.dumps(current, indent=2),
            encoding="utf-8",
        )
        return {
            "status": "baseline_created",
            "message": "Run the evaluator again to compare against this baseline.",
        }

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    comparisons = {}

    for model, metrics in current.items():
        old = baseline.get(model)
        if not old:
            comparisons[model] = {
                "status": "no_matching_baseline",
            }
            continue

        old_score = old.get("average_score", 0)
        new_score = metrics.get("average_score", 0)
        old_latency = old.get("average_latency_seconds", 0)
        new_latency = metrics.get("average_latency_seconds", 0)

        score_drop = round(old_score - new_score, 3)
        latency_increase_percent = None

        if old_latency > 0:
            latency_increase_percent = round(
                ((new_latency - old_latency) / old_latency) * 100,
                2,
            )

        comparisons[model] = {
            "baseline_average_score": old_score,
            "current_average_score": new_score,
            "score_drop": score_drop,
            "baseline_average_latency_seconds": old_latency,
            "current_average_latency_seconds": new_latency,
            "latency_change_percent": latency_increase_percent,
            "possible_regression": (
                score_drop > 0.1
                or (
                    latency_increase_percent is not None
                    and latency_increase_percent > 50
                )
            ),
        }

    return {
        "status": "compared",
        "thresholds": {
            "score_drop_greater_than": 0.1,
            "latency_increase_percent_greater_than": 50,
        },
        "models": comparisons,
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():
    all_results = []
    model_summaries = {}

    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {model}")
        print("=" * 60)

        model_records = []

        with mlflow.start_run(run_name=model):
            log_param("model", model)
            log_param("test_count", len(TESTS))
            log_param("temperature", 0)
            log_param("timeout_seconds", TIMEOUT_SECONDS)
            log_param("python_version", __import__("sys").version.split()[0])

            for test in TESTS:
                print(
                    f"\nTest: {test['id']} "
                    f"[{test['category']}]"
                )

                record = {
                    "model": model,
                    "test_id": test["id"],
                    "category": test["category"],
                    "error": None,
                }

                try:
                    result = query_ollama(model, test["prompt"])
                    evaluation = evaluate_answer(
                        result["answer"], test
                    )

                    record.update(result)
                    record.update(evaluation)
                    record["passed"] = (
                        evaluation["passed_checks"]
                        == evaluation["total_checks"]
                    )

                    print(f"  Latency: {result['latency_seconds']} s")
                    print(f"  Score: {evaluation['score']}")
                    print(f"  Checks passed: "
                          f"{evaluation['passed_checks']}/"
                          f"{evaluation['total_checks']}")
                    print(f"  Answer: {result['answer'][:200]!r}")

                except Exception as exc:
                    record.update({
                        "passed": False,
                        "score": 0.0,
                        "checks": {},
                        "latency_seconds": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "tokens_per_second": None,
                        "resource": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  ERROR: {record['error']}")

                model_records.append(record)
                all_results.append(record)

            successful = [
                row for row in model_records
                if row.get("error") is None
            ]
            latencies = [
                row["latency_seconds"] for row in successful
                if row.get("latency_seconds") is not None
            ]
            scores = [
                row["score"] for row in model_records
                if row.get("score") is not None
            ]
            speeds = [
                row["tokens_per_second"] for row in successful
                if row.get("tokens_per_second") is not None
            ]
            cpu_values = [
                row["resource"]["avg_system_cpu_percent"]
                for row in successful if row.get("resource")
            ]
            memory_values = [
                row["resource"]["avg_system_memory_percent"]
                for row in successful if row.get("resource")
            ]

            summary = {
                "successful_tests": len(successful),
                "failed_tests": len(model_records) - len(successful),
                "average_score": round(mean(scores), 3) if scores else 0,
                "average_latency_seconds": round(mean(latencies), 3)
                if latencies else 0,
                "max_latency_seconds": round(max(latencies), 3)
                if latencies else 0,
                "average_tokens_per_second": round(mean(speeds), 3)
                if speeds else None,
                "average_system_cpu_percent": round(mean(cpu_values), 2)
                if cpu_values else None,
                "average_system_memory_percent": round(mean(memory_values), 2)
                if memory_values else None,
                "tests_per_minute": round(
                    len(successful) / (sum(latencies) / 60), 3
                ) if latencies and sum(latencies) > 0 else 0,
            }
            model_summaries[model] = summary

            for metric, value in summary.items():
                if isinstance(value, (int, float)) and value is not None:
                    log_metric(metric, value)

            log_metric("functional_tests_passed", sum(
                row.get("passed", False)
                for row in model_records
                if row["category"] == "functional_correctness"
            ))
            log_metric("robustness_tests_passed", sum(
                row.get("passed", False)
                for row in model_records
                if row["category"] == "robustness"
            ))

            log_text(
                json.dumps(model_records, indent=2, ensure_ascii=False),
                "model_evaluation_results.json",
            )

            print("\nModel summary:")
            print(json.dumps(summary, indent=2))

    regression = compare_baseline(model_summaries)

    final_report = {
        "experiment": EXPERIMENT_NAME,
        "models": model_summaries,
        "regression_comparison": regression,
        "results": all_results,
        "notes": [
            "Scores are heuristic checks, not a complete measure of answer quality.",
            "Hallucination checks use simple expected-answer checks, not a true hallucination detector.",
            "CPU and memory readings are system-wide samples and may miss short peaks.",
            "Throughput uses Ollama generation duration when available.",
            "The first run creates a regression baseline; subsequent runs compare against it.",
        ],
    }

    OUTPUT_FILE.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Detailed report: {OUTPUT_FILE}")
    print(f"Regression baseline: {BASELINE_FILE}")
    print("\nRegression comparison:")
    print(json.dumps(regression, indent=2))
    print("\nTo view MLflow experiments, run: mlflow ui")


if __name__ == "__main__":
    main()