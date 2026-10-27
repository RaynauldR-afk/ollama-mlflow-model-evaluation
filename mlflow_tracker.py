
import json
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "benchmark_mlflow.db"
EXPERIMENT_NAME = "Ollama Benchmark Framework"


def configure_mlflow():
    """Configure a separate MLflow database for the new framework."""

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_evaluation(
    model: str,
    round_name: str,
    results: list[dict],
    suite_version: str = "1.0.0",
) -> str:
    """Record one model's evaluation round as an MLflow run."""

    if round_name not in {"baseline", "robustness", "regression"}:
        raise ValueError(f"Unsupported evaluation round: {round_name}")

    configure_mlflow()

    total = len(results)
    passed = sum(bool(item.get("passed")) for item in results)
    successful = sum(bool(item.get("success")) for item in results)

    latencies = [
        float(item["inference"]["latency_seconds"])
        for item in results
        if item.get("inference")
        and "latency_seconds" in item["inference"]
    ]

    token_rates = [
        float(item["inference"]["tokens_per_second"])
        for item in results
        if item.get("inference")
        and "tokens_per_second" in item["inference"]
    ]

    with mlflow.start_run(
        run_name=f"{round_name}-{model.replace(':', '-')}"
    ) as run:
        mlflow.set_tags({
            "model": model,
            "evaluation_round": round_name,
            "suite_version": suite_version,
            "framework": "ollama-benchmark",
        })

        mlflow.log_params({
            "model": model,
            "round_name": round_name,
            "suite_version": suite_version,
            "test_count": total,
        })

        mlflow.log_metrics({
            "test_count": total,
            "passed_count": passed,
            "successful_inference_count": successful,
            "pass_rate": passed / total if total else 0.0,
            "inference_success_rate": successful / total if total else 0.0,
            "mean_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "mean_tokens_per_second": (
                sum(token_rates) / len(token_rates) if token_rates else 0.0
            ),
        })

        report_path = PROJECT_ROOT / "reports" / (
            f"{round_name}_{model.replace(':', '_')}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(report_path), artifact_path="test_results")

        return run.info.run_id
