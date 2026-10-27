import json
from pathlib import Path

import mlflow

# Use the SQLite database in the current project folder.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Ollama Three-Model Evaluation")

report_path = Path("evaluation_summary.json")

if not report_path.exists():
    raise FileNotFoundError(
        "evaluation_summary.json was not found in the project folder."
    )

report = json.loads(report_path.read_text(encoding="utf-8"))

if not isinstance(report, dict) or not report.get("models"):
    raise ValueError("The evaluation report has no model summaries.")

results = report.get("results", [])
models = report["models"]

for model, summary in models.items():
    model_results = [
        row for row in results
        if row.get("model") == model
    ]

    print(f"Restoring saved results for {model}...")

    with mlflow.start_run(run_name=f"{model}-recovered"):
        mlflow.log_param("model", model)
        mlflow.log_param("source", "evaluation_summary.json")
        mlflow.log_param("reconstructed_run", True)
        mlflow.log_param("successful_tests", summary["successful_tests"])
        mlflow.log_param("failed_tests", summary["failed_tests"])

        for name, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(name, value)

        for row in model_results:
            test_id = row.get("test_id", "unknown")
            score = row.get("score")

            if isinstance(score, (int, float)):
                mlflow.log_metric(
                    f"score_{test_id}",
                    float(score),
                )

            latency = row.get("latency_seconds")
            if isinstance(latency, (int, float)):
                mlflow.log_metric(
                    f"latency_{test_id}",
                    float(latency),
                )

            mlflow.log_metric(
                f"passed_{test_id}",
                1 if row.get("passed") else 0,
            )

        mlflow.log_text(
            json.dumps(model_results, indent=2, ensure_ascii=False),
            "model_evaluation_results.json",
        )

    print(f"  Restored {len(model_results)} test records.")

print("\nRecovery complete.")
print("Tracking database: mlflow.db")
print("Experiment: Ollama Three-Model Evaluation")