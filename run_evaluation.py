
import json
import sys
from pathlib import Path

from evaluation_engine import evaluate_case, save_results
from mlflow_tracker import log_evaluation


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "test_cases.json"
BASELINE_PATH = ROOT / "reports" / "baseline_results.json"
REPORT_DIR = ROOT / "reports"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))



def compare_with_baseline(current_results, baseline_path):
    """Compare pass rates and latency only when test IDs match."""

    if not baseline_path.exists():
        return {
            "status": "baseline_missing",
            "comparable": False,
            "message": "No baseline report exists; comparison was not performed.",
        }

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Reject empty, duplicate, or mismatched test sets.
    baseline_ids = [item.get("test_id") for item in baseline]
    current_ids = [item.get("test_id") for item in current_results]

    if (
        not baseline_ids
        or not current_ids
        or any(test_id is None for test_id in baseline_ids + current_ids)
        or len(set(baseline_ids)) != len(baseline_ids)
        or len(set(current_ids)) != len(current_ids)
        or set(baseline_ids) != set(current_ids)
    ):
        return {
            "status": "incomparable_test_sets",
            "comparable": False,
            "baseline_test_ids": baseline_ids,
            "current_test_ids": current_ids,
            "message": (
                "Baseline and current results must contain the same "
                "unique test IDs before comparison."
            ),
        }

    def summarize(results):
        latencies = [
            item["inference"]["latency_seconds"]
            for item in results
            if item.get("inference")
            and item["inference"].get("latency_seconds") is not None
        ]

        return {
            "test_count": len(results),
            "pass_rate": (
                sum(bool(item.get("passed")) for item in results)
                / len(results)
            ),
            "mean_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
        }

    baseline_summary = summarize(baseline)
    current_summary = summarize(current_results)

    score_drop = (
        baseline_summary["pass_rate"] - current_summary["pass_rate"]
    )

    old_latency = baseline_summary["mean_latency_seconds"]
    new_latency = current_summary["mean_latency_seconds"]

    latency_increase = (
        (new_latency - old_latency) / old_latency
        if old_latency > 0 and new_latency > 0
        else None
    )

    score_threshold = 0.1
    latency_threshold = 0.5

    regression_detected = (
        score_drop > score_threshold
        or (
            latency_increase is not None
            and latency_increase > latency_threshold
        )
    )

    return {
        "status": (
            "regression_detected"
            if regression_detected
            else "no_threshold_exceeded"
        ),
        "comparable": True,
        "baseline": baseline_summary,
        "current": current_summary,
        "score_drop": score_drop,
        "latency_increase_ratio": latency_increase,
        "thresholds": {
            "score_drop": score_threshold,
            "latency_increase_ratio": latency_threshold,
        },
        "note": "Compare the same model and test cases for meaningful results.",
    }



def run_round(config, round_name):
    models = config["models"]
    selected_cases = [
        case for case in config["test_cases"]
        if round_name in case["rounds"]
    ]

    if not selected_cases:
        raise ValueError(f"No test cases configured for round: {round_name}")

    all_results = []

    for model in models:
        print(f"\nModel: {model} | Round: {round_name}")

        model_results = []

        for case in selected_cases:
            print(f"  Running {case['id']} ...")
            result = evaluate_case(model, case)
            model_results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"  {status} | "
                f"{result['inference']['latency_seconds']:.2f}s"
            )

        all_results.extend(model_results)

        if round_name == "baseline":
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            save_results(model_results, BASELINE_PATH.with_name(
                f"baseline_{model.replace(':', '_')}.json"
            ))

        elif round_name == "regression":
            baseline_file = BASELINE_PATH.with_name(
                f"baseline_{model.replace(':', '_')}.json"
            )
            comparison = compare_with_baseline(
                model_results, baseline_file
            )
            comparison_path = REPORT_DIR / (
                f"regression_{model.replace(':', '_')}.json"
            )
            comparison_path.write_text(
                json.dumps(comparison, indent=2),
                encoding="utf-8",
            )
            print("  Regression comparison:", comparison["status"])

            if not comparison.get("comparable", False):
                print(
                        f"  WARNING: Regression comparison unavailable for {model}. "
                        f"{comparison.get('message', 'Check the baseline and test IDs.')}"
                    )
    

        run_id = log_evaluation(
            model=model,
            round_name=round_name,
            results=model_results,
            suite_version=config["suite_version"],
        )
        print("  MLflow run:", run_id)

    output_path = REPORT_DIR / f"{round_name}_all_models.json"
    save_results(all_results, output_path)
    print(f"\nRound report saved to: {output_path}")

    return all_results


def main():
    config = load_config()

    if len(sys.argv) != 2:
        print("Usage: python run_evaluation.py baseline|robustness|regression")
        raise SystemExit(2)

    round_name = sys.argv[1].lower()
    if round_name not in {"baseline", "robustness", "regression"}:
        print("Invalid round name.")
        raise SystemExit(2)

    results = run_round(config, round_name)
    print(f"\nCompleted {round_name}: {len(results)} test results.")


if __name__ == "__main__":
    main()
