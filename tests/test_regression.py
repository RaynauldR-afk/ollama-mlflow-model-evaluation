
import json

from run_evaluation import compare_with_baseline


def make_result(test_id, passed, latency):
    return {
        "test_id": test_id,
        "passed": passed,
        "success": True,
        "inference": {
            "latency_seconds": latency,
            "tokens_per_second": 10.0,
        },
    }


def test_missing_baseline_is_not_treated_as_success(tmp_path):
    results = [make_result("T1", True, 1.0)]

    comparison = compare_with_baseline(
        results, tmp_path / "missing.json"
    )

    assert comparison["status"] == "baseline_missing"


def test_identical_results_do_not_trigger_regression(tmp_path):
    baseline = [
        make_result("T1", True, 1.0),
        make_result("T2", False, 2.0),
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    comparison = compare_with_baseline(baseline, baseline_path)

    assert comparison["status"] == "no_threshold_exceeded"
    assert comparison["score_drop"] == 0.0
    assert comparison["latency_increase_ratio"] == 0.0


def test_significant_score_drop_triggers_regression(tmp_path):
    baseline = [
        make_result("T1", True, 1.0),
        make_result("T2", True, 1.0),
        make_result("T3", True, 1.0),
        make_result("T4", True, 1.0),
        make_result("T5", True, 1.0),
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    current = [
        make_result("T1", True, 1.0),
        make_result("T2", True, 1.0),
        make_result("T3", False, 1.0),
        make_result("T4", False, 1.0),
        make_result("T5", False, 1.0),
    ]

    comparison = compare_with_baseline(current, baseline_path)

    assert comparison["status"] == "regression_detected"
    assert comparison["score_drop"] > 0.1


def test_significant_latency_increase_triggers_regression(tmp_path):
    baseline = [make_result("T1", True, 10.0)]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    current = [make_result("T1", True, 16.0)]

    comparison = compare_with_baseline(current, baseline_path)

    assert comparison["status"] == "regression_detected"
    assert comparison["latency_increase_ratio"] > 0.5


def test_different_test_ids_are_not_compared(tmp_path):
    baseline = [
        make_result("T1", True, 1.0),
        make_result("T2", True, 1.0),
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    current = [
        make_result("T1", True, 1.0),
        make_result("T3", True, 1.0),
    ]

    comparison = compare_with_baseline(current, baseline_path)

    assert comparison["status"] == "incomparable_test_sets"
    assert comparison["comparable"] is False
