
import mlflow

from mlflow_tracker import log_evaluation


def test_logs_evaluation_to_separate_database(tmp_path, monkeypatch):
    import mlflow_tracker

    tracking_db = tmp_path / "test_mlflow.db"
    monkeypatch.setattr(mlflow_tracker, "TRACKING_DB", tracking_db)

    results = [
        {
            "test_id": "TEST-001",
            "success": True,
            "passed": True,
            "inference": {
                "latency_seconds": 1.5,
                "tokens_per_second": 12.0,
            },
        },
        {
            "test_id": "TEST-002",
            "success": True,
            "passed": False,
            "inference": {
                "latency_seconds": 2.5,
                "tokens_per_second": 8.0,
            },
        },
    ]

    run_id = log_evaluation(
        model="test-model",
        round_name="baseline",
        results=results,
    )

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    run = mlflow.get_run(run_id)

    assert run.data.tags["model"] == "test-model"
    assert run.data.tags["evaluation_round"] == "baseline"
    assert run.data.metrics["test_count"] == 2
    assert run.data.metrics["passed_count"] == 1
    assert run.data.metrics["pass_rate"] == 0.5
    assert run.data.metrics["mean_latency_seconds"] == 2.0
