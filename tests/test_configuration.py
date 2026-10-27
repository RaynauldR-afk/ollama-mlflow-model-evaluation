
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "test_cases.json"


@pytest.fixture(scope="module")
def config():
    """Load the evaluation configuration once for this test module."""
    assert CONFIG_FILE.exists(), f"Configuration file not found: {CONFIG_FILE}"

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_configuration_has_three_models(config):
    assert len(config["models"]) == 3
    assert len(set(config["models"])) == 3


def test_all_three_evaluation_rounds_exist(config):
    round_names = {item["name"] for item in config["rounds"]}

    assert round_names == {"baseline", "robustness", "regression"}


def test_test_case_ids_are_unique(config):
    test_ids = [item["id"] for item in config["test_cases"]]

    assert len(test_ids) == len(set(test_ids))


def test_every_test_case_has_required_fields(config):
    required_fields = {"id", "category", "prompt", "check_type", "rounds"}
    valid_rounds = {item["name"] for item in config["rounds"]}

    for case in config["test_cases"]:
        assert required_fields.issubset(case.keys()), case.get("id")
        assert case["prompt"].strip(), case["id"]
        assert case["rounds"], case["id"]
        assert set(case["rounds"]).issubset(valid_rounds), case["id"]


def test_json_object_case_has_expected_structure(config):
    cases = {
        item["id"]: item
        for item in config["test_cases"]
    }

    case = cases["ROB-002"]

    assert case["check_type"] == "json_object"
    assert case["expected_answer"] == {
        "status": "ok",
        "count": 3,
    }


def test_evaluation_limits_are_valid(config):
    limits = config["evaluation_limits"]

    assert limits["request_timeout_seconds"] > 0
    assert 0 <= limits["temperature"] <= 2
    assert 0 <= limits["score_regression_threshold"] <= 1
    assert limits["latency_regression_threshold"] >= 0


def test_models_use_expected_local_ollama_names(config):
    assert config["models"] == [
        "llama3.2:3b",
        "qwen2.5:3b",
        "gemma2:2b",
    ]
