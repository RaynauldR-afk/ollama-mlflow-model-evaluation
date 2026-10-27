
from evaluation_engine import check_response


def test_exact_match_passes():
    assert check_response("96", "exact_match", "96")["passed"]


def test_exact_match_rejects_wrong_answer():
    assert not check_response("97", "exact_match", "96")["passed"]


def test_json_array_passes():
    answer = '["Python", "Java", "Go"]'
    assert check_response(answer, "json_array")["passed"]


def test_json_object_passes():
    answer = '{"status": "ok", "count": 3}'
    expected = {"status": "ok", "count": 3}

    assert check_response(
        answer, "json_object", expected
    )["passed"]


def test_invalid_json_fails():
    assert not check_response(
        "{bad json", "json_object", {}
    )["passed"]


def test_empty_response_fails():
    assert not check_response(
        "", "exact_match", "96"
    )["passed"]


def test_unsupported_check_type_fails():
    result = check_response("hello", "unknown_check")

    assert not result["passed"]
    assert "Unsupported check type" in result["details"]
