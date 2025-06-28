# tests/test_metrics/test_qa.py

from benchai.metrics.qa import QAValidator

def test_exact_match():
    validator = QAValidator()
    score, passed, feedback = validator.score("Paris", "Paris")
    assert passed is True
    assert score == 1.0

def test_fuzzy_match():
    validator = QAValidator()
    score, passed, feedback = validator.score("paris", ["Paris", "Lyon"])
    assert passed is True
    assert score > 0.7