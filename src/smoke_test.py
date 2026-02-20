from __future__ import annotations
"""Lightweight smoke checks for portfolio readiness.

Run after training to verify core project flows quickly.
"""

import csv
from pathlib import Path

from chat import get_answer_with_source, get_available_topics, load_index
from evaluate import evaluate_method, write_csv
from utils import project_root


def assert_true(condition: bool, message: str) -> None:
	if not condition:
		raise AssertionError(message)


def main() -> None:
	root = project_root()
	models_dir = root / "models"

	# 1) Ensure trained artifact exists.
	model_path = models_dir / "faq_index_tfidf.joblib"
	assert_true(model_path.exists(), "Missing model artifact: models/faq_index_tfidf.joblib")

	# 2) Ensure chatbot can answer and expose source URL when available.
	payload = load_index("tfidf")
	answer, score, source = get_answer_with_source("What is Python?", payload)
	assert_true(bool(answer.strip()), "Empty chatbot answer")
	assert_true(score >= 0.0, "Score must be non-negative")
	assert_true(source.startswith("http") or source == "", "Invalid source URL format")

	# 3) Ensure expanded topic list is available.
	topics = get_available_topics(payload)
	for required_topic in ["sports", "weather", "food", "politics", "news", "health", "technology"]:
		assert_true(required_topic in topics, f"Expected topic not found: {required_topic}")

	# 4) Ensure evaluation can run and CSV output has expected headers.
	result = evaluate_method("tfidf")
	assert_true(result.total > 0, "Evaluation total must be > 0")

	out_path = root / "results" / "smoke_eval.csv"
	write_csv([result], str(out_path))
	assert_true(out_path.exists(), "Smoke evaluation CSV not created")

	with out_path.open("r", encoding="utf-8", newline="") as file:
		reader = csv.reader(file)
		headers = next(reader)
		assert_true(
			headers == ["method", "total", "correct", "accuracy", "fallback_rate"],
			"Unexpected evaluation CSV headers",
		)

	print("Smoke test passed ✅")


if __name__ == "__main__":
	main()
