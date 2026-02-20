from __future__ import annotations
"""Evaluation script for comparing retrieval methods.

Uses leave-one-pattern-out style checking:
- each pattern is used as a query
- that same row is excluded from candidate matches
- top-1 tag match is counted as correct
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from chat import compute_scores, load_index
from utils import SUPPORTED_METHODS, normalize_text


@dataclass
class EvalResult:
	"""Container for evaluation metrics of a single method."""
	method: str
	total: int
	correct: int
	fallbacks: int

	@property
	def accuracy(self) -> float:
		return 0.0 if self.total == 0 else self.correct / self.total

	@property
	def fallback_rate(self) -> float:
		return 0.0 if self.total == 0 else self.fallbacks / self.total


def evaluate_method(method: str) -> EvalResult:
	"""Evaluate one retrieval method and return aggregated metrics."""
	payload = load_index(method)
	examples = payload["examples"]
	threshold = float(payload.get("threshold", 0.25))

	total = 0
	correct = 0
	fallbacks = 0

	for query_idx, example in enumerate(examples):
		# Query each stored pattern and evaluate whether top-1 returns same tag.
		normalized = normalize_text(example["pattern"])
		if not normalized:
			continue

		scores = compute_scores(normalized, payload)
		if not scores:
			continue

		total += 1
		# Prevent trivial self-match by excluding the current row from ranking.
		scores[query_idx] = float("-inf")

		best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
		best_score = float(scores[best_idx])

		if best_score < threshold:
			fallbacks += 1
			continue

		if examples[best_idx]["tag"] == example["tag"]:
			correct += 1

	return EvalResult(method=method, total=total, correct=correct, fallbacks=fallbacks)


def parse_args() -> argparse.Namespace:
	"""Parse CLI args for method selection and optional CSV export."""
	parser = argparse.ArgumentParser(description="Evaluate FAQ chatbot retrieval methods")
	parser.add_argument(
		"--method",
		default="all",
		help="One method (tfidf|bow|bm25|boolean) or 'all'",
	)
	parser.add_argument(
		"--out",
		default="",
		help="Optional CSV output path, e.g. results/eval.csv",
	)
	return parser.parse_args()


def write_csv(results: list[EvalResult], out_path: str) -> Path:
	"""Write evaluation table to CSV for plotting/reporting."""
	path = Path(out_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(["method", "total", "correct", "accuracy", "fallback_rate"])
		for result in results:
			writer.writerow(
				[
					result.method,
					result.total,
					result.correct,
					f"{result.accuracy:.6f}",
					f"{result.fallback_rate:.6f}",
				]
			)

	return path


def main() -> None:
	"""CLI entrypoint: run evaluation and optionally save CSV output."""
	args = parse_args()
	requested = args.method.lower().strip()

	if requested == "all":
		methods = sorted(SUPPORTED_METHODS)
	else:
		if requested not in SUPPORTED_METHODS:
			raise ValueError(
				f"Unsupported method '{requested}'. Use one of: {', '.join(sorted(SUPPORTED_METHODS))}, all"
			)
		methods = [requested]

	print("Method   Total   Correct   Accuracy   FallbackRate")
	print("-" * 52)
	results: list[EvalResult] = []
	for method in methods:
		result = evaluate_method(method)
		results.append(result)
		print(
			f"{result.method:<8} {result.total:<7} {result.correct:<9} "
			f"{result.accuracy:>7.2%}   {result.fallback_rate:>11.2%}"
		)

	if args.out:
		csv_path = write_csv(results, args.out)
		print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
	main()
