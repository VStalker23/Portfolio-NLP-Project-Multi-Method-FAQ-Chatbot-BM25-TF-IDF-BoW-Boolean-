from __future__ import annotations
"""Visualization script for portfolio-ready NLP outputs.

Generates:
- wordclouds for questions, answers, topics
- topic count chart (questions vs answers)
- evaluation method comparison chart from CSV
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from utils import load_intents, project_root


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for visualization input/output paths."""
	parser = argparse.ArgumentParser(description="Generate FAQ project visualizations")
	parser.add_argument(
		"--eval-csv",
		default="results/eval_metrics.csv",
		help="Path to evaluation CSV from src/evaluate.py",
	)
	parser.add_argument(
		"--out-dir",
		default="results/plots",
		help="Output directory for charts and wordcloud images",
	)
	return parser.parse_args()


def save_wordcloud(text: str, title: str, out_path: Path) -> None:
	"""Create and save one wordcloud PNG from text data."""
	if not text.strip():
		return

	wordcloud = WordCloud(
		width=1600,
		height=900,
		background_color="white",
		collocations=False,
	).generate(text)

	plt.figure(figsize=(14, 8))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.title(title)
	plt.tight_layout()
	plt.savefig(out_path, dpi=180)
	plt.close()


def build_text_blocks(intents_data: dict) -> tuple[str, str, str, list[str], list[int], list[int]]:
	"""Prepare combined text and counts from intents for plotting.

	Returns:
	- full question text
	- full answer text
	- weighted topic text (for wordcloud emphasis)
	- topic names
	- question count per topic
	- answer count per topic
	"""
	all_questions: list[str] = []
	all_answers: list[str] = []
	topic_weights: list[str] = []
	topic_names: list[str] = []
	question_counts: list[int] = []
	answer_counts: list[int] = []

	for intent in intents_data.get("intents", []):
		tag = str(intent.get("tag", "unknown"))
		patterns = [str(value) for value in intent.get("patterns", [])]
		responses = [str(value) for value in intent.get("responses", [])]

		all_questions.extend(patterns)
		all_answers.extend(responses)
		topic_names.append(tag)
		question_counts.append(len(patterns))
		answer_counts.append(len(responses))

		weight = max(1, len(patterns) + len(responses))
		topic_weights.extend([tag.replace("_", " ")] * weight)

	return (
		" ".join(all_questions),
		" ".join(all_answers),
		" ".join(topic_weights),
		topic_names,
		question_counts,
		answer_counts,
	)


def save_topic_counts(topics: list[str], question_counts: list[int], answer_counts: list[int], out_path: Path) -> None:
	"""Save bar chart comparing question/answer counts by topic tag."""
	positions = range(len(topics))
	bar_width = 0.4

	plt.figure(figsize=(14, 8))
	plt.bar([index - bar_width / 2 for index in positions], question_counts, width=bar_width, label="Questions")
	plt.bar([index + bar_width / 2 for index in positions], answer_counts, width=bar_width, label="Answers")
	plt.xticks(list(positions), topics, rotation=30, ha="right")
	plt.ylabel("Count")
	plt.title("Question and Answer Counts by Topic")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=180)
	plt.close()


def save_eval_chart(eval_csv: Path, out_path: Path) -> None:
	"""Save bar chart from evaluation CSV (accuracy vs fallback rate)."""
	if not eval_csv.exists():
		print(f"Skip eval chart: CSV not found at {eval_csv}")
		return

	methods: list[str] = []
	accuracies: list[float] = []
	fallback_rates: list[float] = []

	with eval_csv.open("r", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		for row in reader:
			methods.append(str(row["method"]))
			accuracies.append(float(row["accuracy"]) * 100)
			fallback_rates.append(float(row["fallback_rate"]) * 100)

	if not methods:
		print("Skip eval chart: CSV has no data rows")
		return

	positions = range(len(methods))
	bar_width = 0.4

	plt.figure(figsize=(12, 7))
	plt.bar([index - bar_width / 2 for index in positions], accuracies, width=bar_width, label="Accuracy %")
	plt.bar([index + bar_width / 2 for index in positions], fallback_rates, width=bar_width, label="Fallback Rate %")
	plt.xticks(list(positions), methods)
	plt.ylabel("Percent")
	plt.title("Method Comparison: Accuracy vs Fallback Rate")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=180)
	plt.close()


def main() -> None:
	"""CLI entrypoint: generate all visual outputs in one run."""
	args = parse_args()
	root = project_root()
	out_dir = root / args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	intents_data = load_intents(root / "data" / "intents.json")
	(
		questions_text,
		answers_text,
		topics_text,
		topic_names,
		question_counts,
		answer_counts,
	) = build_text_blocks(intents_data)

	save_wordcloud(questions_text, "Wordcloud: FAQ Questions", out_dir / "wordcloud_questions.png")
	save_wordcloud(answers_text, "Wordcloud: FAQ Answers", out_dir / "wordcloud_answers.png")
	save_wordcloud(topics_text, "Wordcloud: FAQ Topics", out_dir / "wordcloud_topics.png")
	save_topic_counts(topic_names, question_counts, answer_counts, out_dir / "topic_question_answer_counts.png")
	save_eval_chart(root / args.eval_csv, out_dir / "evaluation_comparison.png")

	print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
	main()
