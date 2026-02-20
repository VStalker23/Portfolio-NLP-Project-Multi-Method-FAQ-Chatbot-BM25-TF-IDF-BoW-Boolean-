from __future__ import annotations
"""Interactive CLI chatbot runtime.

Loads previously trained artifacts, scores user input against known FAQ
patterns, and returns an answer or fallback message.
"""

import argparse
import os
import random

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from utils import normalize_text, project_root, score_bm25, score_boolean, validate_method


FALLBACK_MESSAGE = "I am not sure about that yet. Please rephrase your question."


class TerminalUI:
	"""Small helper for friendly terminal rendering (colors + commands text)."""

	RESET = "\033[0m"
	BOLD = "\033[1m"
	CYAN = "\033[96m"
	GREEN = "\033[92m"
	YELLOW = "\033[93m"
	MAGENTA = "\033[95m"
	GRAY = "\033[90m"

	def __init__(self) -> None:
		# Enable ANSI colors on modern Windows terminals.
		if os.name == "nt":
			os.system("")
		self.use_color = True

	def style(self, text: str, color: str = "", bold: bool = False) -> str:
		if not self.use_color:
			return text
		prefix = ""
		if bold:
			prefix += self.BOLD
		if color:
			prefix += color
		return f"{prefix}{text}{self.RESET}"

	def banner(self, method: str) -> None:
		print(self.style("\n╔══════════════════════════════════════════════╗", self.CYAN, True))
		print(self.style("║          FAQ Chatbot Terminal UI            ║", self.CYAN, True))
		print(self.style("╚══════════════════════════════════════════════╝", self.CYAN, True))
		print(self.style(f"Method: {method}", self.MAGENTA, True))
		print(self.style("Type /help to see commands.", self.GRAY))

	def help(self) -> None:
		print(self.style("\nCommands:", self.YELLOW, True))
		print(self.style("  /help   Show this help", self.GRAY))
		print(self.style("  /topics Show available topic names", self.GRAY))
		print(self.style("  /topic  Show or change active topic filter", self.GRAY))
		print(self.style("  /list   Show available topics and question examples", self.GRAY))
		print(self.style("  /clear  Clear terminal screen", self.GRAY))
		print(self.style("  /quit   Exit chat", self.GRAY))

	def user_prompt(self) -> str:
		return self.style("You » ", self.GREEN, True)

	def bot_answer(self, text: str) -> None:
		print(self.style("Bot » ", self.CYAN, True) + text)

	def meta(self, text: str) -> None:
		print(self.style(text, self.GRAY))

	def warn(self, text: str) -> None:
		print(self.style(text, self.YELLOW, True))

	def clear_screen(self) -> None:
		os.system("cls" if os.name == "nt" else "clear")


def format_topic_name(tag: str) -> str:
	"""Convert internal tag to readable topic name for terminal output."""
	cleaned = tag
	if cleaned.startswith("web_"):
		cleaned = cleaned[4:]
	return cleaned.replace("_", " ").strip().title()


def normalized_topic(tag: str) -> str:
	"""Normalize tag to internal topic key used for filtering."""
	cleaned = tag.strip().lower()
	if cleaned.startswith("web_"):
		cleaned = cleaned[4:]
	return cleaned.replace(" ", "_")


def get_available_topics(index_payload: dict) -> list[str]:
	"""Return sorted unique topic keys from loaded examples."""
	examples = index_payload.get("examples", [])
	topics = {normalized_topic(str(example.get("tag", "general"))) for example in examples}
	return sorted(topic for topic in topics if topic)


def show_available_topics(index_payload: dict, ui: TerminalUI) -> None:
	"""Print available question topics and short sample patterns."""
	examples = index_payload.get("examples", [])
	if not examples:
		ui.warn("No examples found in the loaded model.")
		return

	topic_to_patterns: dict[str, list[str]] = {}
	for example in examples:
		tag = normalized_topic(str(example.get("tag", "general")))
		pattern = str(example.get("pattern", "")).strip()
		if not pattern:
			continue
		topic_to_patterns.setdefault(tag, [])
		if pattern not in topic_to_patterns[tag]:
			topic_to_patterns[tag].append(pattern)

	ui.meta("\nAvailable topics and sample questions:")
	for tag in sorted(topic_to_patterns.keys()):
		topic_label = format_topic_name(tag)
		patterns = topic_to_patterns[tag][:3]
		ui.meta(f"- {topic_label} ({len(topic_to_patterns[tag])} questions)")
		for sample in patterns:
			ui.meta(f"    • {sample}")


def choose_initial_topic(index_payload: dict, ui: TerminalUI) -> str:
	"""Ask user for initial topic filter (or all)."""
	topics = get_available_topics(index_payload)
	ui.meta("\nChoose a topic filter now (press Enter for all):")
	ui.meta("  all | " + " | ".join(topics))
	selected = input(ui.style("Topic » ", ui.MAGENTA, True)).strip().lower()

	if not selected or selected == "all":
		return "all"

	candidate = normalized_topic(selected)
	if candidate not in topics:
		ui.warn(f"Unknown topic '{selected}'. Using all topics.")
		return "all"

	return candidate


def load_index(method: str) -> dict:
	"""Load method-specific retrieval artifacts from models directory."""
	method = validate_method(method)
	index_path = project_root() / "models" / f"faq_index_{method}.joblib"
	return joblib.load(index_path)


def compute_scores(normalized_query: str, index_payload: dict) -> list[float]:
	"""Compute similarity/relevance scores for all patterns.

	Scoring depends on the method stored in payload:
	- tfidf/bow -> cosine similarity
	- bm25 -> BM25 relevance
	- boolean -> overlap ratio
	"""
	method = index_payload.get("method", "tfidf")

	if method in {"tfidf", "bow"}:
		vectorizer = index_payload["vectorizer"]
		matrix = index_payload["matrix"]
		query_vector = vectorizer.transform([normalized_query])
		return cosine_similarity(query_vector, matrix).ravel().tolist()
	if method == "bm25":
		query_tokens = normalized_query.split()
		return score_bm25(query_tokens, index_payload["bm25_index"])

	query_tokens = normalized_query.split()
	return score_boolean(query_tokens, index_payload["doc_tokens"])


def get_answer_with_source(user_text: str, index_payload: dict, topic_filter: str = "all") -> tuple[str, float, str]:
	"""Return chatbot answer text, score, and optional source URL."""
	normalized = normalize_text(user_text)
	if not normalized:
		return FALLBACK_MESSAGE, 0.0, ""

	examples = index_payload["examples"]
	threshold = index_payload.get("threshold", 0.25)
	scores = compute_scores(normalized, index_payload)

	if topic_filter != "all":
		for idx, example in enumerate(examples):
			if normalized_topic(str(example.get("tag", "general"))) != topic_filter:
				scores[idx] = float("-inf")

	if not scores or max(scores) == float("-inf"):
		return FALLBACK_MESSAGE, 0.0, ""

	best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
	best_score = float(scores[best_idx])

	# If relevance is too low, respond with a fixed fallback message.
	if best_score < threshold:
		return FALLBACK_MESSAGE, best_score, ""

	responses = examples[best_idx]["responses"]
	answer = random.choice(responses) if responses else FALLBACK_MESSAGE
	source_url = str(examples[best_idx].get("source_url", ""))
	return answer, best_score, source_url


def get_answer(user_text: str, index_payload: dict) -> tuple[str, float]:
	"""Return chatbot answer text and match score for one user query."""
	answer, score, _source_url = get_answer_with_source(user_text, index_payload)
	return answer, score


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for chat runtime."""
	parser = argparse.ArgumentParser(description="Run FAQ chatbot")
	parser.add_argument(
		"--method",
		default="tfidf",
		choices=["tfidf", "bow", "bm25", "boolean"],
		help="Retrieval method",
	)
	return parser.parse_args()


def run_chat(method: str = "tfidf") -> None:
	"""Start interactive chat loop for the selected retrieval method."""
	ui = TerminalUI()
	try:
		index_payload = load_index(method)
	except FileNotFoundError:
		ui.warn(f"Model not found for method '{method}'. Run: python src/train.py --method {method}")
		return

	ui.banner(method)
	active_topic = choose_initial_topic(index_payload, ui)
	ui.meta(f"Active topic: {format_topic_name(active_topic)}")
	ui.meta("Use /topics to see topics, /topic <name> to switch, /topic all to remove filter.")
	while True:
		user_text = input(ui.user_prompt()).strip()
		command = user_text.lower()

		if command in {"/quit", "quit", "exit", "bye"}:
			ui.bot_answer("Goodbye! 👋")
			break
		if command == "/help":
			ui.help()
			continue
		if command == "/topics":
			ui.meta("Available topics: all | " + " | ".join(get_available_topics(index_payload)))
			continue
		if command == "/list":
			show_available_topics(index_payload, ui)
			continue
		if command.startswith("/topic"):
			parts = user_text.split(maxsplit=1)
			if len(parts) == 1:
				ui.meta(f"Current topic: {format_topic_name(active_topic)}")
				ui.meta("Usage: /topic <name> or /topic all")
				continue

			requested = normalized_topic(parts[1])
			available_topics = get_available_topics(index_payload)
			if requested == "all":
				active_topic = "all"
				ui.meta("Topic filter removed. Using all topics.")
			elif requested in available_topics:
				active_topic = requested
				ui.meta(f"Switched topic to: {format_topic_name(active_topic)}")
			else:
				ui.warn("Unknown topic. Use /topics to see valid topic names.")
			continue
		if command == "/clear":
			ui.clear_screen()
			ui.banner(method)
			ui.meta(f"Active topic: {format_topic_name(active_topic)}")
			continue
		if not user_text:
			ui.meta("(Tip: Ask a question or type /help)")
			continue

		answer, score, source_url = get_answer_with_source(user_text, index_payload, topic_filter=active_topic)
		ui.bot_answer(answer)
		ui.meta(f"  score: {score:.3f}")
		if source_url:
			ui.meta(f"  source: {source_url}")


if __name__ == "__main__":
	args = parse_args()
	run_chat(method=args.method)
