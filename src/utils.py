from __future__ import annotations
"""Shared NLP and retrieval utilities.

This module keeps all reusable logic in one place:
- reading dataset files
- normalizing text
- validating retrieval method names
- scoring for Boolean and BM25 retrieval
- preparing training corpus examples
"""

import json
import math
import re
import csv
from collections import Counter
from pathlib import Path
from typing import Any


EN_STOPWORDS = {
	"a",
	"an",
	"the",
	"is",
	"are",
	"am",
	"was",
	"were",
	"to",
	"for",
	"of",
	"and",
	"or",
	"in",
	"on",
	"at",
	"how",
	"what",
}

RU_STOPWORDS = {
	"и",
	"в",
	"на",
	"с",
	"по",
	"о",
	"что",
	"как",
	"это",
	"для",
	"к",
	"из",
}

SUPPORTED_METHODS = {"tfidf", "bow", "bm25", "boolean"}


def project_root() -> Path:
	"""Return project root directory (parent of src/)."""
	return Path(__file__).resolve().parent.parent


def load_intents(intents_path: Path | None = None) -> dict[str, Any]:
	"""Load intents JSON file and return it as a dictionary."""
	path = intents_path or project_root() / "data" / "intents.json"
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def contains_cyrillic(text: str) -> bool:
	"""Detect whether text contains Cyrillic characters (Russian alphabet)."""
	return bool(re.search(r"[а-яё]", text.lower()))


def tokenize(text: str) -> list[str]:
	"""Split text into alphanumeric tokens for English/Russian content."""
	return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


def validate_method(method: str) -> str:
	"""Normalize and validate retrieval method name from user input."""
	method_normalized = method.lower().strip()
	if method_normalized not in SUPPORTED_METHODS:
		raise ValueError(
			f"Unsupported method '{method}'. Supported methods: {', '.join(sorted(SUPPORTED_METHODS))}"
		)
	return method_normalized


def normalize_text(text: str) -> str:
	"""Normalize text by tokenizing and removing lightweight stopwords.

	Language is inferred using Cyrillic detection:
	- Russian stopwords for Cyrillic text
	- English stopwords otherwise
	"""
	tokens = tokenize(text)
	if not tokens:
		return ""

	if contains_cyrillic(text):
		cleaned = [token for token in tokens if token not in RU_STOPWORDS]
	else:
		cleaned = [token for token in tokens if token not in EN_STOPWORDS]

	return " ".join(cleaned)


def score_boolean(query_tokens: list[str], doc_tokens: list[list[str]]) -> list[float]:
	"""Score documents using simple Boolean overlap ratio.

	Formula: matched_query_terms / total_query_terms
	"""
	if not query_tokens:
		return [0.0 for _ in doc_tokens]

	query_set = set(query_tokens)
	scores: list[float] = []
	for doc in doc_tokens:
		doc_set = set(doc)
		matches = len(query_set.intersection(doc_set))
		score = matches / len(query_set)
		scores.append(score)
	return scores


def build_bm25_index(doc_tokens: list[list[str]]) -> dict[str, Any]:
	"""Precompute BM25 structures from tokenized documents.

	Returns a dictionary with IDF values, term frequencies per document,
	document lengths, and BM25 constants.
	"""
	total_docs = len(doc_tokens)
	if total_docs == 0:
		return {
			"idf": {},
			"doc_term_freqs": [],
			"doc_lengths": [],
			"avg_doc_length": 0.0,
			"k1": 1.5,
			"b": 0.75,
		}

	doc_freq: Counter[str] = Counter()
	doc_term_freqs: list[Counter[str]] = []
	doc_lengths: list[int] = []

	for tokens in doc_tokens:
		term_freq = Counter(tokens)
		doc_term_freqs.append(term_freq)
		doc_lengths.append(len(tokens))
		doc_freq.update(term_freq.keys())

	avg_doc_length = sum(doc_lengths) / total_docs
	idf: dict[str, float] = {}
	for term, freq in doc_freq.items():
		idf[term] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

	return {
		"idf": idf,
		"doc_term_freqs": doc_term_freqs,
		"doc_lengths": doc_lengths,
		"avg_doc_length": avg_doc_length,
		"k1": 1.5,
		"b": 0.75,
	}


def score_bm25(query_tokens: list[str], bm25_index: dict[str, Any]) -> list[float]:
	"""Score each document with BM25 using precomputed index data."""
	if not query_tokens:
		return [0.0 for _ in bm25_index.get("doc_term_freqs", [])]

	idf = bm25_index["idf"]
	doc_term_freqs = bm25_index["doc_term_freqs"]
	doc_lengths = bm25_index["doc_lengths"]
	avg_doc_length = bm25_index["avg_doc_length"]
	k1 = bm25_index["k1"]
	b = bm25_index["b"]

	if avg_doc_length == 0:
		return [0.0 for _ in doc_term_freqs]

	query_terms = Counter(query_tokens)
	scores: list[float] = []
	for term_freqs, doc_length in zip(doc_term_freqs, doc_lengths):
		score = 0.0
		for term, qf in query_terms.items():
			if term not in term_freqs:
				continue
			term_idf = idf.get(term, 0.0)
			term_tf = term_freqs[term]
			numerator = term_tf * (k1 + 1)
			denominator = term_tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
			score += term_idf * (numerator / denominator) * qf
		scores.append(score)

	return scores


def prepare_corpus(intents_data: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
	"""Convert intents JSON into normalized corpus + aligned example metadata.

	Returns:
	- corpus: list of normalized patterns (for vectorization/scoring)
	- examples: list of dictionaries used later to map scores to responses
	"""
	corpus: list[str] = []
	examples: list[dict[str, Any]] = []

	for intent in intents_data.get("intents", []):
		tag = intent["tag"]
		responses = intent.get("responses", [])
		for pattern in intent.get("patterns", []):
			normalized = normalize_text(pattern)
			if not normalized:
				continue
			corpus.append(normalized)
			examples.append(
				{
					"tag": tag,
					"pattern": pattern,
					"responses": responses,
				}
			)

	return corpus, examples


def load_csv_qa_rows(csv_path: Path | None = None) -> list[dict[str, str]]:
	"""Load web-sourced FAQ rows from CSV.

	Expected headers:
	- question
	- answer
	- topic
	- source_url
	"""
	rows: list[dict[str, str]] = []
	path = csv_path or project_root() / "data" / "web_faq.csv"
	if not path.exists():
		return rows

	csv_files: list[Path]
	if path.is_dir():
		csv_files = sorted(path.glob("*.csv"))
	else:
		csv_files = [path]

	for csv_file in csv_files:
		with csv_file.open("r", encoding="utf-8", newline="") as file:
			reader = csv.DictReader(file)
			for row in reader:
				question = str(row.get("question", "")).strip()
				answer = str(row.get("answer", "")).strip()
				topic = str(row.get("topic", "general")).strip() or "general"
				source_url = str(row.get("source_url", "")).strip()

				if not question or not answer:
					continue

				rows.append(
					{
						"question": question,
						"answer": answer,
						"topic": topic,
						"source_url": source_url,
					}
				)

	return rows


def load_csv_qa_rows_from_sources(sources: list[Path]) -> list[dict[str, str]]:
	"""Load and merge Q/A rows from multiple file or directory sources."""
	rows: list[dict[str, str]] = []
	for source in sources:
		rows.extend(load_csv_qa_rows(source))
	return rows


def prepare_corpus_from_csv_rows(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]]]:
	"""Convert CSV Q/A rows into normalized corpus and training examples."""
	corpus: list[str] = []
	examples: list[dict[str, Any]] = []

	for row in rows:
		normalized = normalize_text(row["question"])
		if not normalized:
			continue

		corpus.append(normalized)
		examples.append(
			{
				"tag": f"web_{row['topic']}",
				"pattern": row["question"],
				"responses": [row["answer"]],
				"source_url": row["source_url"],
			}
		)

	return corpus, examples
