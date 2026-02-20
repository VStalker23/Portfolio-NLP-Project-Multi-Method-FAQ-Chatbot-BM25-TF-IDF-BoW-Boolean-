from __future__ import annotations
"""Training script for FAQ retrieval indexes.

Builds and saves model artifacts for one retrieval method:
- tfidf
- bow
- bm25
- boolean
"""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils import (
	build_bm25_index,
	load_csv_qa_rows_from_sources,
	load_intents,
	prepare_corpus,
	prepare_corpus_from_csv_rows,
	project_root,
	validate_method,
)


def train_and_save(
	method: str = "tfidf",
	threshold: float = 0.25,
	qa_sources: list[str] | None = None,
) -> None:
	"""Train selected retrieval method and persist artifacts to models/.

	`threshold` is stored in payload and reused during inference to
	decide fallback vs matched response.
	"""
	method = validate_method(method)
	root = project_root()
	models_dir = root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	intents = load_intents(root / "data" / "intents.json")
	corpus, examples = prepare_corpus(intents)

	# Optionally enrich training data with web-sourced Q/A CSV rows.
	resolved_sources = qa_sources or ["data/web_faq.csv", "data/topics"]
	csv_source_paths = [root / Path(source) for source in resolved_sources]
	csv_rows = load_csv_qa_rows_from_sources(csv_source_paths)
	csv_corpus, csv_examples = prepare_corpus_from_csv_rows(csv_rows)
	corpus.extend(csv_corpus)
	examples.extend(csv_examples)

	if not corpus:
		raise ValueError("No training patterns found in data/intents.json")

	payload = {"method": method, "examples": examples, "threshold": threshold}

	# Method-specific artifact building.
	if method == "tfidf":
		vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
		matrix = vectorizer.fit_transform(corpus)
		payload.update({"vectorizer": vectorizer, "matrix": matrix})
		vocabulary_words = vectorizer.get_feature_names_out().tolist()
	elif method == "bow":
		vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
		matrix = vectorizer.fit_transform(corpus)
		payload.update({"vectorizer": vectorizer, "matrix": matrix})
		vocabulary_words = vectorizer.get_feature_names_out().tolist()
	elif method == "bm25":
		doc_tokens = [doc.split() for doc in corpus]
		payload.update({"bm25_index": build_bm25_index(doc_tokens), "doc_tokens": doc_tokens})
		vocabulary_words = sorted({token for doc in doc_tokens for token in doc})
	else:
		doc_tokens = [doc.split() for doc in corpus]
		payload.update({"doc_tokens": doc_tokens})
		vocabulary_words = sorted({token for doc in doc_tokens for token in doc})

	joblib.dump(payload, models_dir / f"faq_index_{method}.joblib")
	# These files are shared helpers for inspection/debugging.
	joblib.dump(sorted({example["tag"] for example in examples}), models_dir / "classes.pkl")
	joblib.dump(vocabulary_words, models_dir / "words.pkl")

	print(
		f"Training complete for method '{method}'. Indexed {len(examples)} patterns "
		f"({len(csv_examples)} from CSV)."
	)
	print(f"Artifacts saved to: {models_dir}")


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for training configuration."""
	parser = argparse.ArgumentParser(description="Train FAQ chatbot retrieval index")
	parser.add_argument(
		"--method",
		default="tfidf",
		choices=["tfidf", "bow", "bm25", "boolean"],
		help="Retrieval method to train",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.25,
		help="Minimum score required to return a matched answer",
	)
	parser.add_argument(
		"--qa-sources",
		nargs="*",
		default=["data/web_faq.csv", "data/topics"],
		help="Q/A CSV file(s) or directory path(s) relative to project root",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train_and_save(method=args.method, threshold=args.threshold, qa_sources=args.qa_sources)
