# GitHub Project Description (Ready to Paste)

## Repository short description
Beginner NLP FAQ chatbot with BM25/BoW/Boolean/TF-IDF retrieval, topic-based terminal UI, evaluation, and visualization pipeline.

## About this project
This is a portfolio-ready beginner NLP project that implements a retrieval-based FAQ chatbot from scratch.

### Key features
- 4 retrieval methods: **BM25**, **BoW**, **Boolean**, **TF-IDF**
- Friendly terminal chat UI with commands (`/help`, `/topics`, `/topic`, `/list`, `/clear`, `/quit`)
- Multi-source CSV ingestion for real-life topics (sports, weather, food, politics, news, technology, health)
- Method comparison with evaluation metrics (accuracy and fallback rate)
- Visualization pipeline (wordclouds + charts)
- Source attribution for web-derived Q/A rows

### Tech stack
Python, scikit-learn, numpy, joblib, matplotlib, wordcloud

### Why this project matters
- Demonstrates end-to-end NLP workflow: **data -> preprocessing -> indexing -> retrieval -> evaluation -> visualization**
- Shows practical software engineering basics: modular code, CLI UX, reproducible scripts, smoke tests, and documentation

### Current best method on this dataset
BM25 achieves the strongest accuracy/fallback balance in the latest evaluation run.

### Run locally
```bash
pip install -r requirements.txt
python src/train.py --method bm25
python src/chat.py --method bm25
```
