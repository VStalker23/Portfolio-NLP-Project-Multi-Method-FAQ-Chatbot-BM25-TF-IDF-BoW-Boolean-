# 60-Second Demo Script

## 1) Open project and run training

```bash
python src/train.py --method bm25
```

Say:
- "This project is a beginner NLP FAQ chatbot with BM25, BoW, Boolean, and TF-IDF retrieval."
- "I use CSV datasets with real-life topics like sports, weather, food, politics, and news."

## 2) Show evaluation table

```bash
python src/evaluate.py --method all --out results/eval_metrics.csv
```

Say:
- "Here I compare methods with accuracy and fallback rate."
- "BM25 currently performs best on my expanded dataset."

## 3) Start chatbot UI

```bash
python src/chat.py --method bm25
```

In chat, type:

```text
/topics
/topic sports
How can I start running as a beginner?
/topic weather
How can I stay safe during heat waves?
/topic all
/list
/quit
```

Say:
- "The terminal UI supports topic filtering and command help."
- "When answers come from web-derived rows, the bot can show source URLs."

## 4) Show visual outputs

Open these files:
- `results/plots/wordcloud_questions.png`
- `results/plots/wordcloud_answers.png`
- `results/plots/wordcloud_topics.png`
- `results/plots/topic_question_answer_counts.png`
- `results/plots/evaluation_comparison.png`

Say:
- "I also generate portfolio visuals from dataset and evaluation outputs."

## Optional quick quality check

```bash
python src/smoke_test.py
```

Say:
- "Smoke test confirms key flows are working before I publish."
