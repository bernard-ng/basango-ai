from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from basango_ai.core.types import ArticleRecord


SENTIMENTS = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}


class TabularisaiSentimentPredictor:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
        self.model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")
        self.model.to(self.device)

    def predict(self, articles: Iterable[ArticleRecord]) -> List[dict]:
        article_list = list(articles)
        texts = [article.content or article.title for article in article_list]
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = torch.argmax(probabilities, dim=-1).tolist()
            scores = probabilities.max(dim=-1).values.tolist()

        return [
            {
                "id": article.id,
                "label": SENTIMENTS.get(label_idx, "Unknown"),
                "score": float(score),
            }
            for article, label_idx, score in zip(article_list, labels, scores)
        ]


def load_model() -> TabularisaiSentimentPredictor:
    return TabularisaiSentimentPredictor()
