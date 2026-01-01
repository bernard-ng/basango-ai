from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            "tabularisai/multilingual-sentiment-analysis"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "tabularisai/multilingual-sentiment-analysis"
        )
        self.model.to(self.device)

    def predict(self, texts: List[str]) -> List[dict]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            labels = outputs.logits.argmax(dim=-1).tolist()

        return [SENTIMENTS.get(i, "unknown") for i in labels]


def load_model() -> TabularisaiSentimentPredictor:
    return TabularisaiSentimentPredictor()
