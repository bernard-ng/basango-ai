from __future__ import annotations

from typing import List

import spacy

from basango_ai.core.constants import CONCURRENCY

LABELS = {"PER", "LOC", "ORG", "GPE", "MISC"}


class SpacyNerPredictor:
    def __init__(self) -> None:
        self.nlp = spacy.load("fr_core_news_lg")

    def predict(self, texts: List[str]) -> List[dict]:
        docs = self.nlp.pipe(texts, batch_size=256, n_process=CONCURRENCY)
        return [
            [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
                if ent.label_ in LABELS
            ]
            for doc in docs
        ]


def load_model() -> SpacyNerPredictor:
    return SpacyNerPredictor()
