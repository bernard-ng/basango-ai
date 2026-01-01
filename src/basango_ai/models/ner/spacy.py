from __future__ import annotations

from typing import Iterable, List

import spacy

from basango_ai.core.types import ArticleRecord

LABELS = {"PER", "LOC", "ORG", "GPE", "MISC"}

class SpacyNerPredictor:
    def __init__(self) -> None:
        self.nlp = spacy.load("fr_core_news_lg")

    def predict(self, articles: Iterable[ArticleRecord]) -> List[dict]:
        articles = list(articles)
        texts = [article.content or article.title for article in articles]
        if not texts:
            return []

        docs = list(self.nlp.pipe(texts))
        results = []
        for article, doc in zip(articles, docs):
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
                if ent.label_ in LABELS
            ]
            results.append({"id": article.id, "entities": entities})
        return results


def load_model() -> SpacyNerPredictor:
    return SpacyNerPredictor()
