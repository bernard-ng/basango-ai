from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from basango_ai.core.utils import get_data_path
from basango_ai.models.ner.spacy import LABELS, load_model
from prefect import flow, task

DATASET_FILENAME = "dataset.csv"
TITLE_ENTITIES_FIELD = "title_entities"
BODY_ENTITIES_FIELD = "body_entities"
BATCH_SIZE = 64


def _iter_batches(
        iterable: Iterable[dict[str, str]], batch_size: int
) -> Iterable[list[dict[str, str]]]:
    batch: list[dict[str, str]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _extract_entities(nlp, texts: list[str]) -> list[list[dict[str, str]]]:
    docs = nlp.pipe(texts)
    return [
        [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in LABELS]
        for doc in docs
    ]


@task(name="annotate", log_prints=True)
def annotate() -> Path:
    source_path = get_data_path("silver") / DATASET_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found at {source_path}")

    destination = source_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")

    model = load_model()
    nlp = model.nlp

    total_rows = 0
    with (
        source_path.open("r", newline="", encoding="utf-8") as src,
        tmp_destination.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        for extra in (TITLE_ENTITIES_FIELD, BODY_ENTITIES_FIELD):
            if extra not in fieldnames:
                fieldnames.append(extra)

        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Annotating dataset with metadata from {source_path}...")
        for batch in _iter_batches(reader, BATCH_SIZE):
            titles = [row.get("title", "") or "" for row in batch]
            bodies = [row.get("body", "") or "" for row in batch]

            title_entities = _extract_entities(nlp, titles)
            body_entities = _extract_entities(nlp, bodies)

            print(f"Processed {total_rows + len(batch)} records...")
            for row, title_ents, body_ents in zip(batch, title_entities, body_entities):
                row[TITLE_ENTITIES_FIELD] = json.dumps(title_ents, ensure_ascii=False)
                row[BODY_ENTITIES_FIELD] = json.dumps(body_ents, ensure_ascii=False)
                writer.writerow(row)
                total_rows += 1

    tmp_destination.replace(destination)
    print(f"Wrote dataset with metadata to {destination} ({total_rows} records)")
    return destination


@flow(name="ner-flow", log_prints=True)
def ner_flow() -> None:
    annotate.submit().result()


if __name__ == "__main__":
    ner_flow()


__all__ = ["ner_flow"]
