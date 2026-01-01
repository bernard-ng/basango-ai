from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from prefect import flow, task

from basango_ai.core.utils import get_data_path
from basango_ai.models.ner.spacy import load_model


BATCH_SIZE = 128


@task(name="annotate-ner", log_prints=True)
def annotate_ner() -> Path:
    destination = get_data_path("silver") / "dataset.csv"
    if not destination.exists():
        raise FileNotFoundError(f"Dataset not found at {destination}")

    df = pl.read_csv(destination, infer_schema_length=10_000, ignore_errors=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    model = load_model()

    batches: list[pl.DataFrame] = []
    processed = 0

    print(f"Annotating dataset with NER from {destination} ({df.height} records)...")
    for batch in df.iter_slices(n_rows=BATCH_SIZE):
        titles = batch.get_column("title").fill_null("").cast(pl.Utf8).to_list()
        bodies = batch.get_column("body").fill_null("").cast(pl.Utf8).to_list()

        title_entities = [
            json.dumps(ents, ensure_ascii=False) for ents in model.predict(titles)
        ]
        body_entities = [
            json.dumps(ents, ensure_ascii=False) for ents in model.predict(bodies)
        ]

        batches.append(
            batch.with_columns(
                [
                    pl.Series("title_entities", title_entities),
                    pl.Series("body_entities", body_entities),
                ]
            )
        )

        processed += batch.height
        print(f"Processed {processed} records")

    out_df = pl.concat(batches, how="vertical", rechunk=True)
    out_df.write_csv(tmp)
    tmp.replace(destination)

    print(f"Wrote dataset with NER to {destination} ({out_df.height} records)")
    return destination


@flow(name="ner-flow", log_prints=True)
def ner_flow() -> Path:
    return annotate_ner.submit().result()


if __name__ == "__main__":
    ner_flow()


__all__ = ["ner_flow"]
