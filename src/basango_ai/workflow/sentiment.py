from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from prefect import flow, task

from basango_ai.core.utils import get_data_path
from basango_ai.core.constants import (
    TORCH_NUM_THREADS,
    TORCH_NUM_INTEROP_THREADS,
    BATCH_SIZE,
)
from basango_ai.models.sentiment.tabularisai import load_model


@task(name="annotate-sentiment", log_prints=True)
def annotate_sentiment() -> Path:
    destination = get_data_path("silver") / "dataset.csv"
    if not destination.exists():
        raise FileNotFoundError(f"Dataset not found at {destination}")

    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

    df = pl.read_csv(destination, infer_schema_length=10_000, ignore_errors=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    model = load_model()

    batches: list[pl.DataFrame] = []
    processed = 0

    print(f"Annotating dataset with sentiment from {destination} ({df.height} records)")
    for batch in df.iter_slices(n_rows=BATCH_SIZE):
        titles = batch.get_column("title").fill_null("").cast(pl.Utf8).to_list()
        bodies = batch.get_column("body").fill_null("").cast(pl.Utf8).to_list()

        title_labels = model.predict(titles)
        body_labels = model.predict(bodies)

        batches.append(
            batch.with_columns(
                [
                    pl.Series("title_sentiment", title_labels),
                    pl.Series("body_sentiment", body_labels),
                ]
            )
        )

        processed += batch.height
        print(f"Processed {processed} records")

    out_df = pl.concat(batches, how="vertical", rechunk=True)
    out_df.write_csv(tmp)
    tmp.replace(destination)

    print(f"Wrote dataset with sentiment to {destination} ({out_df.height} records)")
    return destination


@flow(name="sentiment-flow", log_prints=True)
def sentiment_flow() -> Path:
    return annotate_sentiment.submit().result()


if __name__ == "__main__":
    sentiment_flow()


__all__ = ["sentiment_flow"]
