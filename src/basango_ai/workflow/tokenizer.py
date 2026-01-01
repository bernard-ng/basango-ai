from __future__ import annotations

from pathlib import Path

import polars as pl
from prefect import flow, task

from basango_ai.core.utils import get_data_path
from basango_ai.models.tokenizer.tiktoken import load_model


BATCH_SIZE = 128


@task(name="count-tokens", log_prints=True)
def count_tokens() -> Path:
    destination = get_data_path("silver") / "dataset.csv"
    if not destination.exists():
        raise FileNotFoundError(f"Dataset not found at {destination}")

    df = pl.read_csv(destination, infer_schema_length=10_000, ignore_errors=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    model = load_model()

    batches: list[pl.DataFrame] = []
    processed = 0

    print(f"Counting tokens from {destination} ({df.height} records)...")
    for batch in df.iter_slices(n_rows=BATCH_SIZE):
        titles = batch.get_column("title").fill_null("").cast(pl.Utf8).to_list()
        bodies = batch.get_column("body").fill_null("").cast(pl.Utf8).to_list()

        title_tokens = [model.count(title) for title in titles]
        body_tokens = [model.count(body) for body in bodies]

        batches.append(
            batch.with_columns(
                [
                    pl.Series("title_tokens", title_tokens),
                    pl.Series("body_tokens", body_tokens),
                ]
            )
        )

        processed += batch.height
        print(f"Processed {processed} records")

    out_df = pl.concat(batches, how="vertical", rechunk=True)
    out_df.write_csv(tmp)
    tmp.replace(destination)

    print(f"Wrote dataset with token counts to {destination} ({out_df.height} records)")
    return destination


@flow(name="tokenizer-flow", log_prints=True)
def tokenizer_flow() -> Path:
    return count_tokens.submit().result()


if __name__ == "__main__":
    tokenizer_flow()


__all__ = ["tokenizer_flow"]
