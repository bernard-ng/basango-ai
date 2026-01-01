from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import polars as pl
from prefect import flow, task

from basango_ai.core.utils import get_data_path
from basango_ai.core.constants import CONCURRENCY, DATASETS_URL, CHUNK_SIZE


async def _download_one(
    client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore
) -> Path:
    filename = Path(urlsplit(url).path).name or "dataset.csv"
    destination = get_data_path("bronze") / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping -> {destination}")
        return destination

    tmp = destination.with_suffix(destination.suffix + ".tmp")

    async with sem:
        print(f"Downloading {filename}")
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                    f.write(chunk)
        tmp.replace(destination)
        print(f"Downloaded {filename} ({destination.stat().st_size} bytes)")
        return destination


@task(name="download", retries=2, retry_delay_seconds=10, log_prints=True)
async def download() -> list[Path]:
    print("Starting dataset downloads...")

    limits = httpx.Limits(
        max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY
    )
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        limits=limits,
        headers={"User-Agent": "basango-ai/1.0"},
    ) as client:
        tasks = [_download_one(client, url, sem) for url in DATASETS_URL]
        return await asyncio.gather(*tasks)


@task(name="combine", log_prints=True)
def combine(files: list[Path]) -> Path:
    destination = get_data_path("silver") / "dataset.csv"
    tmp = destination.with_suffix(destination.suffix + ".tmp")

    print(f"Combining {len(files)} files into {destination}...")
    scans: list[pl.LazyFrame] = []
    for f in files:
        scans.append(pl.scan_csv(f, infer_schema_length=10_000, ignore_errors=True))

    lf = pl.concat(scans, how="vertical", rechunk=False)
    lf.collect(streaming=True).write_csv(tmp)
    tmp.replace(destination)

    print(f"Wrote combined dataset to {destination}")
    return destination


@flow(name="dataset-flow", log_prints=True)
def dataset_flow() -> Path:
    downloaded_files = download.submit().result()
    return combine.submit(downloaded_files).result()


if __name__ == "__main__":
    dataset_flow()

__all__ = ["dataset_flow"]
