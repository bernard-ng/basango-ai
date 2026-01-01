from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from urllib.parse import urlsplit


import httpx
from basango_ai.core.utils import get_data_path
from prefect import flow, task


DATASETS = [
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/actualite.cd.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/beto.cd.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/mediacongo.net.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/radiookapi.net.csv?download=true",
]


async def _download(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> Path:
    filename = Path(urlsplit(url).path).name or "dataset.csv"
    destination = get_data_path("bronze") / filename

    if destination.exists():
        print(f"Skipping download (exists): {destination}")
        return destination

    async with semaphore:
        response = await client.get(url)
        response.raise_for_status()
        destination.write_bytes(response.content)
        print(f"Downloaded {url} -> {destination}")

    return destination


@task(name="download", retries=2, retry_delay_seconds=10, log_prints=True)
async def download() -> list[Path]:
    print("Starting dataset downloads...")
    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
        tasks = [_download(client, url, semaphore) for url in DATASETS]
        return await asyncio.gather(*tasks)


@task(name="combine", log_prints=True)
def combine(files: list[Path]) -> Path:
    output_path = get_data_path("silver") / "dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Combining {len(files)} files into {output_path}...")
    writer = None
    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        for file_path in files:
            with file_path.open("r", newline="", encoding="utf-8") as in_file:
                reader = csv.DictReader(in_file)
                if reader.fieldnames is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
                    writer.writeheader()
                elif writer.fieldnames != reader.fieldnames:
                    raise ValueError(
                        f"Field mismatch between files; expected {writer.fieldnames}, "
                        f"got {reader.fieldnames} from {file_path}"
                    )
                writer.writerows(reader)

    print(f"Wrote combined dataset to {output_path}")
    return output_path


@flow(name="dataset-flow", log_prints=True)
def dataset_flow() -> None:
    downloaded_files = download.submit()
    combine.submit(downloaded_files).result()


if __name__ == "__main__":
    dataset_flow.serve()

__all__ = ["dataset_flow"]
