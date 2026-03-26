"""
m_02_002_download_paul_graham_data.py

Downloads the Paul Graham essay file used throughout the Module 2 LlamaIndex
examples.

Usage:
    python m_02_002_download_paul_graham_data.py
    python m_02_002_download_paul_graham_data.py --data-dir ./data/paul_graham
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.request


URLS = [
    "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
    "https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt",
    "https://www.paulgraham.com/worked.html",
]


def download_file(output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to: {output_path}")
            return output_path
        except Exception as exc:
            last_error = exc
            print(f"Failed: {exc}")

    raise RuntimeError(f"All download sources failed. Last error: {last_error}")


def preview_file(path: pathlib.Path, lines: int = 10) -> None:
    print(f"\nPreview of {path}:")
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh):
            if idx >= lines:
                break
            print(line.rstrip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Paul Graham essay dataset.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory to store the essay file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = pathlib.Path(args.data_dir) / "paul_graham_essay.txt"
    download_file(output_path)
    preview_file(output_path)


if __name__ == "__main__":
    main()
