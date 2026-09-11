import argparse
import json
import os
import numpy as np

from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator
from huggingface_hub import HfApi, hf_hub_download
from pyarrow import parquet

from common.paths import DATASETS_DIR
from data.tokenizer import GPT2Tokenizer


_tokenizer: GPT2Tokenizer | None = None

def _init_worker() -> None:
    global _tokenizer
    _tokenizer = GPT2Tokenizer()

def _tokenize(text: str) -> np.ndarray:
    assert _tokenizer is not None, "Tokenizer must be initialized before tokenizing"
    return np.asarray(_tokenizer.encode(text) + [_tokenizer.eot_id], dtype=np.uint16)


def write_splits(
    docs: Iterable[str], 
    out_dir: Path, 
    *, 
    val_tokens: int, 
    num_workers: int, 
    source: str, 
    shards: list[str]
) -> dict:
    """
    Tokenizes docs in order and writes document-aligned val and train in uint16 + meta.json.
    Returns: metadata dict
    
    Note: val boundary rounds up to a whole doc.
    """

    if val_tokens <= 0:
        raise ValueError(f"val_tokens must be > 0, received: {val_tokens}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, received: {num_workers}")

    example_tokenizer = GPT2Tokenizer()
    if example_tokenizer.vocab_size > 2**16:
        raise ValueError(f"vocab_size must be <= 2**16 to fit uint16, received: {example_tokenizer.vocab_size}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "val.bin", "wb") as val_file:
        with open(out_dir / "train.bin", "wb") as train_file:
            val_count, train_count, val_docs, train_docs = 0, 0, 0, 0
            with Pool(num_workers, initializer=_init_worker) as pool:
                for arr in pool.imap(_tokenize, docs, chunksize=64):
                    if val_count < val_tokens:
                        val_file.write(arr.tobytes())
                        val_count += len(arr)
                        val_docs += 1
                    else:
                        train_file.write(arr.tobytes())
                        train_count += len(arr)
                        train_docs += 1

    meta = {
        "source": source,
        "shards": shards,
        "vocab_size": example_tokenizer.vocab_size,
        "eot_id": example_tokenizer.eot_id,
        "dtype": "uint16",
        "val_tokens": val_count,
        "val_docs": val_docs,
        "train_tokens": train_count,
        "train_docs": train_docs,
        "tokenizer": "gpt2",
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def _download_shards(cache_dir: Path | None, max_shards: int | None) -> list[Path]:
    all_repo_files = HfApi().list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
    all_valid_files = []
    for fname in all_repo_files:
        if fname.startswith("sample/10BT/") and fname.endswith(".parquet"):
            all_valid_files.append(fname)

    all_valid_files.sort()
    files = all_valid_files[:max_shards]
    downloaded_paths = []
    for fname in files:
        downloaded_paths.append(Path(hf_hub_download("HuggingFaceFW/fineweb-edu", fname, repo_type="dataset", cache_dir=cache_dir)))

    return downloaded_paths


def _iter_documents(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
            yield from batch.column("text").to_pylist()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parses the prepare-fineweb-edu command line; None means sys.argv."""
    parser = argparse.ArgumentParser(description="Tokenize fineweb-edu sample-10BT into uint16 train/val files.")
    parser.add_argument("--out-dir", type=Path, default=DATASETS_DIR / "fineweb-edu-10bt", help="directory for train.bin, val.bin and meta.json")
    parser.add_argument("--val-tokens", type=int, default=100_000_000, help="minimum number of tokens routed to val.bin, rounded up to a whole document")
    parser.add_argument("--max-docs", type=int, default=None, help="stop after this many documents (smoke runs)")
    parser.add_argument("--max-shards", type=int, default=None, help="download and read only the first N parquet shards (smoke runs)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="tokenizer worker processes")
    parser.add_argument("--cache-dir", type=Path, default=None, help="huggingface_hub cache directory; default is the hub's own")
    args = parser.parse_args(argv)

    for name in ("max_docs", "max_shards"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0, received: {value}")
    if args.num_workers is None or args.num_workers <= 0:
        parser.error(f"--num-workers must be > 0, received: {args.num_workers}")

    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    paths = _download_shards(args.cache_dir, args.max_shards)
    docs = _iter_documents(paths)
    if args.max_docs is not None:
        docs = islice(docs, args.max_docs)

    meta = write_splits(
        docs, 
        args.out_dir,
        val_tokens=args.val_tokens,
        num_workers=args.num_workers,
        source="HuggingFaceFW/fineweb-edu sample-10BT",
        shards=[p.name for p in paths],
    )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__": 
    main()
