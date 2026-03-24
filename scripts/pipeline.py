from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator, Optional

from tqdm import tqdm

from scripts.common import AppConfig, ensure_dir
from scripts.io_readers import text_stream_for_dataset
from scripts.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_preprocess(
    project: AppConfig,
    *,
    max_files_per_dataset: Optional[int] = None,
    max_total_chars: Optional[int] = None,
    max_chars_per_dataset: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    rng = random.Random(seed if seed is not None else project.seed)
    data_root = project.data_root
    out_dir = ensure_dir(project.processed_dir)
    corpus_path = out_dir / "corpus.jsonl"
    meta_path = out_dir / "meta.json"

    manifest = {"datasets": project.datasets}

    max_files = max_files_per_dataset
    if max_files is None:
        max_files = project.preprocess_max_files_per_dataset
    max_chars = max_total_chars
    if max_chars is None:
        max_chars = project.preprocess_max_total_chars

    per_ds_cap = max_chars_per_dataset
    if per_ds_cap is None:
        per_ds_cap = project.preprocess_max_chars_per_dataset

    documents: list[dict[str, str]] = []
    stats: dict[str, Any] = {"datasets": {}, "total_docs": 0, "total_chars": 0}

    total_chars = 0
    for ds in manifest["datasets"]:
        ds_id, factory = text_stream_for_dataset(
            data_root=data_root,
            dataset=ds,
            max_files=max_files,
            max_chars=None,
        )
        ds_chars = 0
        ds_docs = 0
        stream = factory()
        for text in tqdm(stream, desc=f"reading {ds_id}", unit="doc"):
            if max_chars is not None and total_chars >= max_chars:
                break
            if per_ds_cap is not None and ds_chars >= per_ds_cap:
                break
            lang = ds.get("language", "und")
            if max_chars is not None:
                remain = max_chars - total_chars
                if len(text) > remain:
                    text = text[:remain]
            if per_ds_cap is not None:
                remain_ds = per_ds_cap - ds_chars
                if len(text) > remain_ds:
                    text = text[:remain_ds]
            documents.append({"lang": lang, "text": text, "source": ds_id})
            ds_docs += 1
            c = len(text)
            ds_chars += c
            total_chars += c
            if max_chars is not None and total_chars >= max_chars:
                break
            if per_ds_cap is not None and ds_chars >= per_ds_cap:
                break
        stats["datasets"][ds_id] = {"docs": ds_docs, "chars": ds_chars}

    if not documents:
        raise ValueError(
            "No text collected — check data paths under data_root, or relax "
            "--max-files-per-dataset / --max-total-chars limits."
        )

    rng.shuffle(documents)
    n = len(documents)
    split_at = int(n * project.train_split)
    train_docs = documents[:split_at]
    val_docs = documents[split_at:]

    with corpus_path.open("w", encoding="utf-8") as f:
        for row in documents:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    _write_plain_corpus(train_docs, train_txt)
    _write_plain_corpus(val_docs, val_txt)

    def train_text_iter() -> Iterator[str]:
        for d in train_docs:
            yield d["text"]

    tokenizer = CharTokenizer.build_from_text(iter(train_text_iter()), project.max_vocab_size)
    tokenizer.save(out_dir / "vocab.json")

    stats["total_docs"] = n
    stats["total_chars"] = sum(len(d["text"]) for d in documents)
    stats["train_docs"] = len(train_docs)
    stats["val_docs"] = len(val_docs)
    stats["vocab_size"] = tokenizer.vocab_size
    stats["corpus_jsonl"] = str(corpus_path)
    stats["train_txt"] = str(train_txt)
    stats["val_txt"] = str(val_txt)
    _write_json(meta_path, stats)
    logger.info(
        "Preprocess done: %d docs, %d chars, vocab_size=%d",
        n,
        stats["total_chars"],
        tokenizer.vocab_size,
    )
    return stats


def _write_plain_corpus(docs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(d["text"].replace("\n", " ").strip() + "\n")
