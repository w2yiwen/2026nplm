from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional

logger = logging.getLogger(__name__)


def iter_jsonl_texts(
    files: Iterable[Path],
    text_key: str = "text",
    max_chars: Optional[int] = None,
) -> Generator[str, None, None]:
    total = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get(text_key)
                    if not t:
                        continue
                    text = str(t).replace("\r\n", "\n").strip()
                    if not text:
                        continue
                    if max_chars is not None:
                        remain = max_chars - total
                        if remain <= 0:
                            return
                        if len(text) > remain:
                            text = text[:remain]
                    total += len(text)
                    yield text
        except OSError as e:
            logger.warning("Skip unreadable file %s: %s", fp, e)


_TAG_RE = re.compile(r"^([^/]+)(?:/.*)?$")


def _strip_brown_token(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        return ""
    if tok.startswith("``") or tok.startswith("''"):
        return tok
    m = _TAG_RE.match(tok)
    return m.group(1) if m else tok


def iter_brown_texts(files: Iterable[Path], max_chars: Optional[int] = None) -> Generator[str, None, None]:
    total = 0
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("Skip unreadable file %s: %s", fp, e)
            continue
        lines: list[str] = []
        for line in raw.splitlines():
            parts = [_strip_brown_token(t) for t in line.split()]
            cleaned = " ".join(p for p in parts if p)
            if cleaned:
                lines.append(cleaned)
        paragraph = "\n".join(lines)
        if max_chars is not None:
            remain = max_chars - total
            if remain <= 0:
                return
            if len(paragraph) > remain:
                paragraph = paragraph[:remain]
        total += len(paragraph)
        if paragraph:
            yield paragraph


def iter_plain_texts(files: Iterable[Path], max_chars: Optional[int] = None) -> Generator[str, None, None]:
    total = 0
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("Skip unreadable file %s: %s", fp, e)
            continue
        text = raw.strip()
        if not text:
            continue
        if max_chars is not None:
            remain = max_chars - total
            if remain <= 0:
                return
            if len(text) > remain:
                text = text[:remain]
        total += len(text)
        yield text


def discover_files(root: Path, pattern: str, max_files: Optional[int]) -> list[Path]:
    if not root.exists():
        logger.warning("Data root does not exist: %s", root)
        return []
    paths = sorted(root.glob(pattern))
    if max_files is not None:
        paths = paths[: max_files]
    return [p for p in paths if p.is_file()]


def text_stream_for_dataset(
    *,
    data_root: Path,
    dataset: dict,
    max_files: Optional[int],
    max_chars: Optional[int],
) -> tuple[str, Callable[[], Generator[str, None, None]]]:
    rel = dataset["root"]
    fmt = dataset["format"]
    root = (data_root / rel).resolve()
    glob_pattern = dataset.get("glob_pattern", "**/*")
    files = discover_files(root, glob_pattern, max_files)

    def factory() -> Generator[str, None, None]:
        if fmt == "jsonl_glob":
            yield from iter_jsonl_texts(files, text_key=dataset.get("text_key", "text"), max_chars=max_chars)
        elif fmt == "brown_tagged":
            yield from iter_brown_texts(files, max_chars=max_chars)
        elif fmt == "plain_text":
            yield from iter_plain_texts(files, max_chars=max_chars)
        else:
            raise ValueError(f"unsupported dataset format: {fmt}")

    return dataset["id"], factory
