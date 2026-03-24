from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return project_root() / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    root: Path
    seed: int
    device: str
    data_root: Path
    processed_dir: Path
    outputs_dir: Path
    default_model: str
    train_split: float
    max_vocab_size: int
    preprocess: dict[str, Any]
    datasets: list[dict[str, Any]]
    models: dict[str, Any]
    generation: dict[str, Any]

    @staticmethod
    def load(path: Optional[Path] = None) -> "AppConfig":
        root = project_root()
        cfg_path = path or default_config_path()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        pp = raw.get("preprocess") or {}
        return AppConfig(
            root=root,
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            data_root=(root / raw["data_root"]).resolve(),
            processed_dir=(root / raw["processed_dir"]).resolve(),
            outputs_dir=(root / raw["outputs_dir"]).resolve(),
            default_model=str(raw["default_model"]),
            train_split=float(raw["train_split"]),
            max_vocab_size=int(raw["max_vocab_size"]),
            preprocess=dict(pp),
            datasets=list(raw.get("datasets") or []),
            models=dict(raw.get("models") or {}),
            generation=dict(raw.get("generation") or {}),
        )

    @property
    def preprocess_max_files_per_dataset(self) -> Optional[int]:
        return self.preprocess.get("max_files_per_dataset")

    @property
    def preprocess_max_total_chars(self) -> Optional[int]:
        return self.preprocess.get("max_total_chars")

    @property
    def preprocess_max_chars_per_dataset(self) -> Optional[int]:
        return self.preprocess.get("max_chars_per_dataset")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(preference: str) -> torch.device:
    pref = (preference or "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref)
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            idx = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if idx:
                logger.info("Using CUDA (CUDA_VISIBLE_DEVICES=%s).", idx)
            return torch.device("cuda:0")
        logger.info("CUDA not available; using CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:  # pragma: no cover
        name = "unknown"
    return f"{device} ({name})"
