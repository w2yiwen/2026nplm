from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from model.base import RunContext, TrainPaths
from model.registry import ModelSpec


def save_stub_checkpoint(
    spec: ModelSpec,
    ctx: RunContext,
    paths: TrainPaths,
    hyperparams: Optional[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    hp = {**dict(spec.hyperparams), **dict(hyperparams or {})}
    ckpt_dir = ctx.outputs_dir / "checkpoints" / spec.checkpoint_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "checkpoint.pt"
    payload = {
        "model_id": spec.model_id,
        "stub": True,
        "hyperparams": hp,
        "vocab_path": str(paths.vocab_json),
        "train_txt": str(paths.train_txt),
        "val_txt": str(paths.val_txt),
    }
    torch.save(payload, ckpt)
    meta = {
        "checkpoint": str(ckpt),
        "val_ppl": float(hp.get("stub_val_ppl", 100.0)),
        "val_loss": float(hp.get("stub_val_loss", 4.605)),
        "status": "stub_complete",
    }
    return ckpt, meta


def train_stub_timing(meta: dict[str, Any], start: float) -> dict[str, Any]:
    out = dict(meta)
    out["elapsed_sec"] = round(time.perf_counter() - start, 3)
    return out


def stub_generate_lines(
    spec: ModelSpec,
    *,
    language: str,
    num_samples: int,
    prompts: Optional[list[str]],
) -> list[str]:
    base = list(prompts) if prompts else []
    if len(base) < num_samples:
        pad = "科学" if language.lower().startswith("zh") else "The"
        base.extend([pad] * (num_samples - len(base)))
    base = base[:num_samples]
    tag = spec.model_id
    return [f"{p} … [{tag}-stub:{language}]" for p in base]


def stub_evaluate_metrics(spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "val_ppl": 100.0,
        "val_loss": 4.605,
        "note": "stub metrics; replace with real evaluation after implementation",
    }
