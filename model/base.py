from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from model.registry import ModelSpec


@dataclass(frozen=True)
class TrainPaths:
    processed_dir: Path
    train_txt: Path
    val_txt: Path
    vocab_json: Path


@dataclass(frozen=True)
class RunContext:
    """Shared runtime objects passed into model entrypoints."""

    device: torch.device
    seed: int
    outputs_dir: Path


class LanguageModel(ABC):
    """Contract used by ``train`` / ``generate`` / ``evaluate`` scripts."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abstractmethod
    def train(
        self,
        ctx: RunContext,
        paths: TrainPaths,
        hyperparams: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Train (or fit) and write checkpoints under ``ctx.outputs_dir``."""

    @abstractmethod
    def generate(
        self,
        ctx: RunContext,
        checkpoint: Path,
        *,
        language: str,
        num_samples: int,
        max_new_tokens: int,
        prompts: Optional[list[str]] = None,
    ) -> list[str]:
        """Produce ``num_samples`` lines of text."""

    @abstractmethod
    def evaluate(
        self,
        ctx: RunContext,
        checkpoint: Path,
        paths: TrainPaths,
    ) -> dict[str, Any]:
        """Return scalar metrics (e.g. validation perplexity)."""

    def default_checkpoint(self, ctx: RunContext) -> Path:
        sub = self.spec.checkpoint_subdir
        return ctx.outputs_dir / "checkpoints" / sub / "checkpoint.pt"
