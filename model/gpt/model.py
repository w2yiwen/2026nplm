from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from model.base import LanguageModel, RunContext, TrainPaths
from model.stub_support import (
    save_stub_checkpoint,
    stub_evaluate_metrics,
    stub_generate_lines,
    train_stub_timing,
)
from model.registry import ModelSpec


class GPTLM(LanguageModel):
    """Placeholder: decoder-only Transformer (GPT-style) LM here."""

    def train(
        self,
        ctx: RunContext,
        paths: TrainPaths,
        hyperparams: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        _, meta = save_stub_checkpoint(self.spec, ctx, paths, hyperparams)
        return train_stub_timing(meta, t0)

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
        _ = (ctx, checkpoint, max_new_tokens)
        return stub_generate_lines(self.spec, language=language, num_samples=num_samples, prompts=prompts)

    def evaluate(
        self,
        ctx: RunContext,
        checkpoint: Path,
        paths: TrainPaths,
    ) -> dict[str, Any]:
        _ = (ctx, checkpoint, paths)
        return stub_evaluate_metrics(self.spec)
