from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from model.base import LanguageModel, RunContext, TrainPaths
from model.registry import ModelSpec

# 句首/句尾占位（单字符，参与统计与生成）
BOS = "\x00"
EOS = "\x01"


def _merge_hp(spec: ModelSpec, hyperparams: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    out = dict(spec.hyperparams)
    if hyperparams:
        out.update(dict(hyperparams))
    return out


def tokenize_chars(text: str) -> list[str]:
    """按字切分（与课程示例一致，中文/英文均以字符为 token）。"""
    return list(text)


def count_ngrams(lines: list[str], n: int) -> tuple[dict[tuple[str, ...], Counter[str]], set[str]]:
    """
    统计 n-gram：prefix 为长度 n-1 的元组，下一个字符的频次。
    每行前后添加 (n-1) 个 BOS 与 1 个 EOS，便于边界与困惑度计算一致。
    """
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    vocab: set[str] = {BOS, EOS}

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        tokens = [BOS] * (n - 1) + tokenize_chars(raw) + [EOS]
        vocab.update(tokens)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            prefix = tuple(tokens[i : i + n - 1])
            nxt = tokens[i + n - 1]
            counts[prefix][nxt] += 1

    return dict(counts), vocab


def smoothed_prob(
    prefix: tuple[str, ...],
    nxt: str,
    counts: dict[tuple[str, ...], Counter[str]],
    vocab: list[str],
    alpha: float,
) -> float:
    """Laplace 平滑：P(nxt | prefix) = (c + alpha) / (sum + alpha * V)."""
    v = len(vocab)
    ctr = counts.get(prefix)
    if ctr is None:
        return 1.0 / v if v > 0 else 1.0
    total = sum(ctr.values())
    c = ctr.get(nxt, 0)
    return (c + alpha) / (total + alpha * v)


def next_token_greedy(
    prefix: tuple[str, ...],
    counts: dict[tuple[str, ...], Counter[str]],
    vocab: list[str],
    alpha: float,
) -> str:
    """在当前前缀下取概率最大的下一字符。"""
    best: Optional[str] = None
    best_p = -1.0
    for w in vocab:
        p = smoothed_prob(prefix, w, counts, vocab, alpha)
        if p > best_p:
            best_p = p
            best = w
    assert best is not None
    return best


def line_logprob(
    text: str,
    n: int,
    counts: dict[tuple[str, ...], Counter[str]],
    vocab: list[str],
    alpha: float,
) -> tuple[float, int]:
    """单句对数似然（所有 n-gram 位置），返回 (log_prob_sum, num_positions)。"""
    tokens = [BOS] * (n - 1) + tokenize_chars(text.strip()) + [EOS]
    if len(tokens) < n:
        return 0.0, 0
    lp = 0.0
    num = 0
    for i in range(len(tokens) - n + 1):
        prefix = tuple(tokens[i : i + n - 1])
        nxt = tokens[i + n - 1]
        p = smoothed_prob(prefix, nxt, counts, vocab, alpha)
        lp += math.log(max(p, 1e-300))
        num += 1
    return lp, num


def save_checkpoint(
    path: Path,
    *,
    n: int,
    smoothing: float,
    counts: dict[tuple[str, ...], Counter[str]],
    vocab: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial_counts = {tuple(k): dict(v) for k, v in counts.items()}
    payload = {
        "kind": "ngram_char",
        "n": n,
        "smoothing": smoothing,
        "counts": serial_counts,
        "vocab": vocab,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _build_counts_from_payload(payload: dict[str, Any]) -> dict[tuple[str, ...], Counter[str]]:
    out: dict[tuple[str, ...], Counter[str]] = {}
    for k, v in payload["counts"].items():
        prefix = tuple(k) if not isinstance(k, tuple) else k
        out[prefix] = Counter(v)
    return out


class NGramLM(LanguageModel):
    """
    字符级 N-gram 语言模型（与课程中 bigram 思路一致，可推广到任意 n）。
    使用 Laplace 平滑；训练从 ``train.txt`` 估计频次；checkpoint 为 PyTorch 保存的字典。
    """

    def train(
        self,
        ctx: RunContext,
        paths: TrainPaths,
        hyperparams: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        _ = ctx
        hp = _merge_hp(self.spec, hyperparams)
        n = int(hp.get("n", 3))
        alpha = float(hp.get("smoothing", 0.01))
        if n < 1:
            raise ValueError("n must be >= 1")

        t0 = time.perf_counter()
        if not paths.train_txt.exists():
            raise FileNotFoundError(paths.train_txt)

        lines = paths.train_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        counts, vocab_set = count_ngrams(lines, n)
        vocab = sorted(vocab_set)

        ckpt_dir = ctx.outputs_dir / "checkpoints" / self.spec.checkpoint_subdir
        ckpt_path = ckpt_dir / "checkpoint.pt"
        save_checkpoint(ckpt_path, n=n, smoothing=alpha, counts=counts, vocab=vocab)

        # 训练集上的平均负对数似然（与验证指标口径一致，仅作参考）
        total_lp, total_n = 0.0, 0
        for line in lines:
            if not line.strip():
                continue
            lp, num = line_logprob(line, n, counts, vocab, alpha)
            total_lp += lp
            total_n += num
        train_loss = -total_lp / total_n if total_n else float("nan")
        train_ppl = math.exp(train_loss) if total_n else float("nan")

        elapsed = time.perf_counter() - t0
        return {
            "checkpoint": str(ckpt_path),
            "n": n,
            "smoothing": alpha,
            "vocab_size": len(vocab),
            "n_prefixes": len(counts),
            "train_lines": len([x for x in lines if x.strip()]),
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "status": "ok",
            "elapsed_sec": round(elapsed, 3),
            "note": "train_loss / train_ppl on training lines (same n-gram formula as eval)",
        }

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
        _ = ctx, language
        payload = load_checkpoint(checkpoint)
        n = int(payload["n"])
        alpha = float(payload["smoothing"])
        counts = _build_counts_from_payload(payload)
        vocab = list(payload["vocab"])

        seeds = list(prompts) if prompts else [" "]
        while len(seeds) < num_samples:
            seeds.append(seeds[len(seeds) % max(len(seeds), 1)])
        seeds = seeds[:num_samples]

        results: list[str] = []
        for prompt in seeds:
            # 前缀不足 n-1 时左侧用 BOS 补齐，与训练时一致
            body = list(prompt)
            tokens = [BOS] * (n - 1) + body
            generated = 0
            while generated < max_new_tokens:
                if n <= 1:
                    prefix = tuple()
                else:
                    prefix = tuple(tokens[-(n - 1) :])
                nxt = next_token_greedy(prefix, counts, vocab, alpha)
                tokens.append(nxt)
                generated += 1
                if nxt == EOS:
                    break
            # 去掉前导 BOS 与尾部 EOS，只保留可读文本
            out_chars = tokens[n - 1 :]
            if out_chars and out_chars[-1] == EOS:
                out_chars = out_chars[:-1]
            results.append("".join(out_chars))
        return results

    def evaluate(
        self,
        ctx: RunContext,
        checkpoint: Path,
        paths: TrainPaths,
    ) -> dict[str, Any]:
        _ = ctx
        payload = load_checkpoint(checkpoint)
        n = int(payload["n"])
        alpha = float(payload["smoothing"])
        counts = _build_counts_from_payload(payload)
        vocab = list(payload["vocab"])

        if not paths.val_txt.exists():
            raise FileNotFoundError(paths.val_txt)

        lines = paths.val_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        total_lp, total_n = 0.0, 0
        for line in lines:
            if not line.strip():
                continue
            lp, num = line_logprob(line, n, counts, vocab, alpha)
            total_lp += lp
            total_n += num

        if total_n == 0:
            return {
                "model_id": self.spec.model_id,
                "val_loss": float("nan"),
                "val_ppl": float("nan"),
                "note": "no n-gram positions in val.txt",
            }

        val_loss = -total_lp / total_n
        val_ppl = math.exp(val_loss)
        return {
            "model_id": self.spec.model_id,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "n": n,
            "smoothing": alpha,
            "num_tokens_scored": total_n,
        }
