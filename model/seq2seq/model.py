from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.base import LanguageModel, RunContext, TrainPaths
from model.registry import ModelSpec
from scripts.tokenizer import CharTokenizer


def _merge_hp(
    spec: ModelSpec, hyperparams: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    hp = dict(spec.hyperparams)
    if hyperparams:
        hp.update(dict(hyperparams))
    return hp


def _read_non_empty_lines(path: Path, max_lines: Optional[int]) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            lines.append(text)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


@dataclass(frozen=True)
class SeqExample:
    src: list[int]
    dec_in: list[int]
    dec_tgt: list[int]


class SeqDataset(Dataset[SeqExample]):
    def __init__(
        self,
        lines: Iterable[str],
        tokenizer: CharTokenizer,
        *,
        max_seq_len: int,
    ) -> None:
        self.items: list[SeqExample] = []
        max_body = max(1, int(max_seq_len))
        bos = tokenizer.bos_id
        eos = tokenizer.eos_id
        for line in lines:
            body = tokenizer.encode(line)[:max_body]
            if not body:
                continue
            src = [bos] + body + [eos]
            dec_in = [bos] + body
            dec_tgt = body + [eos]
            self.items.append(SeqExample(src=src, dec_in=dec_in, dec_tgt=dec_tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SeqExample:
        return self.items[idx]


def _collate_batch(batch: list[SeqExample], pad_id: int) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    src_lens = torch.tensor([len(x.src) for x in batch], dtype=torch.long)
    max_src = int(src_lens.max().item())
    max_dec = max(len(x.dec_in) for x in batch)

    src = torch.full((bsz, max_src), pad_id, dtype=torch.long)
    dec_in = torch.full((bsz, max_dec), pad_id, dtype=torch.long)
    dec_tgt = torch.full((bsz, max_dec), pad_id, dtype=torch.long)

    for i, ex in enumerate(batch):
        src[i, : len(ex.src)] = torch.tensor(ex.src, dtype=torch.long)
        dec_in[i, : len(ex.dec_in)] = torch.tensor(ex.dec_in, dtype=torch.long)
        dec_tgt[i, : len(ex.dec_tgt)] = torch.tensor(ex.dec_tgt, dtype=torch.long)

    return {
        "src": src,
        "src_lens": src_lens,
        "dec_in": dec_in,
        "dec_tgt": dec_tgt,
    }


class Seq2SeqNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        real_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.decoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src: torch.Tensor, src_lens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return hidden

    def decode(self, dec_in: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(dec_in)
        dec_out, _ = self.decoder(emb, hidden)
        return self.output(dec_out)

    def forward(
        self, src: torch.Tensor, src_lens: torch.Tensor, dec_in: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encode(src, src_lens)
        return self.decode(dec_in, hidden)


def _build_model(
    tokenizer: CharTokenizer,
    *,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> Seq2SeqNet:
    return Seq2SeqNet(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_id=tokenizer.pad_id,
    )


def _eval_loss(
    model: Seq2SeqNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    pad_id: int,
) -> tuple[float, int]:
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            src_lens = batch["src_lens"].to(device)
            dec_in = batch["dec_in"].to(device)
            dec_tgt = batch["dec_tgt"].to(device)
            logits = model(src, src_lens, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
            n_tokens = int((dec_tgt != pad_id).sum().item())
            total_loss += float(loss.item())
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan"), 0
    return total_loss / total_tokens, total_tokens


def _save_checkpoint(
    path: Path,
    *,
    model: Seq2SeqNet,
    tokenizer: CharTokenizer,
    hp: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "seq2seq_char_gru",
        "state_dict": model.state_dict(),
        "id_to_char": tokenizer.id_to_char,
        "hyperparams": dict(hp),
    }
    torch.save(payload, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


class Seq2SeqLM(LanguageModel):
    def train(
        self,
        ctx: RunContext,
        paths: TrainPaths,
        hyperparams: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        hp = _merge_hp(self.spec, hyperparams)
        embed_dim = int(hp.get("embed_dim", 128))
        hidden_dim = int(hp.get("hidden_dim", 256))
        num_layers = int(hp.get("num_layers", 1))
        dropout = float(hp.get("dropout", 0.1))
        lr = float(hp.get("learning_rate", 1e-3))
        epochs = int(hp.get("epochs", 1))
        batch_size = int(hp.get("batch_size", 64))
        max_seq_len = int(hp.get("max_seq_len", 160))
        max_train_lines = hp.get("max_train_lines")
        max_val_lines = hp.get("max_val_lines")
        max_train_lines = int(max_train_lines) if max_train_lines is not None else None
        max_val_lines = int(max_val_lines) if max_val_lines is not None else 5000

        torch.manual_seed(ctx.seed)
        tokenizer = CharTokenizer.load(paths.vocab_json)

        train_lines = _read_non_empty_lines(paths.train_txt, max_train_lines)
        val_lines = _read_non_empty_lines(paths.val_txt, max_val_lines)
        train_ds = SeqDataset(train_lines, tokenizer, max_seq_len=max_seq_len)
        val_ds = SeqDataset(val_lines, tokenizer, max_seq_len=max_seq_len)
        if len(train_ds) == 0:
            raise ValueError("No training examples after preprocessing/filters.")

        collate_fn = lambda b: _collate_batch(b, tokenizer.pad_id)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = _build_model(
            tokenizer,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(ctx.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        t0 = time.perf_counter()
        last_train_loss = float("nan")
        for _ in range(max(1, epochs)):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in train_loader:
                src = batch["src"].to(ctx.device)
                src_lens = batch["src_lens"].to(ctx.device)
                dec_in = batch["dec_in"].to(ctx.device)
                dec_tgt = batch["dec_tgt"].to(ctx.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(src, src_lens, dec_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_steps += 1
            if epoch_steps > 0:
                last_train_loss = epoch_loss / epoch_steps

        val_loss, val_tokens = _eval_loss(
            model,
            val_loader,
            device=ctx.device,
            pad_id=tokenizer.pad_id,
        )
        val_ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("nan")

        ckpt = (
            ctx.outputs_dir
            / "checkpoints"
            / self.spec.checkpoint_subdir
            / "checkpoint.pt"
        )
        full_hp = {
            **dict(hp),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
        }
        _save_checkpoint(ckpt, model=model, tokenizer=tokenizer, hp=full_hp)

        return {
            "checkpoint": str(ckpt),
            "status": "ok",
            "train_loss": last_train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "val_tokens_scored": val_tokens,
            "vocab_size": tokenizer.vocab_size,
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "note": "Character-level GRU encoder-decoder with teacher forcing.",
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
        _ = language
        payload = _load_checkpoint(checkpoint)
        tokenizer = CharTokenizer(list(payload["id_to_char"]))
        hp = dict(payload.get("hyperparams") or {})
        embed_dim = int(hp.get("embed_dim", 128))
        hidden_dim = int(hp.get("hidden_dim", 256))
        num_layers = int(hp.get("num_layers", 1))
        dropout = float(hp.get("dropout", 0.1))
        max_seq_len = int(hp.get("max_seq_len", 160))

        model = _build_model(
            tokenizer,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(ctx.device)
        model.eval()

        seeds = list(prompts or [""])
        while len(seeds) < num_samples:
            seeds.append(seeds[len(seeds) % max(1, len(seeds))])
        seeds = seeds[:num_samples]

        out: list[str] = []
        for prompt in seeds:
            src_body = tokenizer.encode(prompt)[:max_seq_len]
            src_ids = [tokenizer.bos_id] + src_body + [tokenizer.eos_id]
            src = torch.tensor([src_ids], dtype=torch.long, device=ctx.device)
            src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=ctx.device)

            with torch.no_grad():
                hidden = model.encode(src, src_lens)
                prev = torch.tensor(
                    [[tokenizer.bos_id]], dtype=torch.long, device=ctx.device
                )
                decoded: list[int] = []
                for _ in range(max_new_tokens):
                    logits = model.decode(prev, hidden)
                    nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    if nxt == tokenizer.eos_id:
                        break
                    decoded.append(nxt)
                    prev = torch.tensor([[nxt]], dtype=torch.long, device=ctx.device)

            text = tokenizer.decode(decoded)
            out.append(text if text else prompt)

        return out

    def evaluate(
        self,
        ctx: RunContext,
        checkpoint: Path,
        paths: TrainPaths,
    ) -> dict[str, Any]:
        payload = _load_checkpoint(checkpoint)
        tokenizer = CharTokenizer(list(payload["id_to_char"]))
        hp = dict(payload.get("hyperparams") or {})
        embed_dim = int(hp.get("embed_dim", 128))
        hidden_dim = int(hp.get("hidden_dim", 256))
        num_layers = int(hp.get("num_layers", 1))
        dropout = float(hp.get("dropout", 0.1))
        batch_size = int(hp.get("batch_size", 64))
        max_seq_len = int(hp.get("max_seq_len", 160))
        max_val_lines = hp.get("max_val_lines")
        max_val_lines = int(max_val_lines) if max_val_lines is not None else None

        model = _build_model(
            tokenizer,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(ctx.device)

        val_lines = _read_non_empty_lines(paths.val_txt, max_val_lines)
        val_ds = SeqDataset(val_lines, tokenizer, max_seq_len=max_seq_len)
        if len(val_ds) == 0:
            return {
                "model_id": self.spec.model_id,
                "val_loss": float("nan"),
                "val_ppl": float("nan"),
                "note": "no validation examples",
            }

        collate_fn = lambda b: _collate_batch(b, tokenizer.pad_id)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        val_loss, total_tokens = _eval_loss(
            model,
            val_loader,
            device=ctx.device,
            pad_id=tokenizer.pad_id,
        )
        val_ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("nan")
        return {
            "model_id": self.spec.model_id,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "num_tokens_scored": total_tokens,
            "vocab_size": tokenizer.vocab_size,
            "note": "Character-level GRU encoder-decoder evaluation on val.txt",
        }
