from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model.base import LanguageModel, RunContext, TrainPaths
from model.registry import ModelSpec

# 占位符定义
BOS = "\x00"
EOS = "\x01"

class CharDataset(Dataset):
    """将文本转换为 NPLM 训练所需的 (prefix_indices, target_index) 序列。"""
    def __init__(self, lines: list[str], n: int, vocab: list[str]):
        self.n = n
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.data = []
        
        for line in lines:
            text = line.strip()
            if not text: continue
            # 补齐 BOS 并添加 EOS
            tokens = [BOS] * (n - 1) + list(text) + [EOS]
            indices = [self.char_to_idx.get(t, 0) for t in tokens]
            
            for i in range(len(indices) - n + 1):
                prefix = indices[i : i + n - 1]
                target = indices[i + n - 1]
                self.data.append((torch.tensor(prefix), torch.tensor(target)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class NPLMNetwork(nn.Module):
    """基础 NPLM 网络结构：Embedding -> Flatten -> Linear -> Tanh -> Linear。"""
    def __init__(self, vocab_size: int, n: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        # 前缀长度为 n-1
        self.linear1 = nn.Linear((n - 1) * emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x shape: [batch, n-1]
        embeds = self.embeddings(x) # [batch, n-1, emb_dim]
        flattened = embeds.view(embeds.size(0), -1) # [batch, (n-1)*emb_dim]
        h = self.activation(self.linear1(flattened))
        logits = self.linear2(h)
        return logits

class NPLM(LanguageModel):
    """神经网络语言模型封装类。"""

    def _resolve_hyperparams(self, override: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """
        统一将配置映射到模型内部使用的键：
        - context_len -> n
        - embed_dim -> emb_dim
        - learning_rate -> lr
        同时保留原始键，保证兼容旧 checkpoint。
        """
        hp = dict(self.spec.hyperparams)
        if override:
            hp.update(dict(override))

        n = int(hp.get("context_len", hp.get("n", 32)))
        emb_dim = int(hp.get("embed_dim", hp.get("emb_dim", 128)))
        hidden_dim = int(hp.get("hidden_dim", 256))
        lr = float(hp.get("learning_rate", hp.get("lr", 1e-3)))
        epochs = int(hp.get("epochs", 1))
        batch_size = int(hp.get("batch_size", 64))

        if n < 2:
            raise ValueError("context_len (or n) must be >= 2.")

        return {
            **hp,
            "context_len": n,
            "n": n,
            "embed_dim": emb_dim,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    def train(
        self,
        ctx: RunContext,
        paths: TrainPaths,
        hyperparams: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        hp = self._resolve_hyperparams(hyperparams)
        n = int(hp["n"])
        emb_dim = int(hp["emb_dim"])
        hidden_dim = int(hp["hidden_dim"])
        epochs = int(hp["epochs"])
        batch_size = int(hp["batch_size"])
        lr = float(hp["lr"])

        # 1. 准备词汇表
        lines = paths.train_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        vocab_set = {BOS, EOS}
        for line in lines:
            vocab_set.update(list(line.strip()))
        vocab = sorted(list(vocab_set))
        vocab_size = len(vocab)
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}

        # 2. 准备数据加载器
        dataset = CharDataset(lines, n, vocab)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. 初始化模型（使用 run.py 解析后的 device）
        model = NPLMNetwork(vocab_size, n, emb_dim, hidden_dim).to(ctx.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 4. 训练循环
        model.train()
        t0 = time.perf_counter()
        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(ctx.device), y_batch.to(ctx.device)
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #print(total_loss)

        # 5. 保存结果
        ckpt_dir = ctx.outputs_dir / "checkpoints" / self.spec.checkpoint_subdir
        ckpt_path = ckpt_dir / "checkpoint.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocab,
            # 保存标准化后的 hp，确保 generate/evaluate 与 config.yaml 一致
            "hp": hp,
        }, ckpt_path)

        return {
            "checkpoint": str(ckpt_path),
            "vocab_size": vocab_size,
            "train_loss": total_loss / len(loader),
            "status": "ok",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
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
        payload = torch.load(checkpoint, map_location="cpu")
        vocab = payload["vocab"]
        hp = self._resolve_hyperparams(payload.get("hp"))
        n = int(hp["n"])
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}

        emb_dim = int(hp["emb_dim"])
        hidden_dim = int(hp["hidden_dim"])

        model = NPLMNetwork(len(vocab), n, emb_dim, hidden_dim).to(ctx.device)
        model.load_state_dict(payload["model_state"])
        model.eval()

        results = []
        seeds = prompts if prompts else [" "]
        
        with torch.no_grad():
            for _ in range(num_samples):
                prompt = seeds[_ % len(seeds)]
                current_tokens = [BOS] * (n - 1) + list(prompt)
                
                for _ in range(max_new_tokens):
                    # 取最后 n-1 个 token 作为输入
                    input_indices = [char_to_idx.get(t, 0) for t in current_tokens[-(n-1):]]
                    x = torch.tensor([input_indices], device=ctx.device)
                    
                    logits = model(x)
                    # 贪心选择
                    next_idx = torch.argmax(logits, dim=1).item()

                    # --- 新增逻辑 1: 惩罚 EOS (防止早退) ---
                    min_gen_length = 20  # 设置你期望的最小字数
                    current_gen_len = len(current_tokens) - (n - 1)
                    if current_gen_len < min_gen_length:
                        eos_idx = char_to_idx[EOS]
                        logits[0, eos_idx] = -1e10  # 还没写够，不许说再见

                    #空格惩罚
                    # 在 generate 函数的循环中，计算出 logits 后
                    space_idx = char_to_idx.get(" ", None)
                    if space_idx is not None:
                        # 强制将空格的概率降至最低，使其永远不会被抽中
                        logits[0, space_idx] = -1e10 

                    # --- 新增逻辑 2: 温度采样 (增加灵活性) ---
                    temperature = 0.8  # 0.7-1.0 之间，数值越高越随机，越低越死板
                    probs = torch.softmax(logits / temperature, dim=1)
                    next_idx = torch.multinomial(probs, num_samples=1).item()
                    next_char = vocab[next_idx]
                    
                    current_tokens.append(next_char)
                    if next_char == EOS:
                        break
                
                # 清洗输出
                out_chars = current_tokens[n-1:]
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
        payload = torch.load(checkpoint, map_location="cpu")
        vocab = payload["vocab"]
        hp = self._resolve_hyperparams(payload.get("hp"))

        n = int(hp["n"])
        emb_dim = int(hp["emb_dim"])
        hidden_dim = int(hp["hidden_dim"])
        batch_size = int(hp["batch_size"])

        if not paths.val_txt.exists():
            raise FileNotFoundError(paths.val_txt)

        lines = paths.val_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        dataset = CharDataset(lines, n, vocab)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        model = NPLMNetwork(len(vocab), n, emb_dim, hidden_dim).to(ctx.device)
        model.load_state_dict(payload["model_state"])
        model.eval()

        # sum reduction 便于计算平均 NLL -> PPL
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss_sum = 0.0
        total_count = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(ctx.device)
                y_batch = y_batch.to(ctx.device)
                logits = model(x_batch)
                loss_sum = criterion(logits, y_batch).item()
                total_loss_sum += loss_sum
                total_count += int(y_batch.numel())

        if total_count == 0:
            return {
                "model_id": self.spec.model_id,
                "val_loss": float("nan"),
                "val_ppl": float("nan"),
                "num_tokens_scored": 0,
                "note": "no n-gram positions in val.txt",
            }

        val_loss = total_loss_sum / total_count
        val_ppl = math.exp(val_loss)
        return {
            "model_id": self.spec.model_id,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "n": n,
            "num_tokens_scored": total_count,
        }