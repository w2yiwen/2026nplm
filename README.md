# NPLM 课题流水线（简洁版）

主线是 **`model/nplm`**，其它模型在 **`model/ngram`**、**`model/seq2seq`**、**`model/gpt`** 做对比。流程脚本在 **`scripts/`**，**一份 `config.yaml`** 管理路径、数据集、模型注册与生成提示。

## 环境

```bash
cd f:\2026nplm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

可选指定 GPU：`$env:CUDA_VISIBLE_DEVICES="0"`

## 命令（入口：`run.py`）

**语料**：`config.yaml` 里当前**只包含** **`data/wiki_zh_2019`**（中文，约 1.23GB）。全量预处理（勿加 `--max-total-chars` 等截断参数）：

```bash
python run.py preprocess
```

冒烟测试（小样本）：

```bash
python run.py preprocess --max-total-chars 100000 --max-files-per-dataset 20
```

训练 / 生成 / 评估 / 画图（示例：`ngram`，其它模型改 `--model` 即可）：

```bash
python run.py train --model ngram
python run.py generate --model ngram --lang zh
python run.py evaluate --model ngram
python run.py plot
```

默认配置文件为项目根目录 **`config.yaml`**，可用 `--config path\to\custom.yaml` 覆盖。

## 目录

| 路径 | 说明 |
|------|------|
| `config.yaml` | 随机种子、设备、路径、预处理、`datasets`、`models`、`generation` |
| `scripts/` | `common`（配置/设备）、`io_readers`、`tokenizer`、`pipeline`（预处理） |
| `model/` | `base.py`、`registry.py`、各子目录模型实现 |
| `processed/` | `train.txt`、`val.txt`、`vocab.json`、`corpus.jsonl` |
| `outputs/` | `checkpoints/`、`samples/`、`metrics/`、`figures/` |

- **`model/ngram`**：已实现**字符级 N-gram**（频次统计 + Laplace 平滑 + 贪心生成 + 验证困惑度），超参 `models.ngram.hyperparams` 中 `n`（如 `2` 为 bigram）、`smoothing`。
- **`model/nplm` / `seq2seq` / `gpt`**：仍为桩实现，可在此目录内补全网络与训练。
