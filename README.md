# Neural / N-gram Language Modeling Pipeline

面向课程与实验的**语言建模流水线**：从原始语料预处理，到多类模型训练、文本生成、验证集评估与指标对比图。默认主线为 **NPLM（神经概率语言模型）**；**N-gram** 已实现完整基线；**Seq2Seq / GPT 式** 结构预留，便于扩展对比实验。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 预处理 | 合并配置中的数据集，划分 `train` / `val`，构建字符级词表 |
| 训练 | 通过 `config.yaml` 注册模型，统一入口训练并保存 checkpoint |
| 生成 | 支持中文 / 英文提示（由 `generation` 配置） |
| 评估 | 输出验证集指标（如困惑度）至 `outputs/metrics/` |
| 对比图 | 汇总各模型 `*_eval.json` 中的指标并绘图 |

---

## 环境要求

- **Python** 3.10+
- **PyTorch** 2.x（有 NVIDIA GPU 时会自动优先使用 CUDA；N-gram 主要为 CPU 统计，亦可在 CPU 上运行）
- 依赖见 `requirements.txt` / `pyproject.toml`

---

## 数据准备

1. 将语料置于 **`data/`** 下，路径与格式需与 **`config.yaml`** 中 `datasets` 一致。
2. 默认配置仅使用 **`data/wiki_zh_2019/wiki_zh`**（JSONL 百科文本，体量约 **1.23GB** 量级，视你下载版本而定）。
3. **本仓库不包含大型语料文件**；请自行获取并放入上述目录，或修改 `datasets` 指向你的数据。

全量预处理前，请确保：

- `config.yaml` 中 `preprocess` 下各项为 `null`（不截断）；
- 命令行 **不要** 使用 `--max-total-chars`、`--max-files-per-dataset` 等限制（除非做冒烟测试）。

---

## 安装

**Linux / macOS**

```bash
git clone <你的仓库地址>.git
cd 2026nplm
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell)**

```powershell
git clone <你的仓库地址>.git
cd 2026nplm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

可选：指定单块 GPU

```bash
export CUDA_VISIBLE_DEVICES=0   # Linux / macOS
```

```powershell
$env:CUDA_VISIBLE_DEVICES="0"  # Windows
```

---

## 快速开始

在项目根目录执行（**按顺序**）：

```bash
# 1. 预处理：生成 processed/train.txt, val.txt, vocab.json 等
python run.py preprocess

# 2. 训练（示例：N-gram）
python run.py train --model ngram

# 3. 生成样例
python run.py generate --model ngram --lang zh --num-samples 5 --max-new-tokens 40

# 4. 验证集评估
python run.py evaluate --model ngram

# 5. 多模型指标柱状图（需先有 outputs/metrics/*_eval.json）
python run.py plot
```

更换模型时，将 `--model ngram` 改为 `nplm` / `seq2seq` / `gpt`（需在对应目录实现或保留桩）。

---

## 命令行说明

入口脚本为 **`run.py`**，子命令如下：

| 子命令 | 作用 |
|--------|------|
| `preprocess` | 读取 `config.yaml` 中的 `datasets`，写出 `processed/` |
| `train` | 训练 `--model` 指定的已注册模型 |
| `generate` | 加载 checkpoint 生成文本；`--lang zh` / `en` |
| `evaluate` | 在验证集上计算指标 |
| `plot` | 读取 `outputs/metrics/*_eval.json` 绘制对比图 |

常用参数：

- `--config <路径>`：指定配置文件（默认项目根目录 `config.yaml`）。
- 各子命令支持 `-v` / `--verbose` 输出更详细日志。

冒烟测试（小样本，仅用于调试流程）示例：

```bash
python run.py preprocess --max-total-chars 100000 --max-files-per-dataset 20
```

---

## 配置说明（`config.yaml`）

- **`seed` / `device`**：随机种子与设备（`auto` 时优先 CUDA）。
- **`data_root` / `processed_dir` / `outputs_dir`**：数据与输出根目录（相对项目根）。
- **`default_model`**：未指定 `--model` 时的默认模型名。
- **`train_split`**：预处理时训练集比例。
- **`max_vocab_size`**：字符词表上限（预处理阶段）。
- **`datasets`**：数据集列表（`id`、`root`、`format`、`glob_pattern`、`text_key` 等）。
- **`models`**：模型注册表（`module`、`class`、`hyperparams`、`checkpoint_subdir`）。
- **`generation`**：按语言配置生成用前缀词列表。

复制 `config.yaml` 为自定义文件后，使用 `python run.py <子命令> --config my.yaml` 即可。

---

## 目录结构

```
├── config.yaml          # 全局配置
├── run.py               # CLI 入口
├── requirements.txt
├── pyproject.toml
├── data/                # 原始语料（不随仓库提交大文件）
├── processed/           # 预处理输出（建议 .gitignore）
├── outputs/             # checkpoint、metrics、samples、figures（建议 .gitignore）
├── scripts/             # 配置加载、数据读取、预处理流水线
└── model/
    ├── base.py          # LanguageModel 抽象接口
    ├── registry.py      # 模型注册与实例化
    ├── ngram/           # 字符级 N-gram（已实现）
    ├── nplm/            # NPLM（需自行实现网络与训练）
    ├── seq2seq/         # 预留
    └── gpt/             # 预留
```

---

## 模型与实现状态

| 目录 | 说明 |
|------|------|
| `model/ngram` | **已实现**：频次统计、Laplace 平滑、贪心生成、验证困惑度；超参见 `models.ngram.hyperparams`（如 `n`、`smoothing`）。 |
| `model/nplm` | 桩实现，可替换为真实 NPLM 训练逻辑。 |
| `model/seq2seq`、`model/gpt` | 桩实现，便于接口对齐与后续对比。 |

---

## 输出说明

| 路径 | 内容 |
|------|------|
| `processed/train.txt`、`val.txt` | 按行存储的文本 |
| `processed/vocab.json` | 字符词表 |
| `outputs/checkpoints/<模型>/checkpoint.pt` | 模型检查点 |
| `outputs/metrics/` | 训练/评估 JSON |
| `outputs/samples/` | 生成样例文本 |
| `outputs/figures/` | 对比图（如 `comparison.png`） |

---

## 常见问题

- **预处理很慢或内存不足**：全量百科体积大，可适当使用 `--max-total-chars` 做子集实验，或增加机器内存；长期可考虑流式预处理（需改代码）。
- **N-gram 生成重复**：贪心解码易陷入高频局部循环，可改为采样解码或提高 `n`（如 trigram），参见课题文档与 `model/ngram` 实现。
- **Windows 路径**：使用正斜杠或原始字符串均可；`--config` 建议使用绝对路径或相对项目根的路径。

---

## 许可证

若用于课程或开源发布，请在本仓库根目录添加 `LICENSE` 并在此处替换为实际许可证名称（如 MIT、Apache-2.0）。

---

## 引用与致谢

若本仓库参考了公开数据集或第三方代码，请在此列出数据来源与引用方式；课程作业可注明学校与课程名称。
