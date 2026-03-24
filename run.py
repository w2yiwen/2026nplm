from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from model.base import RunContext, TrainPaths
from model.registry import get_model_spec, instantiate_model
from scripts.common import AppConfig, default_config_path, describe_device, ensure_dir, resolve_device
from scripts.pipeline import run_preprocess


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _config_path(args: argparse.Namespace) -> Optional[Path]:
    return Path(args.config) if getattr(args, "config", None) else None


def cmd_preprocess(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(_config_path(args))
    stats = run_preprocess(
        cfg,
        max_files_per_dataset=args.max_files_per_dataset,
        max_total_chars=args.max_total_chars,
        max_chars_per_dataset=args.max_chars_per_dataset,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(_config_path(args))
    model_id = args.model or cfg.default_model
    spec = get_model_spec(model_id, cfg.models)
    device = resolve_device(args.device or cfg.device)
    logging.getLogger(__name__).info("Device: %s", describe_device(device))

    processed = cfg.processed_dir
    paths = TrainPaths(
        processed_dir=processed,
        train_txt=processed / "train.txt",
        val_txt=processed / "val.txt",
        vocab_json=processed / "vocab.json",
    )
    for p in (paths.train_txt, paths.val_txt, paths.vocab_json):
        if not p.exists():
            logging.error("Missing %s — run `python run.py preprocess` first.", p)
            return 2

    out = ensure_dir(cfg.outputs_dir)
    ctx = RunContext(device=device, seed=cfg.seed, outputs_dir=out)

    model = instantiate_model(spec)
    hp_override: Optional[dict[str, Any]] = None
    if args.hyperparams_json:
        with Path(args.hyperparams_json).open("r", encoding="utf-8") as f:
            hp_override = json.load(f)

    metrics = model.train(ctx, paths, hyperparams=hp_override)
    metrics_path = out / "metrics" / f"{model_id}_train.json"
    payload = {"model_id": model_id, **metrics}
    _write_json(metrics_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(_config_path(args))
    model_id = args.model or cfg.default_model
    spec = get_model_spec(model_id, cfg.models)
    device = resolve_device(args.device or cfg.device)

    out = ensure_dir(cfg.outputs_dir)
    ctx = RunContext(device=device, seed=cfg.seed, outputs_dir=out)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is None:
        ckpt = ctx.outputs_dir / "checkpoints" / spec.checkpoint_subdir / "checkpoint.pt"
    if not ckpt.exists():
        logging.error("Checkpoint not found: %s — train first.", ckpt)
        return 2

    lang = args.lang.lower()
    prompts: Optional[list[str]] = None
    if args.prompt:
        prompts = [args.prompt]
    else:
        block = cfg.generation.get(lang) or cfg.generation.get("en")
        prompts = list(block.get("prefixes", [])) if isinstance(block, dict) else None

    model = instantiate_model(spec)
    lines = model.generate(
        ctx,
        ckpt,
        language=lang,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        prompts=prompts,
    )

    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{model_id}_{lang}.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
    print(str(out_path))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(_config_path(args))
    model_id = args.model or cfg.default_model
    spec = get_model_spec(model_id, cfg.models)
    device = resolve_device(args.device or cfg.device)

    processed = cfg.processed_dir
    paths = TrainPaths(
        processed_dir=processed,
        train_txt=processed / "train.txt",
        val_txt=processed / "val.txt",
        vocab_json=processed / "vocab.json",
    )
    out = ensure_dir(cfg.outputs_dir)
    ctx = RunContext(device=device, seed=cfg.seed, outputs_dir=out)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is None:
        ckpt = ctx.outputs_dir / "checkpoints" / spec.checkpoint_subdir / "checkpoint.pt"
    if not ckpt.exists():
        logging.error("Checkpoint not found: %s", ckpt)
        return 2

    model = instantiate_model(spec)
    metrics = model.evaluate(ctx, ckpt, paths)
    metrics_path = out / "metrics" / f"{model_id}_eval.json"
    payload = {"model_id": model_id, **metrics}
    _write_json(metrics_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = AppConfig.load(_config_path(args))
    metrics_dir = cfg.outputs_dir / "metrics"
    if not metrics_dir.exists():
        logging.error("No metrics directory at %s", metrics_dir)
        return 2

    files = sorted(metrics_dir.glob("*_eval.json"))
    if args.include_train:
        files = sorted(set(files) | set(metrics_dir.glob("*_train.json")))
    if not files:
        logging.error("No metrics JSON files in %s", metrics_dir)
        return 2

    rows: list[tuple[str, float]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mid = str(data.get("model_id", fp.stem))
        ppl = data.get("val_ppl")
        if ppl is None:
            continue
        rows.append((mid, float(ppl)))

    if not rows:
        logging.error("No val_ppl fields found in metrics files.")
        return 2

    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#3B82F6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("val_ppl (lower is better)")
    ax.set_title("Model comparison (stub metrics until real training)")
    fig.tight_layout()

    fig_dir = ensure_dir(cfg.outputs_dir / "figures")
    out_path = fig_dir / (args.output_name or "comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(str(out_path))
    return 0


def _add_verbose(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nplm",
        description="LM pipeline: preprocess → train → generate → evaluate → plot (NPLM + baselines under model/).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Merge datasets and build vocab")
    _add_verbose(pp)
    pp.add_argument("--config", type=str, default=None, help=f"YAML (default: {default_config_path()})")
    pp.add_argument("--max-files-per-dataset", type=int, default=None)
    pp.add_argument("--max-total-chars", type=int, default=None)
    pp.add_argument(
        "--max-chars-per-dataset",
        type=int,
        default=None,
        help="Cap characters per dataset (recommended for zh+en mix).",
    )
    pp.add_argument("--seed", type=int, default=None)
    pp.set_defaults(func=cmd_preprocess)

    tr = sub.add_parser("train", help="Train a model registered in config.yaml")
    _add_verbose(tr)
    tr.add_argument("--config", type=str, default=None)
    tr.add_argument("--model", type=str, default=None)
    tr.add_argument("--device", type=str, default=None)
    tr.add_argument("--hyperparams-json", type=str, default=None)
    tr.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate", help="Generate text (zh / en)")
    _add_verbose(gen)
    gen.add_argument("--config", type=str, default=None)
    gen.add_argument("--model", type=str, default=None)
    gen.add_argument("--device", type=str, default=None)
    gen.add_argument("--checkpoint", type=str, default=None)
    gen.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    gen.add_argument("--num-samples", type=int, default=8)
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--prompt", type=str, default=None)
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("evaluate", help="Evaluate checkpoint")
    _add_verbose(ev)
    ev.add_argument("--config", type=str, default=None)
    ev.add_argument("--model", type=str, default=None)
    ev.add_argument("--device", type=str, default=None)
    ev.add_argument("--checkpoint", type=str, default=None)
    ev.set_defaults(func=cmd_evaluate)

    pl = sub.add_parser("plot", help="Bar chart of val_ppl from outputs/metrics/*.json")
    _add_verbose(pl)
    pl.add_argument("--config", type=str, default=None)
    pl.add_argument("--output-name", type=str, default="comparison.png")
    pl.add_argument("--include-train", action="store_true", help="Also read *_train.json")
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(bool(getattr(args, "verbose", False)))
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
