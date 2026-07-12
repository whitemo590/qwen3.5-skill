#!/usr/bin/env python3
"""Generate comparable baseline and optimized Qwen3.5 FSDP2 benchmark configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def suffix_path(value: Any, suffix: str) -> Any:
    if not isinstance(value, str) or not value:
        return value
    return value.rstrip("/\\") + suffix


def configure_variant(
    source: dict[str, Any], name: str, effective_batch: int, micro_batch: int,
    use_triton_gdn: bool, train_iters: int,
) -> dict[str, Any]:
    if effective_batch % micro_batch != 0:
        raise ValueError(f"effective batch {effective_batch} is not divisible by micro batch {micro_batch}")
    config = copy.deepcopy(source)
    training = ensure_dict(config, "training")
    model = ensure_dict(config, "model")
    data = ensure_dict(config, "data")
    dataset_param = ensure_dict(data, "dataset_param")
    basic = ensure_dict(dataset_param, "basic_parameters")
    dataloader = ensure_dict(data, "dataloader_param")
    preprocess = ensure_dict(dataset_param, "preprocess_parameters")
    tools = ensure_dict(config, "tools")

    training["micro_batch_size"] = micro_batch
    training["gradient_accumulation_steps"] = effective_batch // micro_batch
    training["train_iters"] = train_iters
    training["save_interval"] = 0
    training["save"] = ""
    model["use_triton_gdn"] = use_triton_gdn
    basic["cache_dir"] = suffix_path(basic.get("cache_dir", "./cache"), f"_{name}")
    dataloader["shuffle"] = False
    dataloader["num_workers"] = 4 if name == "baseline" else 8
    preprocess.setdefault("use_fast_tokenizer", True)
    ensure_dict(tools, "profile")["enable"] = False
    ensure_dict(tools, "memory_profile")["enable"] = False
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--effective-batch", type=int, default=4)
    parser.add_argument("--baseline-micro-batch", type=int, default=1)
    parser.add_argument("--optimized-micro-batch", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.config).resolve()
    source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = configure_variant(
        source, "baseline", args.effective_batch, args.baseline_micro_batch, False, args.train_iters
    )
    optimized = configure_variant(
        source, "optimized", args.effective_batch, args.optimized_micro_batch, True, args.train_iters
    )
    baseline_path = output_dir / "qwen3_5_0.8B_baseline.yaml"
    optimized_path = output_dir / "qwen3_5_0.8B_optimized.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline, allow_unicode=True, sort_keys=False), encoding="utf-8")
    optimized_path.write_text(yaml.safe_dump(optimized, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = {
        "source_config": str(source_path), "effective_batch_per_data_parallel_rank": args.effective_batch,
        "train_iters": args.train_iters,
        "baseline": {"config": str(baseline_path), "micro_batch": args.baseline_micro_batch,
                     "gradient_accumulation": args.effective_batch // args.baseline_micro_batch,
                     "use_triton_gdn": False},
        "optimized": {"config": str(optimized_path), "micro_batch": args.optimized_micro_batch,
                      "gradient_accumulation": args.effective_batch // args.optimized_micro_batch,
                      "use_triton_gdn": True},
        "controlled_variables": [
            "seed", "dataset", "cutoff_len", "learning_rate", "train_iters",
            "effective batch per data-parallel rank", "frozen modules", "precision",
        ],
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
