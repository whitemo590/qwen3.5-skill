#!/usr/bin/env python3
"""Generate a Qwen3.5 FSDP2 YAML config from the local template."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal systems
    yaml = None


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def join_path_text(base: str, child: str) -> str:
    if base.endswith(("/", "\\")):
        return base + child
    sep = "/" if base.startswith("/") else os.sep
    return base + sep + child


def ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Source YAML template.")
    parser.add_argument("--model-path", required=True, help="HF model path used by tokenizer/model config.")
    parser.add_argument("--data-dir", required=True, help="Dataset root directory containing COCO2017/ and annotations.")
    parser.add_argument("--data-file", help="Dataset JSON path. Defaults to <data-dir>/annotations_slim.json.")
    parser.add_argument("--output-dir", required=True, help="Training output checkpoint directory.")
    parser.add_argument("--dcp-path", help="DCP checkpoint load path. Defaults to <model-path>_dcp.")
    parser.add_argument("--cache-dir", help="Preprocessing cache directory. Defaults to <output-dir>/cache.")
    parser.add_argument("--output-config", required=True, help="Generated YAML path.")
    parser.add_argument("--strict-paths", action="store_true", help="Fail if local data paths do not exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    if yaml is None:
        print("[FAIL] pyyaml is required: pip install pyyaml", file=sys.stderr)
        return 1

    args = parse_args(argv or sys.argv[1:])
    template_path = Path(args.template)
    if not template_path.is_file():
        print(f"[FAIL] template not found: {template_path}", file=sys.stderr)
        return 1

    data_file = args.data_file or join_path_text(args.data_dir, "annotations_slim.json")
    dcp_path = args.dcp_path or args.model_path.rstrip("/\\") + "_dcp"
    cache_dir = args.cache_dir or join_path_text(args.output_dir, "cache")

    if args.strict_paths:
        data_dir_path = Path(args.data_dir)
        data_file_path = Path(data_file)
        failures = []
        if not data_dir_path.is_dir():
            failures.append(f"data-dir not found: {data_dir_path}")
        if not data_file_path.is_file():
            failures.append(f"data-file not found: {data_file_path}")
        if failures:
            for failure in failures:
                print(f"[FAIL] {failure}", file=sys.stderr)
            return 1

    with template_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data = ensure_dict(config, "data")
    dataset_param = ensure_dict(data, "dataset_param")
    preprocess = ensure_dict(dataset_param, "preprocess_parameters")
    basic = ensure_dict(dataset_param, "basic_parameters")
    model = ensure_dict(config, "model")
    training = ensure_dict(config, "training")

    preprocess["model_name_or_path"] = args.model_path
    model["model_name_or_path"] = args.model_path
    basic["dataset_dir"] = args.data_dir
    basic["dataset"] = data_file
    basic["cache_dir"] = cache_dir
    training["load"] = dcp_path
    training["save"] = args.output_dir

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"[PASS] generated config: {output_path}")
    print(f"[INFO] model.model_name_or_path: {args.model_path}")
    print(f"[INFO] data.dataset_param.basic_parameters.dataset_dir: {args.data_dir}")
    print(f"[INFO] data.dataset_param.basic_parameters.dataset: {data_file}")
    print(f"[INFO] training.load: {dcp_path}")
    print(f"[INFO] training.save: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
