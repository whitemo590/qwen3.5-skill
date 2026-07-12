#!/usr/bin/env python3
"""Statically validate Qwen3.5 YAML, model contract, and FSDP2 module plans."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml


FORWARD_FIELDS = {
    "input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds",
    "labels", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
    "cache_position", "logits_to_keep",
}
SYNTHETIC_MODULES = {
    "model", "model.visual", "model.visual.blocks", "model.visual.blocks.0",
    "model.visual.blocks.1", "model.language_model", "model.language_model.embed_tokens",
    "model.language_model.layers", "model.language_model.layers.0",
    "model.language_model.layers.1", "model.language_model.norm", "lm_head",
}
LOSS_TYPES = {"raw", "default", "per_sample_loss", "per_token_loss", "token_loss"}


@dataclass
class Check:
    name: str
    level: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.level == "FAIL"


def module_name_match(pattern: str, name: str) -> bool:
    expression = re.escape(pattern).replace(r"\{\*\}", r"[^.]+")
    return re.fullmatch(expression, name) is not None


def nested_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    return next((node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name), None)


def find_method(class_node: ast.ClassDef | None, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if class_node is None:
        return None
    return next(
        (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name),
        None,
    )


def looks_windows_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or "\\" in value or PureWindowsPath(value).drive != ""


def validate(args: argparse.Namespace) -> tuple[list[Check], dict[str, Any]]:
    config_path = Path(args.config).resolve()
    mindspeed_root = Path(args.mindspeed_root).resolve()
    model_source = mindspeed_root / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
    modelhub_source = mindspeed_root / "mindspeed_mm/fsdp/models/modelhub.py"
    train_engine_source = mindspeed_root / "mindspeed_mm/fsdp/train/train_engine.py"
    checks: list[Check] = []

    def record(name: str, passed: bool, detail: str, warning: bool = False) -> None:
        level = "PASS" if passed else ("WARN" if warning else "FAIL")
        checks.append(Check(name, level, detail))

    record("config exists", config_path.is_file(), str(config_path))
    record("MindSpeed model source exists", model_source.is_file(), str(model_source))
    record("ModelHub source exists", modelhub_source.is_file(), str(modelhub_source))
    record("TrainEngine source exists", train_engine_source.is_file(), str(train_engine_source))
    if any(check.failed for check in checks):
        return checks, {}

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    record("top-level YAML is mapping", isinstance(config, dict), type(config).__name__)
    for section in ("parallel", "data", "model", "training"):
        record(f"section {section}", isinstance(config.get(section), dict), section)

    model_id = nested_get(config, "model.model_id")
    plugins = nested_get(config, "training.plugin", [])
    loss_type = nested_get(config, "model.loss_cfg.loss_type")
    record("model id", model_id == "qwen3_5", str(model_id))
    record(
        "model plugin configured",
        isinstance(plugins, list) and "mindspeed_mm/fsdp/models/qwen3_5" in plugins,
        str(plugins),
    )
    record(
        "Hugging Face data plugin configured",
        isinstance(plugins, list) and "mindspeed_mm/fsdp/data/datasets/huggingface" in plugins,
        str(plugins),
    )
    record("supported loss type", loss_type in LOSS_TYPES, str(loss_type))
    record(
        "meta init has checkpoint",
        not nested_get(config, "training.init_model_with_meta_device", False)
        or bool(nested_get(config, "training.load")),
        f"meta={nested_get(config, 'training.init_model_with_meta_device')}, load={nested_get(config, 'training.load')}",
    )

    model_text = model_source.read_text(encoding="utf-8")
    tree = ast.parse(model_text)
    model_class = find_class(tree, "Qwen3_5ForConditionalGeneration")
    forward = find_method(model_class, "forward")
    forward_args = set()
    has_kwargs = False
    if forward is not None:
        forward_args = {arg.arg for arg in [*forward.args.args, *forward.args.kwonlyargs]}
        has_kwargs = forward.args.kwarg is not None
    record("registered model class", '@model_register.register("qwen3_5")' in model_text, "qwen3_5")
    record("conditional generation class", model_class is not None, "Qwen3_5ForConditionalGeneration")
    record("forward method", forward is not None, "forward")
    missing_forward = sorted(FORWARD_FIELDS - forward_args)
    record("training batch fields accepted", not missing_forward, f"missing={missing_forward}")
    record("forward accepts extra kwargs", has_kwargs, "**kwargs")
    record(
        "forward returns loss output",
        "return Qwen3_5CausalLMOutputWithPast(" in model_text and "loss=loss" in model_text,
        "Qwen3_5CausalLMOutputWithPast(loss=loss)",
    )
    record("Triton GDN override", "use_triton_gdn" in model_text, "use_triton_gdn")

    modelhub_text = modelhub_source.read_text(encoding="utf-8")
    train_engine_text = train_engine_source.read_text(encoding="utf-8")
    record("ModelHub registry lookup", "model_register.get(model_id)" in modelhub_text, "model_register.get")
    record("Trainer consumes output.loss", "loss = output.loss" in train_engine_text, "output.loss")

    plans = {
        "FSDP": nested_get(config, "parallel.fsdp_plan.apply_modules", []),
        "recompute": nested_get(config, "parallel.recompute_plan.apply_modules", []),
        "freeze": nested_get(config, "model.freeze", []),
    }
    plan_matches: dict[str, dict[str, list[str]]] = {}
    for plan_name, patterns in plans.items():
        valid_list = isinstance(patterns, list) and all(isinstance(pattern, str) for pattern in patterns)
        record(f"{plan_name} plan format", valid_list, str(patterns))
        plan_matches[plan_name] = {}
        if not valid_list:
            continue
        for pattern in patterns:
            matches = sorted(name for name in SYNTHETIC_MODULES if module_name_match(pattern, name))
            plan_matches[plan_name][pattern] = matches
            record(f"{plan_name} pattern {pattern}", bool(matches), ", ".join(matches) or "no match")

    path_fields = [
        "data.dataset_param.preprocess_parameters.model_name_or_path",
        "data.dataset_param.basic_parameters.dataset_dir",
        "data.dataset_param.basic_parameters.dataset",
        "data.dataset_param.basic_parameters.cache_dir",
        "model.model_name_or_path", "training.load", "training.save",
    ]
    windows_paths = [path for path in path_fields if looks_windows_path(nested_get(config, path))]
    record(
        "Linux-portable configured paths",
        not windows_paths,
        "windows-style fields=" + str(windows_paths),
        warning=not args.strict_linux_paths,
    )

    attr = nested_get(config, "data.dataset_param.attr", {})
    expected_attr = {
        "images": "images", "messages": "messages", "role_tag": "role", "content_tag": "content",
        "user_tag": "user", "assistant_tag": "assistant",
    }
    record("dataset field mapping", attr == expected_attr, str(attr))
    stats = {
        "config": str(config_path), "mindspeed_root": str(mindspeed_root),
        "forward_args": sorted(forward_args), "plan_matches": plan_matches,
        "windows_path_fields": windows_paths,
    }
    return checks, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mindspeed-root", required=True)
    parser.add_argument("--strict-linux-paths", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, stats = validate(args)
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    failed = sum(check.failed for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    print(f"[SUMMARY] {len(checks) - failed}/{len(checks)} non-failing, {failed} failed, {warnings} warnings")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"checks": [asdict(check) for check in checks], "stats": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
