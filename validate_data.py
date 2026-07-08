#!/usr/bin/env python3
"""Validate Qwen3.5 MLLM JSON data against the FSDP2 YAML mapping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal systems
    yaml = None


DEFAULT_ATTR = {
    "images": "images",
    "messages": "messages",
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    level: str = "PASS"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def _resolve_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _load_config(config_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if config_path is None:
        return {}, DEFAULT_ATTR.copy()
    if yaml is None:
        raise RuntimeError("pyyaml is required when --config is used")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    dataset_param = config.get("data", {}).get("dataset_param", {})
    attr = DEFAULT_ATTR.copy()
    attr.update(dataset_param.get("attr") or {})
    return config, attr


def _record(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail, level="PASS" if passed else "FAIL"))


def _format_examples(examples: list[str], max_errors: int) -> str:
    if not examples:
        return ""
    shown = examples[:max_errors]
    suffix = "" if len(examples) <= max_errors else f" ... (+{len(examples) - max_errors} more)"
    return "; ".join(shown) + suffix


def _message_rounds(messages: Any) -> int:
    return len(messages) // 2 if isinstance(messages, list) else 0


def validate(args: argparse.Namespace) -> tuple[list[Check], dict[str, Any]]:
    config_path = _resolve_path(args.config) if args.config else None
    config, attr = _load_config(config_path)
    config_dir = config_path.parent if config_path else None

    basic = config.get("data", {}).get("dataset_param", {}).get("basic_parameters", {})
    yaml_data = _resolve_path(basic.get("dataset"), config_dir)
    yaml_data_dir = _resolve_path(basic.get("dataset_dir"), config_dir)

    data_path = _resolve_path(args.data, Path.cwd()) if args.data else yaml_data
    data_dir = _resolve_path(args.data_dir, Path.cwd()) if args.data_dir else yaml_data_dir
    if data_dir is None and data_path is not None:
        data_dir = data_path.parent

    checks: list[Check] = []
    stats: dict[str, Any] = {
        "config": str(config_path) if config_path else None,
        "data": str(data_path) if data_path else None,
        "data_dir": str(data_dir) if data_dir else None,
        "cutoff_len": basic.get("cutoff_len"),
        "attr": attr,
    }

    _record(checks, "dataset path configured", data_path is not None, str(data_path) if data_path else "missing")
    _record(checks, "dataset_dir configured", data_dir is not None, str(data_dir) if data_dir else "missing")
    if data_path is None or data_dir is None:
        return checks, stats

    _record(checks, "dataset JSON exists", data_path.is_file(), str(data_path))
    _record(checks, "dataset_dir exists", data_dir.is_dir(), str(data_dir))
    if not data_path.is_file():
        return checks, stats

    try:
        with data_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)
        json_ok = True
    except Exception as exc:
        samples = None
        json_ok = False
        _record(checks, "JSON parses OK", False, str(exc))
    if not json_ok:
        return checks, stats

    is_list = isinstance(samples, list)
    _record(checks, "top-level JSON is list", is_list, type(samples).__name__)
    if not is_list:
        return checks, stats

    sample_count = len(samples)
    stats["sample_count"] = sample_count
    _record(
        checks,
        "sample count",
        sample_count == args.expected_samples,
        f"{sample_count} samples, expected {args.expected_samples}",
    )

    images_key = attr["images"]
    messages_key = attr["messages"]
    role_key = attr["role_tag"]
    content_key = attr["content_tag"]
    user_value = attr["user_tag"]
    assistant_value = attr["assistant_tag"]

    missing_images: list[str] = []
    missing_messages: list[str] = []
    bad_image_count: list[str] = []
    bad_messages_shape: list[str] = []
    missing_message_fields: list[str] = []
    bad_roles: list[str] = []
    bad_alternation: list[str] = []
    empty_content: list[str] = []
    missing_first_image_token: list[str] = []
    bad_image_token_count: list[str] = []
    missing_image_files: list[str] = []
    unreadable_image_files: list[str] = []
    bad_image_ext: list[str] = []
    round_counter: Counter[int] = Counter()
    message_counts: list[int] = []
    total_image_refs = 0

    for idx, sample in enumerate(samples):
        sample_label = f"sample[{idx}]"
        if not isinstance(sample, dict):
            missing_images.append(f"{sample_label}: not object")
            missing_messages.append(f"{sample_label}: not object")
            continue

        images = sample.get(images_key)
        messages = sample.get(messages_key)

        if images_key not in sample:
            missing_images.append(sample_label)
        if messages_key not in sample:
            missing_messages.append(sample_label)

        if not isinstance(images, list):
            bad_image_count.append(f"{sample_label}: images is {type(images).__name__}")
            images = []
        elif len(images) != args.expected_images_per_sample:
            bad_image_count.append(f"{sample_label}: {len(images)} images")

        if not isinstance(messages, list) or not messages or len(messages) % 2 != 0:
            length = len(messages) if isinstance(messages, list) else type(messages).__name__
            bad_messages_shape.append(f"{sample_label}: messages={length}")
            messages = messages if isinstance(messages, list) else []

        total_image_refs += len(images)
        message_counts.append(len(messages))
        round_counter[_message_rounds(messages)] += 1

        expected_role = user_value
        total_image_tokens = 0
        first_user_content = None
        for msg_idx, message in enumerate(messages):
            message_label = f"{sample_label}.messages[{msg_idx}]"
            if not isinstance(message, dict):
                missing_message_fields.append(f"{message_label}: not object")
                continue

            role = message.get(role_key)
            content = message.get(content_key)
            if role_key not in message or content_key not in message:
                missing_message_fields.append(message_label)
                continue
            if role not in (user_value, assistant_value):
                bad_roles.append(f"{message_label}: role={role!r}")
            if role != expected_role:
                bad_alternation.append(f"{message_label}: role={role!r}, expected={expected_role!r}")
            expected_role = assistant_value if expected_role == user_value else user_value

            if not isinstance(content, str) or not content.strip():
                empty_content.append(message_label)
            if isinstance(content, str):
                total_image_tokens += content.count("<image>")
                if first_user_content is None and role == user_value:
                    first_user_content = content

        if first_user_content is None or "<image>" not in first_user_content:
            missing_first_image_token.append(sample_label)
        if total_image_tokens != len(images):
            bad_image_token_count.append(f"{sample_label}: tokens={total_image_tokens}, images={len(images)}")

        for image in images:
            image_label = f"{sample_label}: {image!r}"
            if not isinstance(image, str):
                missing_image_files.append(f"{image_label} is not string")
                continue
            if Path(image).suffix.lower() != ".jpg":
                bad_image_ext.append(image_label)
            image_path = Path(image)
            final_path = image_path if image_path.is_absolute() else data_dir / image_path
            if not final_path.is_file():
                missing_image_files.append(str(final_path))
            elif not os.access(final_path, os.R_OK):
                unreadable_image_files.append(str(final_path))

    _record(checks, f'field "{images_key}" present', not missing_images, _format_examples(missing_images, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, f'field "{messages_key}" present', not missing_messages, _format_examples(missing_messages, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, "single-image samples", not bad_image_count, _format_examples(bad_image_count, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, "messages non-empty and even", not bad_messages_shape, _format_examples(bad_messages_shape, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, f'messages have "{role_key}" and "{content_key}"', not missing_message_fields, _format_examples(missing_message_fields, args.max_errors) or "all messages")
    _record(checks, "roles are valid", not bad_roles, _format_examples(bad_roles, args.max_errors) or f"{user_value}/{assistant_value}")
    _record(checks, "role alternation correct", not bad_alternation, _format_examples(bad_alternation, args.max_errors) or f"{sample_count}/{sample_count} start with {user_value}")
    _record(checks, "message content non-empty", not empty_content, _format_examples(empty_content, args.max_errors) or "all messages")
    _record(checks, "first user message contains <image>", not missing_first_image_token, _format_examples(missing_first_image_token, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, "<image> token count matches images", not bad_image_token_count, _format_examples(bad_image_token_count, args.max_errors) or f"{sample_count}/{sample_count}")
    _record(checks, "image extension is .jpg", not bad_image_ext, _format_examples(bad_image_ext, args.max_errors) or f"{total_image_refs}/{total_image_refs}")
    _record(checks, "image files exist", not missing_image_files, _format_examples(missing_image_files, args.max_errors) or f"{total_image_refs}/{total_image_refs}")
    _record(checks, "image files readable", not unreadable_image_files, _format_examples(unreadable_image_files, args.max_errors) or f"{total_image_refs}/{total_image_refs}")

    total_messages = sum(message_counts)
    avg_messages = total_messages / sample_count if sample_count else 0.0
    avg_rounds = avg_messages / 2.0
    min_messages = min(message_counts) if message_counts else 0
    max_messages = max(message_counts) if message_counts else 0
    stats.update(
        {
            "total_messages": total_messages,
            "total_image_refs": total_image_refs,
            "round_distribution": dict(sorted(round_counter.items())),
            "min_messages": min_messages,
            "max_messages": max_messages,
            "avg_messages": avg_messages,
            "avg_rounds": avg_rounds,
        }
    )
    checks.append(
        Check(
            name="conversation round stats",
            passed=True,
            detail=f"min={min_messages} ({min_messages // 2} rounds), max={max_messages} ({max_messages // 2} rounds), avg={avg_messages:.2f} ({avg_rounds:.2f} rounds)",
            level="INFO",
        )
    )
    if basic.get("cutoff_len") is not None:
        checks.append(Check(name="cutoff_len", passed=True, detail=str(basic.get("cutoff_len")), level="INFO"))

    return checks, stats


def write_report(path: Path, checks: list[Check], stats: dict[str, Any]) -> None:
    payload = {
        "checks": [check.__dict__ for check in checks],
        "stats": stats,
        "summary": {
            "passed": sum(1 for check in checks if check.level != "INFO" and check.passed),
            "failed": sum(1 for check in checks if check.level != "INFO" and not check.passed),
            "info": sum(1 for check in checks if check.level == "INFO"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="FSDP2 YAML config path. Used for data attr mapping and default paths.")
    parser.add_argument("--data", help="Override dataset JSON path.")
    parser.add_argument("--data-dir", help="Override dataset root directory used to resolve image paths.")
    parser.add_argument("--expected-samples", type=int, default=2000)
    parser.add_argument("--expected-images-per-sample", type=int, default=1)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--report-json", help="Optional machine-readable JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv or sys.argv[1:])
    checks, stats = validate(args)

    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")

    pass_count = sum(1 for check in checks if check.level != "INFO" and check.passed)
    fail_count = sum(1 for check in checks if check.level != "INFO" and not check.passed)
    total_count = pass_count + fail_count
    print(f"[SUMMARY] {pass_count}/{total_count} checks passed, {fail_count} failed")

    if args.report_json:
        write_report(Path(args.report_json), checks, stats)
        print(f"[INFO] JSON report written: {args.report_json}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
