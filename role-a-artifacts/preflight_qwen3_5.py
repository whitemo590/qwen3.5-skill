#!/usr/bin/env python3
"""Offline preflight checks for the Qwen3.5 MindSpeed-MM migration assets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--mindspeed-root", default=str(root / "third_party/MindSpeed-MM"))
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project_root).resolve()
    mindspeed = Path(args.mindspeed_root).resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    plugin = mindspeed / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
    converter = mindspeed / "checkpoint/vlm_model/converters/qwen3_5.py"
    config = project / "role-a-artifacts/qwen3_5_0.8B_config.yaml"
    launch = project / "role-a-artifacts/finetune_qwen3_5_0.8B.sh"
    conversion = project / "role-a-artifacts/convert_qwen3_5_0.8B_weights.sh"
    inference = project / "role-a-artifacts/inference_qwen3_5.py"
    alignment = project / "role-a-artifacts/precision_align_qwen3_5.py"

    for name, path in [
        ("MindSpeed Qwen3.5 plugin", plugin), ("Qwen35Converter", converter),
        ("0.8B YAML", config), ("training launcher", launch),
        ("weight conversion launcher", conversion), ("inference script", inference),
        ("precision alignment script", alignment),
    ]:
        check(name, path.is_file(), str(path))

    if plugin.is_file():
        text = plugin.read_text(encoding="utf-8")
        check("model registration", '@model_register.register("qwen3_5")' in text, "qwen3_5")
        check("Triton GDN switch", "use_triton_gdn" in text, "use_triton_gdn present")
    if converter.is_file():
        text = converter.read_text(encoding="utf-8")
        check("converter class", "class Qwen35Converter" in text, "Qwen35Converter present")
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        check("YAML model id", bool(re.search(r"^\s*model_id:\s*qwen3_5\s*$", text, re.M)), "qwen3_5")
        check("YAML plugin path", "mindspeed_mm/fsdp/models/qwen3_5" in text, "qwen3_5 plugin")

    failed = sum(not bool(item["passed"]) for item in checks)
    summary = {"checks": checks, "passed": len(checks) - failed, "failed": failed}
    print(f"[SUMMARY] {summary['passed']}/{len(checks)} passed, {failed} failed")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
