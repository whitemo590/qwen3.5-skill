#!/usr/bin/env python3
"""Run the no-NPU mock flow for the Qwen3.5 FSDP2 migration skill."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def find_project_root(start: Path) -> Path:
    for current in [start, *start.parents]:
        if (current / "dataset" / "dataset" / "annotations_slim.json").is_file():
            return current
    return start


def run_command(command: list[str]) -> int:
    print("[CMD] " + " ".join(str(part) for part in command))
    completed = subprocess.run(command, text=True)
    return completed.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    default_template = project_root / "role-a-artifacts" / "qwen3_5_0.8B_config.yaml"
    default_data_dir = project_root / "dataset" / "dataset"
    default_data_file = default_data_dir / "annotations_slim.json"
    default_generated = script_dir / "generated"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(default_template))
    parser.add_argument("--model-path", default="/home/data/qwen3_5_0.8B")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--data-file", default=str(default_data_file))
    parser.add_argument("--output-dir", default=str(default_generated / "qwen3_5_0.8B_finetune"))
    parser.add_argument("--dcp-path", default="/home/data/qwen3_5_0.8B_dcp")
    parser.add_argument("--generated-dir", default=str(default_generated))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv or sys.argv[1:])
    script_dir = Path(__file__).resolve().parent
    generated_dir = Path(args.generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Environment check (mock)")
    print(f"[INFO] Python: {platform.python_version()}")
    print(f"[INFO] OS: {platform.platform()}")
    print("[INFO] torch and mindspeed_mm are intentionally not imported")

    print("[STEP 2] Data validation")
    report_json = generated_dir / "validate_report.json"
    validate_code = run_command(
        [
            sys.executable,
            str(script_dir / "validate_data.py"),
            "--config",
            args.template,
            "--data",
            args.data_file,
            "--data-dir",
            args.data_dir,
            "--report-json",
            str(report_json),
        ]
    )
    if validate_code != 0:
        print("[SUMMARY] data validation failed; config generation skipped")
        return validate_code

    print("[STEP 3] Config generation")
    generated_config = generated_dir / "qwen3_5_0.8B_config.generated.yaml"
    generate_code = run_command(
        [
            sys.executable,
            str(script_dir / "generate_config.py"),
            "--template",
            args.template,
            "--model-path",
            args.model_path,
            "--data-dir",
            args.data_dir,
            "--data-file",
            args.data_file,
            "--output-dir",
            args.output_dir,
            "--dcp-path",
            args.dcp_path,
            "--output-config",
            str(generated_config),
        ]
    )
    if generate_code != 0:
        print("[SUMMARY] config generation failed")
        return generate_code

    print("[STEP 4] Training command preview")
    print("source /usr/local/Ascend/cann/set_env.sh")
    print("torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \\")
    print(f"  {generated_config}")

    print("[STEP 5] Inference command preview")
    print(f"# Load checkpoint from: {args.output_dir}")
    print("# Run the Qwen3.5 inference script after the NPU environment is available")

    print("[SUMMARY] mock skill flow completed")
    print(f"[INFO] validation report: {report_json}")
    print(f"[INFO] generated config: {generated_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
