#!/usr/bin/env python3
"""Run one Qwen3.5 image-text inference sample and emit JSON metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF-format model directory, original or DCP-exported.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_path = Path(args.model_path).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {model_path / 'config.json'}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return model_path, image_path


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    npu = getattr(torch, "npu", None)
    return "npu" if npu is not None and npu.is_available() else "cpu"


def synchronize(torch: Any, device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()


def main() -> int:
    args = parse_args()
    model_path, image_path = validate_paths(args)
    if args.dry_run:
        print(f"[PASS] model: {model_path}")
        print(f"[PASS] image: {image_path}")
        return 0

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = resolve_device(torch, args.device)
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU was requested but torch.npu is unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": args.prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(device)
    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    synchronize(torch, device)
    duration = time.perf_counter() - start
    prompt_length = inputs.input_ids.shape[1]
    generated = generated_ids[:, prompt_length:]
    output_tokens = generated.shape[1]
    output_text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result = {
        "model_path": str(model_path), "image": str(image_path), "prompt": args.prompt,
        "device": device, "dtype": args.dtype, "input_tokens": int(prompt_length),
        "output_tokens": int(output_tokens), "duration_seconds": duration,
        "tokens_per_second": output_tokens / duration if duration else 0.0,
        "output_text": output_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
