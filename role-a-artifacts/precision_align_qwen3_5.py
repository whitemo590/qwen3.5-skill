#!/usr/bin/env python3
"""Compare Hugging Face Qwen3.5 and the MindSpeed-MM patched model on one sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mindspeed-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--logits-to-keep", type=int, default=1)
    parser.add_argument("--output-json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sample(data_path: Path, data_dir: Path, index: int) -> tuple[Path, str]:
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must be a non-empty JSON list")
    sample = samples[index]
    image = Path(sample["images"][0])
    image = image if image.is_absolute() else data_dir / image
    first_user = next(message for message in sample["messages"] if message["role"] == "user")
    prompt = first_user["content"].replace("<image>", "").strip()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    return image.resolve(), prompt


def tensor_metrics(torch: Any, reference: Any, candidate: Any) -> dict[str, float]:
    ref, cand = reference.float(), candidate.float()
    diff = (ref - cand).abs()
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "cosine_similarity": float(cosine.item()),
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    mindspeed_root = Path(args.mindspeed_root).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    required = [
        model_path / "config.json",
        mindspeed_root / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py",
        data_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    image, prompt = load_sample(data_path, data_dir, args.sample_index)
    if args.dry_run:
        print(f"[PASS] model: {model_path}")
        print(f"[PASS] MindSpeed-MM: {mindspeed_root}")
        print(f"[PASS] sample image: {image}")
        print(f"[INFO] prompt: {prompt}")
        return 0

    import sys
    sys.path.insert(0, str(mindspeed_root))
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration as HFModel
    from mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as MindSpeedModel

    if args.device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU was requested but torch.npu is unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image)}, {"type": "text", "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(args.device)
    reference = HFModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).to(args.device).eval()
    candidate = MindSpeedModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).to(args.device).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    with torch.inference_mode():
        ref_output = reference(**inputs, use_cache=False, logits_to_keep=args.logits_to_keep)
        candidate_output = candidate(**inputs, use_cache=False, logits_to_keep=args.logits_to_keep)
    if ref_output.logits.shape != candidate_output.logits.shape:
        raise RuntimeError(f"Logit shape mismatch: {ref_output.logits.shape} vs {candidate_output.logits.shape}")
    metrics = {
        "sample_index": args.sample_index, "device": args.device, "dtype": args.dtype,
        "shape": list(ref_output.logits.shape),
        **tensor_metrics(torch, ref_output.logits, candidate_output.logits),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
