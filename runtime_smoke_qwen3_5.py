#!/usr/bin/env python3
"""Construct a tiny Qwen3.5 model and verify the MindSpeed-MM training contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REQUIRED_MODULES = ("torch", "transformers", "accelerate", "mindspeed", "mindspeed_mm")


def dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mindspeed-root", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.mindspeed_root).resolve()
    plugin = root / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
    status = dependency_status()
    result: dict[str, object] = {"dependencies": status, "plugin": str(plugin), "plugin_exists": plugin.is_file()}
    print(json.dumps(result, indent=2))
    if args.check_only:
        return 0 if plugin.is_file() else 1
    missing = [name for name, available in status.items() if not available]
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing))

    sys.path.insert(0, str(root))
    import torch
    from transformers import Qwen3_5Config
    from mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text_config = {
        "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "max_position_embeddings": 64, "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 2, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "tie_word_embeddings": True,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25, "mrope_section": [1, 0, 0],
        },
    }
    vision_config = {
        "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "patch_size": 2, "spatial_merge_size": 1, "temporal_patch_size": 1,
        "out_hidden_size": 32, "num_position_embeddings": 16,
    }
    config = Qwen3_5Config(
        text_config=text_config, vision_config=vision_config,
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, tie_word_embeddings=True,
    )
    config.text_config.use_triton_gdn = False
    model = Qwen3_5ForConditionalGeneration._from_config(config).float().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"Invalid loss: {output.loss}")
    names = {name for name, _ in model.named_modules()}
    expected = {"model.visual.blocks.0", "model.language_model.layers.0", "model.language_model.embed_tokens", "lm_head"}
    missing_names = sorted(expected - names)
    if missing_names:
        raise RuntimeError(f"Missing module names: {missing_names}")
    result.update({
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(output.loss.item()), "logits_shape": list(output.logits.shape),
        "required_module_names": sorted(expected),
    })
    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
