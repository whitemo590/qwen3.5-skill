---
name: qwen35-fsdp2-migration
description: Prepare, validate, benchmark, and execute Qwen3.5-0.8B migration to Huawei Ascend NPU with MindSpeed-MM FSDP2. Use for MLLM data checks, MindSpeed-MM contract validation, FSDP module-plan checks, tiny-model smoke tests, baseline/optimized config generation, HF/DCP conversion, fine-tuning, inference, numerical alignment, training-log analysis, or performance reporting.
---

# Qwen3.5 Ascend FSDP2 Migration Skill

Use this skill for the Qwen3.5-0.8B MindSpeed-MM FSDP2 lifecycle. Run the mock path before an Ascend server is available; run conversion, training, inference, and alignment only in the matching NPU environment.

## Prerequisites

Before running the workflow, read `references/MINDSPEED_MM_VERSION.md` and clone the pinned MindSpeed-MM source. Follow the official MindSpeed-MM 26.0.0 Qwen3.5 installation guide for CANN, PyTorch, torch_npu, and extensions.

Expected real training stack:

- CANN 9.0.0, PyTorch 2.7.1, torch_npu, MindSpeed, MindSpeed-MM 26.0.0 branch.
- Qwen3.5-0.8B HF weights plus DCP-converted weights.
- COCO-style MLLM JSON data with image paths relative to `dataset_dir`.

## Workflow

```text
inputs: model_path, dcp_path, dataset_dir, dataset_json, output_dir
  |
  +-- 1. role-a-artifacts/preflight_qwen3_5.py
  |      Check the pinned MindSpeed-MM plugin, converter, config, and launch assets.
  |
  +-- 2. validate_data.py
  |      Validate JSON, field mapping, conversation order, <image> tokens, and image files.
  |
  +-- 3. generate_config.py
  |      Generate a runnable FSDP2 YAML from A's template and user paths.
  |
  +-- 4. validate_integration.py
  |      Check model registration, forward fields, .loss contract, and FSDP plans.
  |
  +-- 5. generate_experiment_configs.py
  |      Generate controlled baseline and optimized benchmark configs.
  |
  +-- 6. runtime_smoke_qwen3_5.py
  |      Build a tiny random model when the target runtime is installed.
  |
  +-- 7. role-a-artifacts/convert_qwen3_5_0.8B_weights.sh
  |      Convert HF weights to DCP with the 0.8B tied-weight mapping.
  |
  +-- 8. role-a-artifacts/finetune_qwen3_5_0.8B.sh
  |      Run torchrun with mindspeed_mm/fsdp/train/trainer.py.
  |
  +-- 9. role-a-artifacts/inference_qwen3_5.py
  |      Validate generated text and inference throughput.
  |
  +-- 10. role-a-artifacts/precision_align_qwen3_5.py
  |      Compare Hugging Face and MindSpeed-MM logits.
  |
  +-- 11. analyze_training_log.py
         Produce JSON/Markdown performance evidence and baseline comparisons.
```

## Model Integration

Qwen3.5 uses MindSpeed-MM's plugin-style FSDP2 path:

- Trainer entry: `mindspeed_mm/fsdp/train/trainer.py`
- Model id: `qwen3_5`
- Model plugin: `mindspeed_mm/fsdp/models/qwen3_5`
- Data plugin: `mindspeed_mm/fsdp/data/datasets/huggingface`
- Weight format for training load: DCP

Use the pinned official implementation instead of copying an unmodified Hugging Face model file:

```text
third_party/MindSpeed-MM
branch: 26.0.0
commit: 08d37c0a08cefd869ac3c99b49d9fc14ee4e612a
```

The YAML template to adapt is:

```text
role-a-artifacts/qwen3_5_0.8B_config.yaml
```

## Data Format

The dataset must be a JSON list. Each sample should use MLLM fields matching `data.dataset_param.attr` in the YAML:

```json
{
  "images": ["COCO2017/train2017/000000033471.jpg"],
  "messages": [
    {"role": "user", "content": "<image>\nWhat are the colors of the bus?"},
    {"role": "assistant", "content": "The bus is white and red."}
  ]
}
```

Current COCO slim dataset facts:

| Item | Value |
|---|---:|
| Samples | 2,000 |
| Unique images | 2,000 |
| Total messages | 18,132 |
| Average QA rounds | 4.53 |
| Round range | 1-6 |

Field alignment with A's YAML:

| YAML attr | Value | Dataset field |
|---|---|---|
| `images` | `images` | `images` |
| `messages` | `messages` | `messages` |
| `role_tag` | `role` | `role` |
| `content_tag` | `content` | `content` |
| `user_tag` | `user` | `user` |
| `assistant_tag` | `assistant` | `assistant` |

## Quick Start

Run the no-NPU mock flow after cloning MindSpeed-MM and preparing an MLLM dataset:

```bash
python run_skill.py \
  --mindspeed-root third_party/MindSpeed-MM \
  --data-dir /path/to/dataset \
  --data-file /path/to/dataset/annotations_slim.json
```

The mock flow performs source/plugin preflight, validates dataset references, generates YAML, and prints concrete weight conversion, training, inference, and alignment commands.

Run the target-environment tiny-model test after installing the pinned dependencies:

```bash
python runtime_smoke_qwen3_5.py --mindspeed-root third_party/MindSpeed-MM
```

Read `references/P1_VALIDATION.md` before comparing baseline and optimized runs.

Validate data directly:

```bash
python validate_data.py \
  --config "role-a-artifacts/qwen3_5_0.8B_config.yaml" \
  --data "dataset/dataset/annotations_slim.json" \
  --data-dir "dataset/dataset"
```

Generate a YAML config:

```bash
python generate_config.py \
  --template "role-a-artifacts/qwen3_5_0.8B_config.yaml" \
  --model-path "/home/data/qwen3_5_0.8B" \
  --dcp-path "/home/data/qwen3_5_0.8B_dcp" \
  --data-dir "/home/usr/data/qwen3_5_0.8B" \
  --data-file "/home/usr/data/qwen3_5_0.8B/annotations_slim.json" \
  --output-dir "/home/usr/save/qwen3_5_0.8B_finetune" \
  --output-config "./generated/qwen3_5_0.8B_config.generated.yaml"
```

## YAML Fields To Update

When moving from local mock validation to the NPU server, update these paths:

| YAML field | Meaning |
|---|---|
| `data.dataset_param.preprocess_parameters.model_name_or_path` | HF model/tokenizer path |
| `model.model_name_or_path` | HF model config path |
| `data.dataset_param.basic_parameters.dataset_dir` | Dataset root used to resolve relative image paths |
| `data.dataset_param.basic_parameters.dataset` | MLLM JSON annotation path |
| `data.dataset_param.basic_parameters.cache_dir` | Preprocessing cache path |
| `training.load` | DCP checkpoint load path |
| `training.save` | Fine-tuned checkpoint output path |

Keep `template: qwen3_vl_nothink`, `dataset_type: huggingface`, and the `attr` mapping unchanged for the current dataset.

## FAQ

**Data validation fails because the YAML paths do not exist.**  
A's YAML uses server placeholder paths. Pass local `--data` and `--data-dir` overrides when validating on this machine.

**A future dataset uses `conversations/from/value`.**  
Either convert it to MLLM JSON or update the YAML `attr` mapping to match the new fields.

**Training runs out of memory.**  
Set `training.micro_batch_size` to `1`, increase `gradient_accumulation_steps`, and keep `parallel.recompute: true`.

**Torchrun reports a bind port conflict.**  
Change `MASTER_PORT` in the launch script or stop the stale torchrun process.

## References

- `references/MINDSPEED_MM_VERSION.md`
- `references/P1_VALIDATION.md`
- `role-a-artifacts/qwen3_5_0.8B_config.yaml`
- Official Qwen3.5 guide: `https://github.com/Ascend/MindSpeed-MM/tree/26.0.0/examples/qwen3_5`
