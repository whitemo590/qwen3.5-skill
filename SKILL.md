---
name: qwen35-fsdp2-migration
description: Automate file-level preparation for Qwen3.5-0.8B migration to Huawei Ascend NPU with MindSpeed-MM FSDP2. Use for validating MLLM JSON data, generating FSDP2 YAML configs, previewing no-NPU mock training flows, and guiding the later NPU fine-tuning, inference, and performance-report workflow.
---

# Qwen3.5 Ascend FSDP2 Migration Skill

Use this skill to prepare and validate the Qwen3.5-0.8B MindSpeed-MM FSDP2 fine-tuning flow before an Ascend NPU server is available. Keep all preflight work file-level only: do not import `torch`, `torch_npu`, or `mindspeed_mm` in the local mock flow.

## Prerequisites

For real NPU training, follow the project root guides:

- `../../../../install_guide.md` for CANN, PyTorch, torch_npu, MindSpeed, and MindSpeed-MM setup.
- `../../../../fsdp2_developer_migration_guide.md` for the plugin-style FSDP2 backend.

Expected real training stack:

- CANN 9.0.0, PyTorch 2.7.1, torch_npu, MindSpeed, MindSpeed-MM 26.0.0 branch.
- Qwen3.5-0.8B HF weights plus DCP-converted weights.
- COCO-style MLLM JSON data with image paths relative to `dataset_dir`.

## Workflow

```text
inputs: model_path, dcp_path, dataset_dir, dataset_json, output_dir
  |
  +-- 1. validate_data.py
  |      Validate JSON, field mapping, conversation order, <image> tokens, and image files.
  |
  +-- 2. generate_config.py
  |      Generate a runnable FSDP2 YAML from A's template and user paths.
  |
  +-- 3. run_skill.py
  |      Execute the no-NPU mock flow and print training/inference command previews.
  |
  +-- 4. real NPU phase
         Run torchrun with mindspeed_mm/fsdp/train/trainer.py and collect logs.
```

## Model Integration

Qwen3.5 uses MindSpeed-MM's plugin-style FSDP2 path:

- Trainer entry: `mindspeed_mm/fsdp/train/trainer.py`
- Model id: `qwen3_5`
- Model plugin: `mindspeed_mm/fsdp/models/qwen3_5`
- Data plugin: `mindspeed_mm/fsdp/data/datasets/huggingface`
- Weight format for training load: DCP

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

Run the full no-NPU mock flow from this directory:

```bash
python run_skill.py
```

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

## Related Skills

- `../mindspeed-mm-env-setup/SKILL.md`
- `../mindspeed-mm-vlm/SKILL.md`
- `../mindspeed-mm-pipeline/SKILL.md`
- `../mindspeed-mm-weight-prep/SKILL.md`

## References

- `../../../../数据集初步说明.md`
- `../../../../fsdp2_developer_migration_guide.md`
- `../../../../MIGRATION_GUIDE_MINDSPEED_MM_FSDP2.md`
- `role-a-artifacts/qwen3_5_0.8B_config.yaml`
- `../mindspeed-mm-vlm/references/data-format.md`
