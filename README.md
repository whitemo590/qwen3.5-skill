# qwen3.5-skill

Qwen3.5-0.8B 昇腾 NPU FSDP2 自动迁移与优化 Skill 的本地准备产物。

本仓库同时覆盖无 NPU 的离线预检，以及 NPU 环境中的权重转换、训练、推理和精度对齐入口。Mock 流程不会导入 `torch`、`torch_npu` 或 `mindspeed_mm`。

## 目录结构

```text
qwen3.5-skill/
├── SKILL.md              # Agent Skill 定义文件
├── README.md             # 人类交接与使用说明
├── validate_data.py      # COCO/MLLM 数据校验脚本
├── generate_config.py    # 基于 A 的 YAML 模板生成训练配置
├── validate_integration.py
├── runtime_smoke_qwen3_5.py
├── generate_experiment_configs.py
├── analyze_training_log.py
├── run_skill.py          # 无 NPU mock demo 入口
├── references/           # 固定的上游版本信息
├── role-a-artifacts/
│   ├── preflight_qwen3_5.py
│   ├── convert_qwen3_5_0.8B_weights.sh
│   ├── finetune_qwen3_5_0.8B.sh
│   ├── inference_qwen3_5.py
│   └── precision_align_qwen3_5.py
└── generated/            # 运行后生成的报告和配置
```

相关上游产物：

```text
role-a-artifacts/qwen3_5_0.8B_config.yaml
role-a-artifacts/finetune_qwen3_5_0.8B.sh
dataset/dataset/annotations_slim.json
dataset/dataset/COCO2017/train2017/
```

## 快速运行

先按 `references/MINDSPEED_MM_VERSION.md` 克隆官方源码，并准备数据集，然后在仓库根目录执行：

```bash
python run_skill.py \
  --mindspeed-root third_party/MindSpeed-MM \
  --data-dir /path/to/dataset \
  --data-file /path/to/dataset/annotations_slim.json
```

该命令会依次完成：

1. 检查官方 MindSpeed-MM 插件、转换器和迁移资产
2. 调用 `validate_data.py` 校验数据
3. 调用 `generate_config.py` 生成 YAML
4. 校验模型接口、`.loss` 合约和 FSDP/Recompute 模块模式
5. 生成等效 batch 的 baseline/optimized 配置
6. 检查目标运行依赖
7. 打印权重转换、训练、推理、对齐和性能报告命令

预期结果：

```text
[SUMMARY] 19/19 checks passed, 0 failed
[SUMMARY] mock skill flow completed
```

## P1 离线测试

```bash
python -m unittest discover -s tests -v
```

目标 NPU 环境安装完成后运行小随机模型构造：

```bash
python runtime_smoke_qwen3_5.py \
  --mindspeed-root third_party/MindSpeed-MM
```

生成可公平比较的性能配置：

```bash
python generate_experiment_configs.py \
  --config generated/qwen3_5_0.8B_config.generated.yaml \
  --output-dir generated/experiments \
  --effective-batch 4 \
  --train-iters 100
```

解析训练日志：

```bash
python analyze_training_log.py \
  --log logs/train.log \
  --warmup-steps 10 \
  --output-json generated/performance.json \
  --output-markdown generated/performance.md
```

## 单独校验数据

```bash
python validate_data.py \
  --config role-a-artifacts/qwen3_5_0.8B_config.yaml \
  --data dataset/dataset/annotations_slim.json \
  --data-dir dataset/dataset \
  --report-json generated/validate_report.json
```

校验内容包括：

- JSON 可读性和顶层结构
- 样本数量是否为 2000
- `images` / `messages` 字段是否与 A 的 YAML `attr` 映射一致
- `role` / `content` 字段完整性
- `user` / `assistant` 是否交替
- 首条用户消息是否包含 `<image>`
- `<image>` 数量是否等于图片数量
- 图片文件是否存在、可读、扩展名是否为 `.jpg`
- 对话轮次分布统计

## 单独生成配置

```bash
python generate_config.py \
  --template role-a-artifacts/qwen3_5_0.8B_config.yaml \
  --model-path /home/data/qwen3_5_0.8B \
  --dcp-path /home/data/qwen3_5_0.8B_dcp \
  --data-dir /home/usr/data/qwen3_5_0.8B \
  --data-file /home/usr/data/qwen3_5_0.8B/annotations_slim.json \
  --output-dir /home/usr/save/qwen3_5_0.8B_finetune \
  --output-config generated/qwen3_5_0.8B_config.generated.yaml
```

脚本只替换路径类字段，保留 A 的 FSDP2 并行策略、训练超参和 plugin 配置。

主要替换字段：

| YAML 字段 | 含义 |
|---|---|
| `data.dataset_param.preprocess_parameters.model_name_or_path` | HF 模型和 tokenizer 路径 |
| `model.model_name_or_path` | 模型配置路径 |
| `data.dataset_param.basic_parameters.dataset_dir` | 数据集根目录 |
| `data.dataset_param.basic_parameters.dataset` | MLLM JSON 标注文件 |
| `data.dataset_param.basic_parameters.cache_dir` | 数据预处理缓存目录 |
| `training.load` | DCP 权重路径 |
| `training.save` | 微调输出目录 |

## 与 A 角色产物的关系

A 角色提供：

- `role-a-artifacts/qwen3_5_0.8B_config.yaml`
- `role-a-artifacts/finetune_qwen3_5_0.8B.sh`

B 角色在本目录提供：

- `validate_data.py`：按 A 的 YAML `attr` 映射校验数据
- `generate_config.py`：以 A 的 YAML 为模板生成目标配置
- `run_skill.py`：串起校验、配置生成和训练命令预览
- `SKILL.md`：把流程固化为 Agent Skill

简化流程：

```text
A 的 YAML + COCO 数据集
        |
        v
validate_data.py
        |
        v
generate_config.py
        |
        v
run_skill.py mock demo
        |
        v
NPU 到位后接入 torchrun + mindspeed_mm/fsdp/train/trainer.py
```

## 当前验证结果

本地已验证：

- Python 语法检查通过
- `run_skill.py` 完整跑通
- 数据校验 `19/19 checks passed`
- 模型/FSDP2 集成校验 `36/36 non-failing`
- P1 离线单元测试 `3/3 passed`
- 训练日志可输出 step time、P95、samples/s、loss、grad norm 和显存统计
- baseline/optimized 配置保持相同 effective batch
- 生成的 YAML 可被 `pyyaml` 正常解析
- `SKILL.md` 已通过 Codex skill `quick_validate.py`

生成产物：

```text
generated/validate_report.json
generated/integration_report.json
generated/qwen3_5_0.8B_config.generated.yaml
generated/experiments/
```

## 固定 MindSpeed-MM 版本

```bash
git clone --depth 1 --branch 26.0.0 \
  https://github.com/Ascend/MindSpeed-MM.git \
  third_party/MindSpeed-MM
```

固定 commit 为 `08d37c0a08cefd869ac3c99b49d9fc14ee4e612a`，详见 `references/MINDSPEED_MM_VERSION.md`。

## NPU 阶段衔接

当昇腾 NPU 服务器可用后，需要：

1. 按 `install_guide.md` 安装 CANN、PyTorch、torch_npu、MindSpeed、MindSpeed-MM。
2. 下载 Qwen3.5-0.8B HF 权重。
3. 转换权重：

```bash
bash role-a-artifacts/convert_qwen3_5_0.8B_weights.sh \
  hf-to-dcp /path/to/hf_model /path/to/dcp_model
```

4. 生成服务器配置并启动训练：

```bash
MINDSPEED_MM_ROOT=/path/to/MindSpeed-MM \
bash role-a-artifacts/finetune_qwen3_5_0.8B.sh \
  generated/qwen3_5_0.8B_config.generated.yaml
```

5. 将 DCP 导出为 HF 后运行 `inference_qwen3_5.py`。
6. 使用 `precision_align_qwen3_5.py` 比较 HF 与 MindSpeed-MM logits。
