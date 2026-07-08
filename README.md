# qwen3.5-skill

Qwen3.5-0.8B 昇腾 NPU FSDP2 自动迁移与优化 Skill 的本地准备产物。

本目录面向“无 NPU 服务器”的初赛阶段，目标是在纯文件级环境下完成数据校验、配置生成和 mock 流程演示。当前脚本不会 import `torch`、`torch_npu`、`mindspeed_mm`，也不会启动真实训练。

## 目录结构

```text
qwen3.5-skill/
├── SKILL.md              # Agent Skill 定义文件
├── README.md             # 人类交接与使用说明
├── validate_data.py      # COCO/MLLM 数据校验脚本
├── generate_config.py    # 基于 A 的 YAML 模板生成训练配置
├── run_skill.py          # 无 NPU mock demo 入口
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

在仓库根目录执行：

```bash
python run_skill.py
```

该命令会依次完成：

1. mock 环境检查
2. 调用 `validate_data.py` 校验数据
3. 调用 `generate_config.py` 生成 YAML
4. 打印未来 NPU 服务器上的训练命令预览
5. 输出报告和生成配置路径

预期结果：

```text
[SUMMARY] 19/19 checks passed, 0 failed
[SUMMARY] mock skill flow completed
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
- 生成的 YAML 可被 `pyyaml` 正常解析
- `SKILL.md` 已通过 Codex skill `quick_validate.py`

生成产物：

```text
generated/validate_report.json
generated/qwen3_5_0.8B_config.generated.yaml
```

## NPU 阶段衔接

当昇腾 NPU 服务器可用后，需要：

1. 按 `install_guide.md` 安装 CANN、PyTorch、torch_npu、MindSpeed、MindSpeed-MM。
2. 准备 Qwen3.5-0.8B HF 权重和 DCP 权重。
3. 把生成配置里的路径替换为服务器真实路径。
4. 使用 A 的启动脚本或等价命令启动训练：

```bash
source /usr/local/Ascend/cann/set_env.sh
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
  qwen3_5_0.8B_config.generated.yaml
```
