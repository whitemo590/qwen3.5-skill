qwen3_5_0.8B_config.yaml

```python
41 model_name_or_path: &HF_MODEL_LOAD_PATH /home/data/qwen3_5_0.8B
58 dataset_dir: /home/usr/data/qwen3_5_0.8B/
59 dataset: &DATASET_PATH /home/usr/data/qwen3_5_0.8B/mllm_format_llava_instruct_data.json
116  load: /home/data/qwen3_5_0.8B_dcp
117  save: /home/usr/save/qwen3_5_0.8B_finetune
```

finetune_qwen3_5_0.8B.sh
```
2 source /usr/local/Ascend/cann/set_env.sh
```

须更新为实际路径