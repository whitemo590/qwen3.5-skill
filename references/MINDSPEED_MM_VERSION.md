# MindSpeed-MM source pin

- Repository: `https://github.com/Ascend/MindSpeed-MM.git`
- Branch: `26.0.0`
- Commit: `08d37c0a08cefd869ac3c99b49d9fc14ee4e612a`
- Local checkout: `third_party/MindSpeed-MM`

Qwen3.5 uses the Transformers source revision documented by MindSpeed-MM:

- Repository: `https://github.com/huggingface/transformers.git`
- Commit: `fc9137225880a9d03f130634c20f9dbe36a7b8bf`

The MindSpeed-MM installer and `examples/qwen3_5/install_extensions.sh` should be preferred over installing the public PyPI Transformers 4.x release, which does not contain Qwen3.5.

Clone the pinned source with:

```bash
git clone --depth 1 --branch 26.0.0 \
  https://github.com/Ascend/MindSpeed-MM.git \
  third_party/MindSpeed-MM
```

The official checkout contains the required implementation:

```text
mindspeed_mm/fsdp/models/qwen3_5/
checkpoint/vlm_model/converters/qwen3_5.py
examples/qwen3_5/
```

Use this implementation because it includes FSDP2 model registration, context parallel support, Ascend handling, and the Triton Gated DeltaNet path.
