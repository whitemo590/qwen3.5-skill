# P1 validation and benchmark rules

## Validation layers

1. Run `validate_integration.py` on every generated YAML. Treat any failed model registration, forward-field, `.loss`, FSDP, recompute, or freeze check as blocking.
2. Run `runtime_smoke_qwen3_5.py --check-only` on file-only machines.
3. Run `runtime_smoke_qwen3_5.py` after installing the pinned Transformers, Accelerate, MindSpeed, and MindSpeed-MM environment. Require finite loss and the expected module names.
4. Run `precision_align_qwen3_5.py` with real weights before performance tuning.

## Fair performance comparison

Keep these values identical between baseline and optimized runs:

- model weights and seed;
- dataset, order, cutoff length, and preprocessing limits;
- effective batch per data-parallel rank (`micro_batch_size * gradient_accumulation_steps`);
- learning rate, warmup, iterations, precision, frozen modules, and world size.

The generated baseline disables Triton GDN and uses micro batch 1 with accumulation. The optimized config enables Triton GDN and increases micro batch while preserving effective batch.

Exclude warmup steps from throughput statistics. Report mean and P95 step time, samples/s, final loss, maximum grad norm, and peak allocated memory. Reject a performance result if loss is non-finite, the run has fewer measured steps than planned, or the effective batches differ.

## Suggested numerical gates

Use these as initial engineering gates, then replace them with official acceptance thresholds when published:

- logits cosine similarity: at least `0.999` in float32 alignment;
- mean absolute logit error: record and investigate regressions rather than hard-code across dtypes;
- optimized final loss: no material divergence from baseline over the same short run;
- no NaN/Inf in loss or gradients.
