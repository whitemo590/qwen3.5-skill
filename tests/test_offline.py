from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class OfflineWorkflowTests(unittest.TestCase):
    def run_script(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / script), *args],
            text=True, encoding="utf-8", errors="replace", capture_output=True,
        )

    def test_log_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            output_json = Path(temp) / "report.json"
            output_md = Path(temp) / "report.md"
            result = self.run_script(
                "analyze_training_log.py", "--log", str(ROOT / "tests/fixtures/sample_train.txt"),
                "--warmup-steps", "1", "--output-json", str(output_json),
                "--output-markdown", str(output_md),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["measured_steps"], 4)
            self.assertAlmostEqual(payload["summary"]["step_time_ms"]["mean"], 400.0)
            self.assertAlmostEqual(payload["summary"]["samples_per_second"], 10.0)
            self.assertEqual(payload["summary"]["memory"]["peak_max_allocated_mb"], 2048.0)

    def test_experiment_configs_keep_effective_batch_equal(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_script(
                "generate_experiment_configs.py", "--config",
                str(ROOT / "role-a-artifacts/qwen3_5_0.8B_config.yaml"),
                "--output-dir", temp, "--effective-batch", "4", "--train-iters", "20",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            baseline = yaml.safe_load((Path(temp) / "qwen3_5_0.8B_baseline.yaml").read_text(encoding="utf-8"))
            optimized = yaml.safe_load((Path(temp) / "qwen3_5_0.8B_optimized.yaml").read_text(encoding="utf-8"))
            b_train, o_train = baseline["training"], optimized["training"]
            self.assertEqual(b_train["micro_batch_size"] * b_train["gradient_accumulation_steps"], 4)
            self.assertEqual(o_train["micro_batch_size"] * o_train["gradient_accumulation_steps"], 4)
            self.assertFalse(baseline["model"]["use_triton_gdn"])
            self.assertTrue(optimized["model"]["use_triton_gdn"])
            self.assertEqual(b_train["train_iters"], o_train["train_iters"])

    def test_integration_validator_with_contract_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            ms = Path(temp) / "MindSpeed-MM"
            plugin = ms / "mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py"
            modelhub = ms / "mindspeed_mm/fsdp/models/modelhub.py"
            engine = ms / "mindspeed_mm/fsdp/train/train_engine.py"
            for path in (plugin, modelhub, engine):
                path.parent.mkdir(parents=True, exist_ok=True)
            plugin.write_text(
                '@model_register.register("qwen3_5")\n'
                'class Qwen3_5ForConditionalGeneration:\n'
                '    def forward(self, input_ids=None, attention_mask=None, position_ids=None, '
                'past_key_values=None, inputs_embeds=None, labels=None, pixel_values=None, '
                'pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, '
                'cache_position=None, logits_to_keep=0, **kwargs):\n'
                '        loss = None\n'
                '        return Qwen3_5CausalLMOutputWithPast(loss=loss)\n'
                'use_triton_gdn = True\n', encoding="utf-8"
            )
            modelhub.write_text("model_cls = model_register.get(model_id)\n", encoding="utf-8")
            engine.write_text("loss = output.loss\n", encoding="utf-8")
            result = self.run_script(
                "validate_integration.py", "--config",
                str(ROOT / "role-a-artifacts/qwen3_5_0.8B_config.yaml"),
                "--mindspeed-root", str(ms), "--strict-linux-paths",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("36/36", result.stdout)


if __name__ == "__main__":
    unittest.main()
