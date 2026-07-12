#!/usr/bin/env python3
"""Parse MindSpeed-MM FSDP2 training logs into JSON and Markdown metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


TRAIN_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\s*\|"
    r".*?consumed samples:\s*(?P<consumed>\d+)\s*\|"
    r".*?elapsed time per iteration \(ms\):\s*(?P<time>[0-9.]+)\s*\|"
    r".*?learning rate:\s*(?P<lr>[0-9.Ee+-]+)\s*\|"
    r".*?global batch size:\s*(?P<gbs>\d+)\s*\|"
    r".*?loss:\s*(?P<loss>[0-9.Ee+-]+)\s*\|"
    r"(?:.*?grad norm:\s*(?P<grad>[0-9.Ee+-]+)\s*\|)?"
)
MEMORY_PATTERN = re.compile(
    r"\[Rank\s+(?P<rank>\d+)\].*?memory \(MB\)\s*\|\s*allocated:\s*(?P<allocated>[0-9.]+)"
    r"\s*\|\s*max allocated:\s*(?P<max_allocated>[0-9.]+)"
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def number_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values), "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90), "p95": percentile(values, 0.95),
        "min": min(values) if values else None, "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    steps = []
    for match in TRAIN_PATTERN.finditer(text):
        item = match.groupdict()
        steps.append({
            "iteration": int(item["iteration"]), "total_iterations": int(item["total"]),
            "consumed_samples": int(item["consumed"]), "step_time_ms": float(item["time"]),
            "learning_rate": float(item["lr"]), "global_batch_size": int(item["gbs"]),
            "loss": float(item["loss"]), "grad_norm": float(item["grad"]) if item["grad"] else None,
        })
    memory = [
        {"rank": int(match["rank"]), "allocated_mb": float(match["allocated"]),
         "max_allocated_mb": float(match["max_allocated"])}
        for match in MEMORY_PATTERN.finditer(text)
    ]
    return steps, memory


def summarize(steps: list[dict[str, Any]], memory: list[dict[str, Any]], warmup_steps: int) -> dict[str, Any]:
    measured = steps[warmup_steps:] if len(steps) > warmup_steps else []
    times = [step["step_time_ms"] for step in measured]
    losses = [step["loss"] for step in steps]
    grad_norms = [step["grad_norm"] for step in steps if step["grad_norm"] is not None]
    mean_time = statistics.fmean(times) if times else None
    gbs = measured[0]["global_batch_size"] if measured else steps[0]["global_batch_size"] if steps else None
    throughput = gbs * 1000.0 / mean_time if gbs and mean_time else None
    return {
        "parsed_steps": len(steps), "warmup_steps_excluded": warmup_steps,
        "measured_steps": len(measured), "global_batch_size": gbs,
        "step_time_ms": number_summary(times), "samples_per_second": throughput,
        "loss": {
            "first": losses[0] if losses else None, "last": losses[-1] if losses else None,
            "min": min(losses) if losses else None, "max": max(losses) if losses else None,
        },
        "grad_norm": number_summary(grad_norms),
        "memory": {
            "records": len(memory),
            "peak_max_allocated_mb": max((item["max_allocated_mb"] for item in memory), default=None),
            "by_rank": memory,
        },
    }


def compare(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    current_tps = summary.get("samples_per_second")
    baseline_tps = baseline.get("samples_per_second")
    current_time = summary.get("step_time_ms", {}).get("mean")
    baseline_time = baseline.get("step_time_ms", {}).get("mean")
    return {
        "throughput_change_percent":
            (current_tps / baseline_tps - 1) * 100 if current_tps and baseline_tps else None,
        "step_time_reduction_percent":
            (1 - current_time / baseline_time) * 100 if current_time and baseline_time else None,
        "final_loss_delta":
            summary.get("loss", {}).get("last") - baseline.get("loss", {}).get("last")
            if summary.get("loss", {}).get("last") is not None
            and baseline.get("loss", {}).get("last") is not None else None,
    }


def markdown_report(log_path: Path, summary: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    time = summary["step_time_ms"]
    loss = summary["loss"]
    memory = summary["memory"]
    lines = [
        "# Qwen3.5 training performance report", "", f"- Log: `{log_path}`",
        f"- Parsed steps: {summary['parsed_steps']}",
        f"- Measured steps: {summary['measured_steps']} (warmup excluded: {summary['warmup_steps_excluded']})",
        "", "## Metrics", "",
        "| Metric | Value |", "|---|---:|",
        f"| Global batch size | {summary['global_batch_size']} |",
        f"| Mean step time (ms) | {time['mean']:.3f} |" if time["mean"] is not None else "| Mean step time (ms) | N/A |",
        f"| P95 step time (ms) | {time['p95']:.3f} |" if time["p95"] is not None else "| P95 step time (ms) | N/A |",
        f"| Samples/s | {summary['samples_per_second']:.3f} |" if summary["samples_per_second"] is not None else "| Samples/s | N/A |",
        f"| First loss | {loss['first']} |", f"| Last loss | {loss['last']} |",
        f"| Peak allocated memory (MB) | {memory['peak_max_allocated_mb']} |",
    ]
    if comparison:
        lines += ["", "## Baseline comparison", "", "| Metric | Change |", "|---|---:|"]
        for key, value in comparison.items():
            lines.append(f"| {key} | {value:.3f} |" if value is not None else f"| {key} | N/A |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--baseline-json", help="JSON produced by this script for a baseline run.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).resolve()
    steps, memory = parse_log(log_path)
    if not steps:
        raise RuntimeError(f"No training steps found in {log_path}")
    summary = summarize(steps, memory, args.warmup_steps)
    comparison = None
    if args.baseline_json:
        baseline_payload = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        comparison = compare(summary, baseline_payload["summary"])
    payload = {"log": str(log_path), "summary": summary, "comparison": comparison, "steps": steps}
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_markdown.write_text(markdown_report(log_path, summary, comparison), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
