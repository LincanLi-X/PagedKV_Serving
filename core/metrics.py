from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_SOURCES = {"measured_by_our_code", "reference_from_existing"}

SUMMARY_FIELDS = [
    "experiment",
    "dataset",
    "model",
    "baseline",
    "block_size",
    "request_rate",
    "decoding_mode",
    "normalized_latency_mean",
    "normalized_latency_p50",
    "normalized_latency_p95",
    "sustainable_request_rate",
    "avg_concurrent_requests",
    "peak_concurrent_requests",
    "useful_capacity_ratio",
    "internal_fragmentation_ratio",
    "external_fragmentation_ratio",
    "prefix_memory_saving_ratio",
    "kernel_or_translation_overhead",
    "source",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_source(source: str) -> str:
    if source not in ALLOWED_SOURCES:
        allowed = ", ".join(sorted(ALLOWED_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    return source


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


@dataclass(slots=True)
class RawTraceLogger:
    """Append-only JSONL logger for auditable experiment records."""

    path: Path | str
    default_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_context.setdefault("timestamp", utc_now_iso())
        self.default_context.setdefault("source", "measured_by_our_code")
        validate_source(str(self.default_context["source"]))

    def log(self, record: dict[str, Any]) -> None:
        merged = {
            "timestamp": utc_now_iso(),
            **self.default_context,
            **record,
        }
        validate_source(str(merged.get("source", "measured_by_our_code")))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(merged), sort_keys=True) + "\n")


def build_request_record(
    *,
    experiment: str,
    run_id: str,
    system: str,
    baseline: str,
    dataset: str,
    model: str,
    block_size: int,
    request_rate: float,
    request_id: str,
    arrival_time: float,
    start_time: float,
    finish_time: float,
    prompt_len: int,
    output_len: int,
    memory_stats: dict[str, Any],
    prefix_hit_tokens: int = 0,
    decoding_mode: str = "basic_sampling",
    source: str = "measured_by_our_code",
) -> dict[str, Any]:
    validate_source(source)
    latency = max(0.0, finish_time - start_time)
    return {
        "experiment": experiment,
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "system": system,
        "baseline": baseline,
        "dataset": dataset,
        "model": model,
        "block_size": block_size,
        "request_rate": request_rate,
        "decoding_mode": decoding_mode,
        "request_id": request_id,
        "arrival_time": arrival_time,
        "start_time": start_time,
        "finish_time": finish_time,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "latency": latency,
        "normalized_latency": latency / max(1, output_len),
        "prefix_hit_tokens": prefix_hit_tokens,
        "source": source,
        **memory_stats,
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("source", "measured_by_our_code"))
        validate_source(source)
        key = (
            record.get("experiment", "unknown"),
            record.get("dataset", "unknown"),
            record.get("model", "unknown"),
            record.get("baseline", "unknown"),
            record.get("block_size", ""),
            record.get("request_rate", ""),
            record.get("decoding_mode", "basic_sampling"),
            source,
        )
        groups.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        (
            experiment,
            dataset,
            model,
            baseline,
            block_size,
            request_rate,
            decoding_mode,
            source,
        ) = key
        normalized = [
            float(row["normalized_latency"])
            for row in rows
            if row.get("normalized_latency") not in (None, "")
        ]
        running = [
            float(row.get("running_requests", row.get("active_requests", 0)) or 0)
            for row in rows
        ]
        useful_ratios = [
            float(row["useful_capacity_ratio"])
            for row in rows
            if row.get("useful_capacity_ratio") not in (None, "")
        ]
        internal_ratios = [
            float(row["internal_fragmentation_ratio"])
            for row in rows
            if row.get("internal_fragmentation_ratio") not in (None, "")
        ]
        external_ratios = [
            float(row["external_fragmentation_ratio"])
            for row in rows
            if row.get("external_fragmentation_ratio") not in (None, "")
        ]
        prefix_ratios = [
            float(row.get("prefix_memory_saving_ratio", row.get("memory_saving_ratio")))
            for row in rows
            if row.get("prefix_memory_saving_ratio", row.get("memory_saving_ratio")) not in (None, "")
        ]
        summaries.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "model": model,
                "baseline": baseline,
                "block_size": block_size,
                "request_rate": request_rate,
                "decoding_mode": decoding_mode,
                "normalized_latency_mean": _mean(normalized),
                "normalized_latency_p50": percentile(normalized, 50),
                "normalized_latency_p95": percentile(normalized, 95),
                "sustainable_request_rate": request_rate if normalized else "",
                "avg_concurrent_requests": _mean(running),
                "peak_concurrent_requests": max(running) if running else 0,
                "useful_capacity_ratio": _mean(useful_ratios),
                "internal_fragmentation_ratio": _mean(internal_ratios),
                "external_fragmentation_ratio": _mean(external_ratios),
                "prefix_memory_saving_ratio": _mean(prefix_ratios),
                "kernel_or_translation_overhead": "",
                "source": source,
            }
        )
    return summaries


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
