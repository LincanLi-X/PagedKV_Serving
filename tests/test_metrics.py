from __future__ import annotations

import json

from core.metrics import RawTraceLogger, build_request_record, summarize_records, validate_source


def test_raw_trace_logger_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    logger = RawTraceLogger(path, {"experiment": "unit", "source": "measured_by_our_code"})
    logger.log({"request_id": "r0", "normalized_latency": 1.0})

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["experiment"] == "unit"
    assert row["source"] == "measured_by_our_code"


def test_development_source_is_explicitly_allowed_for_non_submission_runs(tmp_path) -> None:
    path = tmp_path / "raw_dev.jsonl"
    logger = RawTraceLogger(path, {"experiment": "unit", "source": "simulated_for_development"})
    logger.log({"request_id": "r0", "normalized_latency": 1.0})

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["source"] == "simulated_for_development"


def test_build_request_record_and_summary_fields() -> None:
    record = build_request_record(
        experiment="e1_throughput",
        run_id="run",
        system="pagedkv",
        baseline="paged",
        dataset="sharegpt",
        model="opt-13b-dev",
        block_size=16,
        request_rate=4.0,
        request_id="r0",
        arrival_time=0.0,
        start_time=1.0,
        finish_time=3.0,
        prompt_len=10,
        output_len=4,
        memory_stats={
            "useful_capacity_ratio": 0.9,
            "internal_fragmentation_ratio": 0.1,
            "external_fragmentation_ratio": 0.0,
        },
    )
    summary = summarize_records([record])[0]

    assert record["normalized_latency"] == 0.5
    assert summary["experiment"] == "e1_throughput"
    assert summary["normalized_latency_mean"] == 0.5
    assert summary["source"] == "measured_by_our_code"


def test_memory_accounting_invariant_useful_plus_internal_never_exceeds_reserved() -> None:
    record = build_request_record(
        experiment="e2_memory_efficiency",
        run_id="run",
        system="pagedkv",
        baseline="pagedkv",
        dataset="sharegpt",
        model="opt-13b-dev",
        block_size=16,
        request_rate=4.0,
        request_id="r0",
        arrival_time=0.0,
        start_time=0.0,
        finish_time=1.0,
        prompt_len=13,
        output_len=5,
        memory_stats={
            "useful_tokens": 18,
            "reserved_token_slots": 32,
            "internal_waste_tokens": 14,
            "external_waste_tokens": 0,
            "useful_capacity_ratio": 18 / 32,
            "internal_fragmentation_ratio": 14 / 32,
            "external_fragmentation_ratio": 0.0,
        },
    )

    assert record["useful_tokens"] + record["internal_waste_tokens"] <= record["reserved_token_slots"]


def test_invalid_metric_source_is_rejected() -> None:
    try:
        validate_source("unlabeled_result")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown source labels should be rejected")
