from __future__ import annotations

from experiments.workloads.prepare_workload_traces import normalize_records


def test_prepare_workload_trace_normalizes_real_input_records() -> None:
    records = [
        {
            "request_id": "real-0",
            "prompt": "Explain paged KV cache.",
            "output": "It stores keys and values in blocks.",
            "arrival_time": 1.5,
            "data_source": "local_real_fixture",
        }
    ]

    normalized = normalize_records(records, dataset="sharegpt", arrival_rate=4.0)

    assert normalized[0]["request_id"] == "real-0"
    assert normalized[0]["dataset"] == "sharegpt"
    assert normalized[0]["arrival_time"] == 1.5
    assert normalized[0]["prompt_len"] == 4
    assert normalized[0]["output_len"] == 7
    assert normalized[0]["data_source"] == "local_real_fixture"
