# Experiments

This directory contains reproducible experiment utilities for the course project.

- Workload preparation is available in `Workloads/prepare_workload_traces.py`.
- Raw-log summarization is available in `summarize_results.py`.
- Baseline policy wrappers live in `baselines/`.

- Use `--sharegpt-input` for the real ShareGPT-format JSON/JSONL file.
- Use `--alpaca-input` for the real Alpaca-format JSON/JSONL file.
- The script normalizes fields into `request_id`, `arrival_time`, `prompt_len`, `output_len`, `prompt_text`, and `dataset`.

Every normalized row records `dataset`, `arrival_time`, `prompt_len`, `output_len`, `prompt_text`, and `data_source` so later experiment scripts can be audited.
