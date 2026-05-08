from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import SUMMARY_FIELDS, summarize_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(raw_files: list[Path], summary_path: Path) -> dict[str, Any]:
    return {
        "fig3_latency_vs_request_rate.png": {
            "paper_location": "Final Report Figure 3 / Presentation Slide 10",
            "script": "experiments/plot_figures.py",
            "input_summary": "results/summary/e1_latency_vs_request_rate.csv",
            "raw_inputs": ["results/raw/e1_throughput/*.jsonl"],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "e2_memory_waste_breakdown.png": {
            "paper_location": "Final Report Table 4 / Presentation Slide 11",
            "script": "experiments/plot_figures.py",
            "input_summary": "results/summary/e2_fragmentation.csv",
            "raw_inputs": ["results/raw/e2_memory_efficiency/*.jsonl"],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "e3_sharegpt_vs_alpaca.png": {
            "paper_location": "Final Report Section 4.5 / Presentation Slide 11",
            "script": "experiments/plot_figures.py",
            "input_summary": "results/summary/e3_workload_sensitivity.csv",
            "raw_inputs": ["results/raw/e3_workload_sensitivity/*.jsonl"],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "fig_prefix_sharing_savings.png": {
            "paper_location": "Final Report Section 4.6 / Presentation Slide 12",
            "script": "experiments/plot_figures.py",
            "input_summary": "results/summary/e4_prefix_sharing.csv",
            "raw_inputs": ["results/raw/e4_prefix_sharing/*.jsonl"],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "table4_memory_efficiency.csv": {
            "paper_location": "Final Report Table 4 / Presentation Slides 11-12",
            "script": "experiments/report_table4.py",
            "input_summaries": [
                "results/summary/e2_fragmentation.csv",
                "results/summary/e4_prefix_sharing.csv",
            ],
            "raw_inputs": [
                "results/raw/e2_memory_efficiency/*.jsonl",
                "results/raw/e4_prefix_sharing/*.jsonl",
            ],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "fig4_block_size_ablation.png": {
            "paper_location": "Final Report Figure 4 / Presentation Slide 13",
            "script": "experiments/plot_figures.py",
            "input_summary": "results/summary/e5_block_size.csv",
            "raw_inputs": ["results/raw/e5_block_size/*.jsonl"],
            "block_sizes": [16, 32, 128],
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        },
        "combined_summary.csv": {
            "paper_location": "All experiment sections",
            "script": "experiments/summarize_results.py",
            "input_raw_files": [str(path.relative_to(ROOT)) for path in raw_files],
            "output_summary": str(summary_path.relative_to(ROOT)),
            "source_policy": "development artifacts use simulated_for_development; final artifacts must use measured_by_our_code",
        }
    }


def write_static_tables(summary_dir: Path) -> None:
    write_csv(
        summary_dir / "table1_components.csv",
        [
            {"component": "Scheduler", "primary_data_structures": "Request queue; active-sequence set; batch metadata", "responsibility": "Admits requests and coordinates prefill/decode iterations", "source": "measured_by_our_code"},
            {"component": "Model runner", "primary_data_structures": "Step execution state; gather/scatter metadata", "responsibility": "Executes prefill/decode and materializes KV states", "source": "measured_by_our_code"},
            {"component": "KV cache manager", "primary_data_structures": "Physical block pool; block tables; descriptors", "responsibility": "Allocates, maps, and reclaims fixed-size KV blocks", "source": "measured_by_our_code"},
            {"component": "Sharing manager", "primary_data_structures": "Reference counts; copy-on-write metadata", "responsibility": "Preserves shared prefix blocks and isolates divergent paths", "source": "measured_by_our_code"},
        ],
    )
    write_csv(
        summary_dir / "table2_eval_settings.csv",
        [
            {"category": "Models", "evaluation_setting": "OPT-13B on 1 GPU; OPT-66B on 4 GPUs; OPT-175B on 8 GPUs", "source": "draft_report_configuration"},
            {"category": "Workloads", "evaluation_setting": "Tokenized ShareGPT and Alpaca traces with Poisson request arrivals", "source": "draft_report_configuration"},
            {"category": "System under test", "evaluation_setting": "PagedKV serving engine with block-based KV-cache allocation and prefix sharing", "source": "draft_report_configuration"},
            {"category": "Baselines", "evaluation_setting": "FasterTransformer-like contiguous allocation; Orca(Oracle); Orca(Pow2); Orca(Max)", "source": "draft_report_configuration"},
            {"category": "Metrics", "evaluation_setting": "Normalized latency; sustainable request rate; concurrent active requests; memory saving; KV-cache waste; kernel overhead", "source": "draft_report_configuration"},
            {"category": "Decoding modes", "evaluation_setting": "Basic sampling; parallel sampling; beam search; shared-prefix serving", "source": "draft_report_configuration"},
            {"category": "Block sizes", "evaluation_setting": "Default block size 16 tokens; ablation over 8, 16, 32, 64, and 128 tokens", "source": "draft_report_configuration"},
        ],
    )


def write_artifact_mapping(path: Path) -> None:
    path.write_text(
        """# Artifact-to-Paper Mapping

| Final Report / Slides | Claim | Code Entry | Raw Data | Summary | Figure/Table |
|---|---|---|---|---|---|
| Final Report Fig. 1, Slide 5 | System overview | `core/engine.py` | N/A | N/A | `results/figures/fig1_system_overview.png` |
| Final Report Fig. 2, Slide 6 | Block table maps logical blocks to physical blocks | `core/paged_kv_cache.py` | `results/raw/e2_memory_efficiency/*.jsonl` | `results/summary/e2_fragmentation.csv` | `results/figures/fig2_block_table.png` |
| Final Report Fig. 3, Slide 10 | Higher sustainable throughput under load | `experiments/run_e1_throughput.py` | `results/raw/e1_throughput/*.jsonl` | `results/summary/e1_latency_vs_request_rate.csv` | `results/figures/fig3_latency_vs_request_rate.png` |
| Final Report Table 1, Slide 8 | Component responsibilities | source files under `core/` | N/A | `results/summary/table1_components.csv` | Table 1 |
| Final Report Table 2, Slide 9 | Evaluation settings | `experiments/configs/*.yaml` | N/A | `results/summary/table2_eval_settings.csv` | Table 2 |
| Final Report Table 3 | Throughput targets and baseline comparison | `experiments/run_e1_throughput.py` | `results/raw/e1_throughput/*.jsonl` | `results/summary/table3_throughput_targets_and_measured.csv` | Table 3 |
| Final Report Table 4, Slide 11/12 | Memory efficiency and prefix sharing | `experiments/run_e2_memory_efficiency.py`, `experiments/run_e4_prefix_sharing.py` | `results/raw/e2_memory_efficiency/*.jsonl`, `results/raw/e4_prefix_sharing/*.jsonl` | `results/summary/table4_memory_efficiency.csv` | Table 4 |
| Final Report Fig. 4, Slide 13 | Block size trade-off | `experiments/run_e5_block_size_ablation.py` | `results/raw/e5_block_size/*.jsonl` | `results/summary/e5_block_size.csv` | `results/figures/fig4_block_size_ablation.png` |

Development artifacts are marked `simulated_for_development`; final submission artifacts must be regenerated from real runs and marked `measured_by_our_code`.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw JSONL experiment logs into CSV.")
    parser.add_argument("--raw-root", default=str(ROOT / "results" / "raw"))
    parser.add_argument("--summary-dir", default=str(ROOT / "results" / "summary"))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    summary_dir = Path(args.summary_dir)
    raw_files = sorted(path for path in raw_root.rglob("*.jsonl") if path.is_file())
    records: list[dict[str, Any]] = []
    for path in raw_files:
        records.extend(read_jsonl(path))

    summaries = summarize_records(records)
    combined_path = summary_dir / "combined_summary.csv"
    write_csv(combined_path, summaries, SUMMARY_FIELDS)
    write_static_tables(summary_dir)
    write_artifact_mapping(ROOT / "results" / "artifact_to_paper_mapping.md")

    manifest = build_manifest(raw_files, combined_path)
    manifest_path = ROOT / "results" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"read {len(records)} raw records from {len(raw_files)} files")
    print(f"wrote {combined_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
