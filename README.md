# PagedKV_Serving

**PagedKV_Serving** is the Code Repository for the `Advanced Operating System course project`: "Efficient KV-Cache Management for Large Language Model Service Systems." The project studies KV-cache management for online LLM serving from an operating-system perspective: instead of reserving one contiguous KV buffer per request, the serving runtime stores KV states in fixed-size blocks, maintains per-request block tables, allocates blocks lazily, and supports prefix sharing with reference counts and copy-on-write.

## Repository Layout

```text
PagedKV_Serving/
├── core/                         # Serving runtime and KV-cache data structures
│   ├── paged_kv_cache.py         # Paged KV block allocator, block tables, ref counts, CoW
│   ├── prefix_cache.py           # Full-block prefix cache and prefix-hit accounting
│   ├── contiguous_kv_cache.py    # Contiguous baseline cache policies
│   ├── scheduler.py              # Waiting/running request scheduler
│   ├── engine.py, llm.py         # Single-process serving engine interface
│   ├── model_runner.py           # Hugging Face runner and fallback local model
│   └── metrics.py                # Raw logging, source labels, and summary helpers
├── api/                          # OpenAI-compatible chat API
│   ├── schemas.py                # Pydantic request and response schemas
│   └── openai_server.py          # /v1/chat/completions endpoint
├── experiments/                  # E1-E5 experiment pipeline
│   ├── run_e1_throughput.py      # E1 throughput under offered load
│   ├── run_e2_memory_efficiency.py
│   ├── run_e3_workload_sensitivity.py
│   ├── run_e4_prefix_sharing.py
│   ├── run_e5_block_size_ablation.py
│   ├── baselines/                # Orca/FasterTransformer-style baseline wrappers
│   ├── configs/                  # YAML experiment configurations
│   ├── workloads/                # Workload trace utilities
│   ├── plot_figures.py           # Figure generation
│   ├── report_table4.py          # Report-facing Table 4 builder
│   └── summarize_results.py      # Raw JSONL to summary CSV and manifest
├── results/                      # Our experimental results
│   ├── raw/                      # Per-request JSONL logs
│   ├── summary/                  # Experimental results .csv tables and metrics
│   └── figures/                  # Experimental results figure plots
├── tests/
│   ├── test_paged_kv_cache.py
│   ├── test_prefix_cache.py
│   ├── test_contiguous_baselines.py
│   ├── test_scheduler.py
│   ├── test_metrics.py
│   └── test_api_response.py
├── run_local_chat.py             # Local command-line chat example
├── run_server.py                 # Local FastAPI server
├── run_client.py                 # Client for local OpenAI-compatible API
├── demo_prefix_cache.py          # Small prefix-cache reuse demonstration
└── requirements.txt              # Python package requirements
```

## Code Architecture

The main implementation is under `core/`.

- `paged_kv_cache.py`: fixed-size physical KV blocks, per-request block tables, lazy allocation, block reclamation, reference counts, memory accounting, and copy-on-write support.
- `prefix_cache.py`: full-block prefix cache using hash chains. It supports longest-prefix lookup and tracks prefix hits, saved blocks, and cached full blocks.
- `contiguous_kv_cache.py`: simplified contiguous-allocation baselines used to compare against paged allocation. It implements Orca-style Oracle, Pow2, Max, and FasterTransformer-like reservation policies.
- `scheduler.py`: a simple waiting/running queue for continuous batching.
- `request.py` and `sampling_params.py`: request state, request output, and generation settings.
- `model_runner.py`: Hugging Face causal LM wrapper plus a local fallback model for small local checks.
- `engine.py` and `llm.py`: connect the scheduler, model runner, paged KV cache, and prefix cache into a single-process serving loop.
- `metrics.py`: raw JSONL logging, source labels, and summary helpers.

The API layer is under `api/`.

- `schemas.py`: Pydantic request/response objects for the chat API.
- `openai_server.py`: `/v1/chat/completions` endpoint and in-process completion helper.

The experiment layer is under `experiments/`.

- `run_e1_throughput.py` through `run_e5_block_size_ablation.py`: experiment entry points.
- `summarize_results.py`: turns raw JSONL logs into summary CSVs and writes `results/manifest.json`.
- `plot_figures.py`: generates the figures used to inspect the experiment trends.
- `workloads/prepare_workload_traces.py`: normalizes real ShareGPT/Alpaca-style input files into trace JSONL format.
- `baselines/`: Wrappers for the contiguous baseline policies.

## Hardware Environment

For GPU experiments, the project is run on CUDA-enabled NVIDIA GPUs with enough memory to hold the model weights, active KV cache, and batching overhead. Our experiments are designed under the following computing setup:

| Resource | GPUs | Useage Description |
|---|---:|---|
| Lab GPU server | 2 x RTX 6000 Ada Generation, 48 GB each | Small-scale serving runs, correctness-oriented GPU checks, baseline comparisons, and reduced-concurrency ablations |
| Lab GPU server | 2 x RTX PRO 6000 Blackwell, 96 GB each | Main single-GPU OPT-13B-class experiments, memory-efficiency runs, prefix-sharing runs, and larger batch/concurrency sweeps |
| Lab GPU server | 1 x NVIDIA A100 80 GB PCIe | Primary high-memory single-GPU reference device for OPT-13B-class serving and block-size ablation |
| University of Florida Computing Center | Multi-GPU high-memory nodes, used only when the lab server was insufficient | Larger multi-GPU configurations such as OPT-66B/OPT-175B-class experiments and runs requiring more aggregate GPU memory |

#### Experimental Deployment:

- E1 throughput under load, E2 memory efficiency, and E3 workload sensitivity can be run on one high-memory GPU for OPT-13B-class settings. The A100 80 GB or a 96 GB RTX PRO 6000 Blackwell GPU is the preferred choice; the 48 GB RTX 6000 Ada GPUs are suitable for smaller batch sizes, shorter traces, or reduced-concurrency checks.
- E4 prefix sharing and E5 block-size ablation are less dependent on a specific GPU model but still benefit from high memory when using longer ShareGPT traces or larger concurrent batches. We use the lab GPUs first and scale to UF compute-center GPUs only if the requested batch/model configuration does not fit locally.
- OPT-66B-class and OPT-175B-class experiments should be treated as multi-GPU runs. In practice, these are best placed on homogeneous high-memory GPU nodes from the UF compute center rather than on a mixed local GPU set.
- The mixed lab server provides substantial aggregate memory, but the GPUs should not be assumed to form one unified memory pool unless the run explicitly uses distributed/tensor-parallel execution.


## Package Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Pinned package versions are listed in `requirements.txt`:

```text
torch==2.6.0
transformers==4.51.3
numpy==1.26.4
pandas==2.2.3
matplotlib==3.8.4
PyYAML==6.0.2
tqdm==4.67.1
fastapi==0.115.12
uvicorn==0.40.0
openai==1.86.0
pydantic==2.11.7
pytest==8.3.5
```

## Basic Usage

Run a local generation example:

```bash
python run_local_chat.py --model fallback-local-model --prompt "Explain paged KV cache."
```

Run the local API server:

```bash
python run_server.py --model fallback-local-model --host 127.0.0.1 --port 8000
```

Call the API in process:

```bash
python run_client.py --model fallback-local-model --in-process
```

Demonstrate prefix-cache reuse:

```bash
python demo_prefix_cache.py --model fallback-local-model --requests 4
```


## Experiments and Result Locations

The experiment section follows the project report structure.

### E1: End-to-End Throughput Under Load

Goal: compare normalized latency and sustainable request rate under increasing offered load.

- Script: `experiments/run_e1_throughput.py`
- Raw logs: `results/raw/e1_throughput/*.jsonl`
- Summary: `results/summary/e1_latency_vs_request_rate.csv`
- Table data: `results/summary/table3_throughput_targets_and_measured.csv`
- Figure: `results/figures/fig3_latency_vs_request_rate.png`
- Extra figures: `results/figures/fig3_latency_vs_request_rate_sharegpt.png`, `results/figures/fig3_latency_vs_request_rate_alpaca.png`

### E2: Memory Efficiency and Fragmentation

Goal: compare useful KV capacity, internal waste, external waste, and concurrency across paged and contiguous allocation policies.

- Script: `experiments/run_e2_memory_efficiency.py`
- Raw logs: `results/raw/e2_memory_efficiency/*.jsonl`
- Summary: `results/summary/e2_fragmentation.csv`
- Table data: `results/summary/table4_memory_efficiency.csv`
- Detailed metric table: `results/summary/table4_memory_metrics.csv`
- Figure: `results/figures/e2_memory_waste_breakdown.png`

### E3: Workload Sensitivity

Goal: compare behavior on ShareGPT-style and Alpaca-style workload distributions.

- Script: `experiments/run_e3_workload_sensitivity.py`
- Raw logs: `results/raw/e3_workload_sensitivity/*.jsonl`
- Summary: `results/summary/e3_workload_sensitivity.csv`
- Workload statistics: `results/summary/workload_stats.csv`
- Figure: `results/figures/e3_sharegpt_vs_alpaca.png`

### E4: Prefix Sharing, Parallel Sampling, and Beam Search

Goal: measure the memory savings from shared prefixes under different decoding modes and widths.

- Script: `experiments/run_e4_prefix_sharing.py`
- Raw logs: `results/raw/e4_prefix_sharing/*.jsonl`
- Summary: `results/summary/e4_prefix_sharing.csv`
- Table data: `results/summary/table4_memory_efficiency.csv`
- Figure: `results/figures/fig_prefix_sharing_savings.png`

### E5: Block Size Ablation

Goal: show the trade-off between block size, internal fragmentation, and translation overhead.

- Script: `experiments/run_e5_block_size_ablation.py`
- Raw logs: `results/raw/e5_block_size/*.jsonl`
- Summary: `results/summary/e5_block_size.csv`
- Figure: `results/figures/fig4_block_size_ablation.png`
The Figure 4 artifact uses block sizes 16, 32, and 128, and records recovery cost, attention overhead, internal waste, and normalized latency in the summary CSV.

## Artifact Mapping

For quick checking, see:

- `results/artifact_to_paper_mapping.md`: maps report figures/tables to code, raw logs, summary CSVs, and figures.
- `results/summary/combined_summary.csv`: combined summary over all raw experiment records.
- `results/summary/table1_components.csv`: component-responsibility table.
- `results/summary/table2_eval_settings.csv`: evaluation setting table.


## Tests


```bash
pytest -q
```

The tests cover paged block allocation, fragmentation accounting, contiguous baselines, prefix sharing, copy-on-write behavior, scheduler state, metrics invariants, workload trace preparation, and API response structure.
