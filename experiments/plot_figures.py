from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pagedkv_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
FIGURES = ROOT / "results" / "figures"

BASELINE_LABELS = {
    "pagedkv": "PagedKV",
    "orca_oracle": "Orca (Oracle)",
    "orca_pow2": "Orca (Pow2)",
    "orca_max": "Orca (Max)",
    "fastertransformer_like": "FasterTransformer-like",
}

BASELINE_COLORS = {
    "pagedkv": "#0072B2",
    "orca_oracle": "#009E73",
    "orca_pow2": "#CC79A7",
    "orca_max": "#E69F00",
    "fastertransformer_like": "#D55E00",
}

BASELINE_ORDER = [
    "pagedkv",
    "orca_oracle",
    "orca_pow2",
    "orca_max",
    "fastertransformer_like",
]


def read_rows(name: str) -> list[dict[str, str]]:
    path = SUMMARY / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_current(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180)
    plt.close()


def ordered_baselines(rows: list[dict[str, str]]) -> list[str]:
    present = {row["baseline"] for row in rows}
    ordered = [baseline for baseline in BASELINE_ORDER if baseline in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_e1() -> None:
    rows = read_rows("e1_latency_vs_request_rate.csv")
    if not rows:
        return
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })
    for dataset in sorted({row["dataset"] for row in rows}):
        plt.figure(figsize=(8.2, 5.0))
        subset = [row for row in rows if row["dataset"] == dataset]
        for baseline in ordered_baselines(subset):
            points = sorted(
                (float(row["request_rate"]), float(row["normalized_latency_p95"]))
                for row in subset
                if row["baseline"] == baseline
            )
            if points:
                xs, ys = zip(*points)
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=5.5,
                    linewidth=2.2,
                    color=BASELINE_COLORS.get(baseline),
                    label=BASELINE_LABELS.get(baseline, baseline),
                )
        plt.xlabel("Request rate (req/s)")
        plt.ylabel("P95 normalized latency")
        plt.title(f"E1 Throughput Under Load: {dataset.capitalize()}")
        plt.grid(True, axis="y", alpha=0.28, linewidth=0.8)
        plt.legend(frameon=False, ncol=2)
        save_current(f"fig3_latency_vs_request_rate_{dataset}.png")
    plt.figure(figsize=(8.2, 5.0))
    sharegpt_rows = [row for row in rows if row["dataset"] == "sharegpt"]
    for baseline in ordered_baselines(sharegpt_rows):
        points = sorted(
            (float(row["request_rate"]), float(row["normalized_latency_p95"]))
            for row in rows
            if row["dataset"] == "sharegpt" and row["baseline"] == baseline
        )
        if points:
            xs, ys = zip(*points)
            plt.plot(
                xs,
                ys,
                marker="o",
                markersize=5.5,
                linewidth=2.2,
                color=BASELINE_COLORS.get(baseline),
                label=BASELINE_LABELS.get(baseline, baseline),
            )
    plt.xlabel("Request rate (req/s)")
    plt.ylabel("P95 normalized latency")
    plt.title("E1 Throughput Under Load")
    plt.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    plt.legend(frameon=False, ncol=2)
    save_current("fig3_latency_vs_request_rate.png")


def plot_e2() -> None:
    rows = read_rows("e2_fragmentation.csv")
    if not rows:
        return
    sharegpt = [row for row in rows if row["dataset"] == "sharegpt"]
    labels = [row["baseline"] for row in sharegpt]
    internal = [float(row["internal_fragmentation_ratio"]) for row in sharegpt]
    external = [float(row["external_fragmentation_ratio"]) for row in sharegpt]
    plt.figure(figsize=(7, 4))
    x = range(len(labels))
    plt.bar(x, internal, label="internal")
    plt.bar(x, external, bottom=internal, label="external")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Fragmentation ratio")
    plt.title("E2 memory waste breakdown")
    plt.legend()
    save_current("e2_memory_waste_breakdown.png")


def plot_e3() -> None:
    rows = read_rows("e3_workload_sensitivity.csv")
    if not rows:
        return
    labels = []
    values = []
    for row in rows:
        if row["baseline"] == "pagedkv":
            labels.append(row["dataset"])
            values.append(float(row["avg_concurrent_requests"]))
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Avg concurrent requests")
    plt.title("E3 workload sensitivity")
    save_current("e3_sharegpt_vs_alpaca.png")


def plot_e4() -> None:
    rows = read_rows("e4_prefix_sharing.csv")
    if not rows:
        return
    rows = [row for row in rows if row["width"] == "6"]
    labels = [f"{row['dataset']}\n{row['decoding_mode']}" for row in rows]
    values = [float(row["memory_saving_ratio"]) for row in rows]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("Memory saving ratio")
    plt.title("E4 prefix sharing savings")
    save_current("fig_prefix_sharing_savings.png")


def plot_e5() -> None:
    rows = read_rows("e5_block_size.csv")
    if not rows:
        return
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })
    block_sizes = sorted({int(row["block_size"]) for row in rows})
    x_positions = list(range(len(block_sizes)))
    x_lookup = {block_size: index for index, block_size in enumerate(block_sizes)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    cost_by_block = {}
    for block_size in block_sizes:
        candidates = [row for row in rows if int(row["block_size"]) == block_size]
        first = candidates[0]
        cost_by_block[block_size] = (
            float(first.get("recompute_recovery_cost_ms", 0) or 0),
            float(first.get("block_transfer_cost_ms", 0) or 0),
        )

    bar_width = 0.28
    recompute = [cost_by_block[block_size][0] for block_size in block_sizes]
    transfer = [cost_by_block[block_size][1] for block_size in block_sizes]
    axes[0].bar(
        [x - bar_width / 2 for x in x_positions],
        recompute,
        width=bar_width,
        color="#D55E00",
        label="Recompute",
    )
    axes[0].bar(
        [x + bar_width / 2 for x in x_positions],
        transfer,
        width=bar_width,
        color="#0072B2",
        label="Block transfer",
    )
    axes[0].set_xticks(x_positions, [str(block_size) for block_size in block_sizes])
    axes[0].set_xlabel("Block size (tokens)")
    axes[0].set_ylabel("Recovery cost (ms)")
    axes[0].set_title("(a) Recovery Overhead")
    axes[0].grid(True, axis="y", alpha=0.28, linewidth=0.8)
    axes[0].legend(frameon=False)

    for dataset in sorted({row["dataset"] for row in rows}):
        latency_points = sorted(
            (int(row["block_size"]), float(row["normalized_latency_p95"]))
            for row in rows
            if row["dataset"] == dataset
        )
        xs = [x_lookup[block_size] for block_size, _ in latency_points]
        ys = [value for _, value in latency_points]
        axes[1].plot(
            xs,
            ys,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label=dataset.capitalize(),
        )
    axes[1].set_xticks(x_positions, [str(block_size) for block_size in block_sizes])
    axes[1].set_xlabel("Block size (tokens)")
    axes[1].set_ylabel("P95 normalized latency")
    axes[1].set_title("(b) Serving Efficiency")
    axes[1].grid(True, axis="y", alpha=0.28, linewidth=0.8)
    axes[1].legend(frameon=False)
    save_current("fig4_block_size_ablation.png")


def main() -> None:
    plot_e1()
    plot_e2()
    plot_e3()
    plot_e4()
    plot_e5()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
