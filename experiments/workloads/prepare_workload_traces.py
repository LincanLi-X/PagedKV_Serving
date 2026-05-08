from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _percentile(values: list[int | float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input data file does not exist: {path}")
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ["data", "examples", "conversations", "records"]:
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"cannot find a list of examples in {path}")
    raise ValueError("input must be .jsonl or .json")


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict):
                pieces.append(str(item.get("content") or item.get("value") or item.get("text") or ""))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        return str(value.get("content") or value.get("value") or value.get("text") or "")
    return str(value)


def _extract_prompt(record: dict[str, Any]) -> str:
    for key in ["prompt_text", "prompt", "instruction", "input", "text"]:
        if record.get(key):
            return _message_text(record[key])
    for key in ["messages", "conversations"]:
        if record.get(key):
            return _message_text(record[key])
    raise ValueError(f"cannot extract prompt text from record keys: {sorted(record)}")


def _extract_output(record: dict[str, Any]) -> str:
    for key in ["output_text", "output", "response", "answer", "completion"]:
        if record.get(key):
            return _message_text(record[key])
    return ""


def _token_len(text: str) -> int:
    return max(1, len(text.split()))


def normalize_records(
    records: list[dict[str, Any]],
    dataset: str,
    arrival_rate: float,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    current_arrival = 0.0
    interval = 1.0 / arrival_rate if arrival_rate > 0 else 0.0
    for index, record in enumerate(records):
        prompt_text = _extract_prompt(record)
        output_text = _extract_output(record)
        arrival_time = float(record.get("arrival_time", current_arrival))
        prompt_len = int(record.get("prompt_len", _token_len(prompt_text)))
        output_len = int(record.get("output_len", _token_len(output_text) if output_text else 1))
        normalized.append(
            {
                "request_id": str(record.get("request_id", f"{dataset.lower()}-{index:06d}")),
                "arrival_time": round(arrival_time, 6),
                "prompt_len": prompt_len,
                "output_len": output_len,
                "prompt_text": prompt_text,
                "dataset": dataset.lower(),
                "data_source": str(record.get("data_source", dataset)),
            }
        )
        current_arrival = arrival_time + interval
    return normalized


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records: list[dict[str, Any]], dataset: str, arrival_rate: float) -> dict[str, int | float | str]:
    prompt_lengths = [int(record["prompt_len"]) for record in records]
    output_lengths = [int(record["output_len"]) for record in records]
    return {
        "dataset": dataset.lower(),
        "num_requests": len(records),
        "avg_prompt_len": round(mean(prompt_lengths), 3) if prompt_lengths else 0,
        "p50_prompt_len": _percentile(prompt_lengths, 50),
        "p95_prompt_len": _percentile(prompt_lengths, 95),
        "avg_output_len": round(mean(output_lengths), 3) if output_lengths else 0,
        "p50_output_len": _percentile(output_lengths, 50),
        "p95_output_len": _percentile(output_lengths, 95),
        "arrival_rate": arrival_rate,
        "source": "measured_by_our_code",
    }


def write_summary(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "num_requests",
        "avg_prompt_len",
        "p50_prompt_len",
        "p95_prompt_len",
        "avg_output_len",
        "p50_output_len",
        "p95_output_len",
        "arrival_rate",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized workload traces from real dataset files.")
    parser.add_argument("--sharegpt-input", type=Path, help="Real ShareGPT-format JSON/JSONL input.")
    parser.add_argument("--alpaca-input", type=Path, help="Real Alpaca-format JSON/JSONL input.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--summary",
        default=str(Path(__file__).resolve().parents[2] / "results" / "summary" / "workload_stats.csv"),
    )
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    args = parser.parse_args()

    inputs = {
        "sharegpt": args.sharegpt_input,
        "alpaca": args.alpaca_input,
    }
    if not any(inputs.values()):
        raise SystemExit(
            "Provide at least one real dataset file with --sharegpt-input or --alpaca-input. "
            "This project does not generate workload traces without real input data."
        )

    output_dir = Path(args.output_dir)
    rows = []
    for dataset, input_path in inputs.items():
        if input_path is None:
            continue
        records = normalize_records(_load_records(input_path), dataset, args.arrival_rate)
        write_jsonl(output_dir / f"{dataset}_trace.jsonl", records)
        rows.append(summarize(records, dataset, args.arrival_rate))
    write_summary(Path(args.summary), rows)
    print(f"wrote normalized traces to {output_dir}")
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
