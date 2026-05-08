from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.llm import MiniLLM
from core.model_runner import ModelRunner
from core.sampling_params import SamplingParams


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate full-block prefix cache reuse.")
    parser.add_argument("--model", default=ModelRunner.FALLBACK_MODEL_NAME)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "raw" / "demo_prefix_cache"))
    args = parser.parse_args()

    shared_prefix = (
        "System prompt: You are evaluating paged KV cache allocation and prefix reuse. "
    ) * 4
    prompts = [
        shared_prefix + f"Request {index}: summarize one cache-management benefit."
        for index in range(args.requests)
    ]
    llm = MiniLLM(
        model=args.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_batch_size=args.requests,
        enable_prefix_cache=True,
    )
    outputs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens))
    stats = llm.stats()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"demo_prefix_cache_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for prompt, output in zip(prompts, outputs):
            record = {
                "experiment": "demo_prefix_cache",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": "pagedkv",
                "baseline": "paged_prefix_cache",
                "model": args.model,
                "block_size": args.block_size,
                "request_id": output.request_id,
                "prompt_len": len(output.prompt_token_ids),
                "output_len": len(output.token_ids),
                "cached_prefix_tokens": output.cached_prefix_tokens,
                "text": output.text,
                "source": "measured_by_our_code",
                **stats,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"raw_log: {path}")


if __name__ == "__main__":
    main()
