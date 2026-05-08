from __future__ import annotations

import argparse
import json

from api.openai_server import complete_chat
from api.schemas import ChatCompletionRequest, ChatMessage
from core.model_runner import ModelRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the local PagedKV server or in-process API.")
    parser.add_argument("--model", default=ModelRunner.FALLBACK_MODEL_NAME)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--prompt", default="What is paged KV cache?")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()

    if args.in_process:
        response = complete_chat(
            ChatCompletionRequest(
                model=args.model,
                messages=[ChatMessage(role="user", content=args.prompt)],
                max_tokens=args.max_tokens,
            )
        )
        print(json.dumps(response.model_dump(), indent=2, sort_keys=True))
        return

    from openai import OpenAI

    client = OpenAI(api_key="not-needed-for-local-server", base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
