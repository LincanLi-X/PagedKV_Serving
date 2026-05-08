from __future__ import annotations

from typing import Any


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    lines.append("Assistant:")
    return "\n".join(lines)
