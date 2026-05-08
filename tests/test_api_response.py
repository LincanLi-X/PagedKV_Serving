from __future__ import annotations

import json

from api.openai_server import complete_chat
from api.schemas import ChatCompletionRequest, ChatMessage
from core.model_runner import ModelRunner


def test_openai_compatible_response_shape() -> None:
    response = complete_chat(
        ChatCompletionRequest(
            model=ModelRunner.FALLBACK_MODEL_NAME,
            messages=[ChatMessage(role="user", content="Explain paging.")],
            max_tokens=4,
        ),
        block_size=4,
        num_blocks=64,
    )

    data = json.loads(response.model_dump_json())
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["usage"]["completion_tokens"] == 4
    assert "num_total_blocks" in data["metrics"]
