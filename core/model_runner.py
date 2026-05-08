from __future__ import annotations

from typing import Any

import torch

from core.paged_kv_cache import concat_past_segments
from core.sampling_params import SamplingParams


class _FallbackTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        token_ids = self.encode(text)
        return type("Tokenized", (), {"input_ids": torch.tensor([token_ids], dtype=torch.long)})

    def encode(self, text: str) -> list[int]:
        if not text:
            return [self.eos_token_id]
        return [min(ord(char), 255) + 2 for char in text]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if token_id in {self.pad_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            if token_id >= 2:
                chars.append(chr(token_id - 2))
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        lines = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
        if add_generation_prompt:
            lines.append("Assistant:")
        text = "\n".join(lines)
        return self.encode(text) if tokenize else text


class _FallbackCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 258) -> None:
        super().__init__()
        answer_text = " paged kv cache uses fixed blocks and prefix sharing."
        self.answer_cycle = [min(ord(char), 255) + 2 for char in answer_text]
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = True,
    ) -> Any:
        batch_size, seq_len = input_ids.shape
        prev_seq_len = 0
        if past_key_values is not None:
            prev_seq_len = past_key_values[0][0].shape[-2]

        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-100.0,
            device=input_ids.device,
        )
        for offset in range(seq_len):
            next_token = self.answer_cycle[(prev_seq_len + offset) % len(self.answer_cycle)]
            logits[:, offset, next_token] = 100.0

        new_segment = input_ids.to(dtype=torch.float32).unsqueeze(1).unsqueeze(-1).detach()
        if past_key_values is None:
            full_past = ((new_segment, new_segment),)
        else:
            prev_keys, prev_values = past_key_values[0]
            full_segment = torch.cat([prev_keys, new_segment], dim=-2)
            full_values = torch.cat([prev_values, new_segment], dim=-2)
            full_past = ((full_segment, full_values),)
        return type("FallbackOutputs", (), {"logits": logits, "past_key_values": full_past})


class ModelRunner:
    """Thin wrapper around a causal LM with a no-network fallback model."""

    FALLBACK_MODEL_NAME = "fallback-local-model"

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = "fallback" if model_name == self.FALLBACK_MODEL_NAME else "transformers"
        self.load_error: Exception | None = None

        if self.backend == "fallback":
            self.tokenizer = _FallbackTokenizer()
            self.model = _FallbackCausalLM()
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                kwargs: dict[str, Any] = {}
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                self.load_error = exc
                raise RuntimeError(
                    f"Failed to load Hugging Face model '{model_name}'. "
                    f"Use '{self.FALLBACK_MODEL_NAME}' for no-network local checks."
                ) from exc

        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _to_tensor(self, token_ids: list[int]) -> torch.Tensor:
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def forward(
        self,
        token_ids: list[int],
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if not token_ids:
            raise ValueError("forward requires at least one token")
        outputs = self.model(
            input_ids=self._to_tensor(token_ids),
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits.detach(), outputs.past_key_values

    def prefill(self, token_ids: list[int], past_key_values: Any | None = None) -> tuple[torch.Tensor, Any]:
        return self.forward(token_ids, past_key_values=past_key_values)

    def decode(self, token_id: int, past_key_values: Any) -> tuple[torch.Tensor, Any]:
        return self.forward([token_id], past_key_values=past_key_values)

    def sample_next_token(self, logits: torch.Tensor, sampling_params: SamplingParams) -> int:
        last_logits = logits[:, -1, :]
        if sampling_params.greedy:
            return int(torch.argmax(last_logits, dim=-1).item())
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
        temperature = max(sampling_params.temperature, 1e-5)
        probs = torch.softmax(last_logits / temperature, dim=-1)
        if sampling_params.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > sampling_params.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_indices.gather(-1, sampled).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def slice_past(past_key_values: Any, start: int, end: int) -> Any:
        if past_key_values is None:
            raise ValueError("past_key_values is required for slicing")
        return tuple(
            (
                layer[0][..., start:end, :].detach().clone(),
                layer[1][..., start:end, :].detach().clone(),
            )
            for layer in past_key_values
        )

    @staticmethod
    def concat_past(segments: list[Any]) -> Any | None:
        return concat_past_segments(segments)
