from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    stop: Sequence[str] | None = field(default_factory=list)
    num_return_sequences: int = 1
    beam_width: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.stop is None:
            self.stop = []

    @property
    def greedy(self) -> bool:
        return self.temperature == 0 or self.top_p <= 0


DEFAULT_SAMPLING_PARAMS = SamplingParams()
