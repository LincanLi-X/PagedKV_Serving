from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True)
class PrefixMatch:
    block_ids: list[int]
    block_hashes: list[str]
    matched_tokens: int

    @property
    def matched_blocks(self) -> int:
        return len(self.block_ids)


class PrefixCache:
    """Hash-chain prefix cache that indexes only complete KV blocks."""

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.hash_to_block_ids: dict[str, list[int]] = defaultdict(list)
        self.block_id_to_hash: dict[int, str] = {}
        self.lookup_count = 0
        self.hit_count = 0
        self.hit_tokens = 0
        self.saved_block_count = 0

    def _compute_block_hash(
        self,
        parent_hash: str | None,
        block_tokens: list[int],
    ) -> str:
        if len(block_tokens) != self.block_size:
            raise ValueError("prefix cache hashes only full blocks")
        payload = f"{parent_hash or 'root'}|{','.join(map(str, block_tokens))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_hash_chain(self, token_ids: list[int]) -> list[str]:
        parent_hash: str | None = None
        hashes: list[str] = []
        full_tokens = (len(token_ids) // self.block_size) * self.block_size
        for start in range(0, full_tokens, self.block_size):
            block_hash = self._compute_block_hash(
                parent_hash,
                token_ids[start : start + self.block_size],
            )
            hashes.append(block_hash)
            parent_hash = block_hash
        return hashes

    def register_full_block(
        self,
        block_id: int,
        block_tokens: list[int],
        parent_hash: str | None,
    ) -> str:
        block_hash = self._compute_block_hash(parent_hash, block_tokens)
        if block_id not in self.hash_to_block_ids[block_hash]:
            self.hash_to_block_ids[block_hash].append(block_id)
        self.block_id_to_hash[block_id] = block_hash
        return block_hash

    def evict_block(self, block_id: int, block_hash: str) -> None:
        block_ids = self.hash_to_block_ids.get(block_hash, [])
        if block_id in block_ids:
            block_ids.remove(block_id)
        if not block_ids and block_hash in self.hash_to_block_ids:
            del self.hash_to_block_ids[block_hash]
        self.block_id_to_hash.pop(block_id, None)

    def lookup_longest_prefix(self, token_ids: list[int]) -> PrefixMatch:
        self.lookup_count += 1
        full_block_count = len(token_ids) // self.block_size
        # Keep at least one block uncached for logits materialization when the
        # prompt length is exactly block-aligned.
        if full_block_count > 0 and len(token_ids) % self.block_size == 0:
            full_block_count -= 1
        if full_block_count <= 0:
            return PrefixMatch(block_ids=[], block_hashes=[], matched_tokens=0)

        block_ids: list[int] = []
        block_hashes: list[str] = []
        parent_hash: str | None = None
        for block_index in range(full_block_count):
            start = block_index * self.block_size
            block_tokens = token_ids[start : start + self.block_size]
            block_hash = self._compute_block_hash(parent_hash, block_tokens)
            candidates = self.hash_to_block_ids.get(block_hash)
            if not candidates:
                break
            block_ids.append(candidates[-1])
            block_hashes.append(block_hash)
            parent_hash = block_hash

        matched_tokens = len(block_ids) * self.block_size
        if matched_tokens:
            self.hit_count += 1
            self.hit_tokens += matched_tokens
            self.saved_block_count += len(block_ids)
        return PrefixMatch(
            block_ids=block_ids,
            block_hashes=block_hashes,
            matched_tokens=matched_tokens,
        )

    def stats(self) -> dict[str, int | float]:
        cached_blocks = len(self.block_id_to_hash)
        return {
            "lookups": self.lookup_count,
            "hits": self.hit_count,
            "hit_tokens": self.hit_tokens,
            "cached_full_blocks": cached_blocks,
            "saved_block_count": self.saved_block_count,
            "memory_saving_ratio": (
                self.saved_block_count / (self.saved_block_count + cached_blocks)
                if self.saved_block_count + cached_blocks
                else 0.0
            ),
        }
