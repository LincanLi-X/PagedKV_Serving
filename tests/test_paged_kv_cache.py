from __future__ import annotations

import math

from core.paged_kv_cache import PagedKVCache


def test_block_count_for_boundary_lengths() -> None:
    block_size = 4
    for length in [0, 1, block_size, block_size + 1, 2 * block_size + 3]:
        cache = PagedKVCache(block_size=block_size, num_blocks=16)
        table = cache.attach_request(f"r{length}")
        cache.append_tokens(f"r{length}", table, list(range(length)))

        assert len(table) == math.ceil(length / block_size)
        assert cache.memory_stats()["num_used_blocks"] == math.ceil(length / block_size)


def test_internal_waste_is_tail_space() -> None:
    cache = PagedKVCache(block_size=4, num_blocks=8)
    table = cache.attach_request("r0")
    cache.append_tokens("r0", table, [1, 2, 3, 4, 5])

    stats = cache.memory_stats()

    assert stats["useful_tokens"] == 5
    assert stats["reserved_token_slots"] == 8
    assert stats["internal_waste_tokens"] == 3
    assert stats["external_waste_tokens"] == 0


def test_paged_kv_never_reports_external_fragmentation_after_interleaving() -> None:
    cache = PagedKVCache(block_size=4, num_blocks=12)
    first = cache.attach_request("first")
    middle = cache.attach_request("middle")
    last = cache.attach_request("last")
    cache.append_tokens("first", first, list(range(8)))
    cache.append_tokens("middle", middle, list(range(4)))
    cache.append_tokens("last", last, list(range(8)))

    cache.free_request("middle")
    extra = cache.attach_request("extra")
    cache.append_tokens("extra", extra, list(range(4)))

    assert cache.memory_stats()["external_waste_tokens"] == 0
    assert cache.memory_stats()["external_fragmentation_ratio"] == 0.0


def test_internal_waste_is_bounded_by_one_tail_block_per_active_sequence() -> None:
    block_size = 4
    cache = PagedKVCache(block_size=block_size, num_blocks=16)
    for request_id, length in [("a", 1), ("b", 5), ("c", 8)]:
        table = cache.attach_request(request_id)
        cache.append_tokens(request_id, table, list(range(length)))

    active_sequences = len(cache.request_blocks)
    stats = cache.memory_stats()

    assert stats["internal_waste_tokens"] <= active_sequences * (block_size - 1)


def test_free_request_returns_blocks_to_pool() -> None:
    cache = PagedKVCache(block_size=4, num_blocks=8)
    table = cache.attach_request("r0")
    cache.append_tokens("r0", table, list(range(9)))

    assert cache.num_free_blocks == 5
    cache.free_request("r0")

    assert cache.num_free_blocks == 8
    assert cache.memory_stats()["num_used_blocks"] == 0


def test_non_contiguous_physical_blocks_keep_logical_order() -> None:
    cache = PagedKVCache(block_size=2, num_blocks=6)
    t0 = cache.attach_request("r0")
    t1 = cache.attach_request("r1")
    cache.append_tokens("r0", t0, [1, 2])
    cache.append_tokens("r1", t1, [9, 10])
    cache.append_tokens("r0", t0, [3, 4])

    assert cache.flatten_block_tokens(t0) == [1, 2, 3, 4]
    assert t0 == [0, 2]


def test_appending_after_shared_full_block_allocates_new_private_tail() -> None:
    cache = PagedKVCache(block_size=4, num_blocks=8)
    shared = cache.attach_request("shared")
    cache.append_tokens("shared", shared, [1, 2, 3, 4])
    cache.acquire_cached_blocks("r1", shared)
    r1 = cache.request_blocks["r1"]
    cache.acquire_cached_blocks("r2", shared)
    r2 = cache.request_blocks["r2"]

    cache.append_tokens("r1", r1, [5])

    assert cache.flatten_block_tokens(r1) == [1, 2, 3, 4, 5]
    assert cache.flatten_block_tokens(r2) == [1, 2, 3, 4]
    assert cache.memory_stats()["cow_count"] == 0
