from __future__ import annotations

from core.paged_kv_cache import PagedKVCache
from core.prefix_cache import PrefixCache


def test_prefix_cache_only_registers_full_blocks() -> None:
    cache = PrefixCache(block_size=4)
    try:
        cache.register_full_block(0, [1, 2, 3], None)
    except ValueError:
        pass
    else:
        raise AssertionError("partial block registration should fail")


def test_shared_prefix_lookup_hits_longest_hash_chain() -> None:
    prefix = PrefixCache(block_size=2)
    h0 = prefix.register_full_block(10, [1, 2], None)
    h1 = prefix.register_full_block(11, [3, 4], h0)

    match = prefix.lookup_longest_prefix([1, 2, 3, 4, 99])

    assert match.block_ids == [10, 11]
    assert match.block_hashes == [h0, h1]
    assert match.matched_tokens == 4
    assert prefix.stats()["hits"] == 1


def test_cached_block_acquire_increments_ref_count() -> None:
    kv = PagedKVCache(block_size=2, num_blocks=4)
    table = kv.attach_request("src")
    kv.append_tokens("src", table, [1, 2])

    kv.acquire_cached_blocks("dst", table)

    assert kv.blocks[table[0]].ref_count == 2
    assert kv.request_blocks["dst"] == table


def test_copy_on_write_isolates_divergent_partial_tail() -> None:
    kv = PagedKVCache(block_size=4, num_blocks=4)
    r1 = kv.attach_request("r1")
    kv.append_tokens("r1", r1, [1, 2])
    shared_block_id = r1[0]
    kv.blocks[shared_block_id].ref_count += 1
    r2 = kv.attach_request("r2")
    r2.append(shared_block_id)

    kv.append_tokens("r1", r1, [3])

    assert kv.flatten_block_tokens(r1) == [1, 2, 3]
    assert kv.flatten_block_tokens(r2) == [1, 2]
    assert r1[0] != r2[0]
    assert kv.memory_stats()["cow_count"] == 1


def test_free_one_shared_request_keeps_block_for_other_request() -> None:
    kv = PagedKVCache(block_size=2, num_blocks=4)
    src = kv.attach_request("src")
    kv.append_tokens("src", src, [1, 2])
    kv.acquire_cached_blocks("dst", src)

    kv.free_request("dst")

    assert kv.blocks[src[0]].ref_count == 1
    assert kv.num_free_blocks == 3
