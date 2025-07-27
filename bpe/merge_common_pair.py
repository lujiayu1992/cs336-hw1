from typing import Optional, Dict, Tuple, Set, List
from collections import Counter


def get_stats(ids: Tuple[int, ...]) -> Counter[Tuple[int, int]]:
    """
    Calculates the frequency of adjacent pairs within a single word.
    """
    return Counter(zip(ids[:-1], ids[1:]))


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    In a list of integers (ids), replace all consecutive occurrences
    of a specific pair with a new integer token (idx).

    Args:
        ids: A list of integer IDs representing a word.
        pair: The tuple of two integers to be merged.
        idx: The new integer ID to replace the pair with.

    Returns:
        A new list of integers with the pair merged.
    """
    merged = []
    i = 0
    while i < len(ids):
        # Check if we are at the end of the list to avoid index out of bounds
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(idx)
            i += 2  # Skip over the two tokens that were just merged
        else:
            merged.append(ids[i])
            i += 1
    return merged


def update_cache(
    subword: Tuple[int, ...],
    subword_to_pair_cnt_cache: Dict[Tuple[int, ...], Counter[Tuple[int, int]]],
    subword_freq: int,
    global_pair_cnt: Counter[Tuple[int, int]],
) -> None:
    """
    Updates the global pair count and the subword-to-pair cache.

    This function calculates the internal pair counts for a given subword,
    updates a cache that stores these counts, and updates the global
    pair frequency map, weighted by the subword's frequency.

    Args:
        subword: The word/subword (as a tuple of int IDs) to process.
        subword_freq: The frequency of this subword in the corpus.
        global_pair_count: The master Counter for all pair frequencies.
        subword_to_pair_cnt_cache: The cache mapping subwords to their internal pair counts.
    """
    local_pair_counts = get_stats(subword)
    subword_to_pair_cnt_cache[subword] = local_pair_counts
    for pair, cnt in local_pair_counts.items():
        global_pair_cnt[pair] = global_pair_cnt.get(pair, 0) + cnt * subword_freq


def remove_from_cache(
    subword: Tuple[int, ...],
    subword_to_pair_cnt_cache: Dict[Tuple[int, ...], Counter[Tuple[int, int]]],
    subword_freq: int,
    global_pair_cnt: Counter[Tuple[int, int]],
) -> None:
    """
    Removes a subword's contribution from the global pair count and deletes
    the subword from the cache before it gets updated with a new merge.
    """
    for pair, cnt in subword_to_pair_cnt_cache[subword].items():
        global_pair_cnt[pair] -= subword_freq*cnt
        if global_pair_cnt[pair]<=0:
            del global_pair_cnt[pair]
    del subword_to_pair_cnt_cache[subword]


def generate_global_pair_cnt(
    subword_to_pair_cnt_cache: Dict[Tuple[int, ...], Counter[Tuple[int, int]]],
    global_subword_count: Dict[Tuple[int, ...], int],
) -> Counter[Tuple[int, int]]:
    global_pair_count = Counter()
    for subword, pair_cnts in subword_to_pair_cnt_cache.items():
        word_freq = global_subword_count[subword]
        for pair, cnt in pair_cnts.items():
            global_pair_count[pair] = global_pair_count.get(pair, 0) + cnt * word_freq
    return global_pair_count
