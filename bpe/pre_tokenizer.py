import multiprocessing
import regex as re
from collections import Counter
from functools import partial
from typing import Dict, List, Tuple

# Assuming these are in other files as in your example
from cs336_basics.pretokenization_example import find_chunk_boundaries
from .regex_tokenizer import RegexTokenizer


# --- Main Orchestration Function (Updated) ---
def run_pre_tokenization(
    input_path: str, tokenizer: RegexTokenizer, special_tokens: list[str]
) -> Dict[Tuple[int, ...], int]:
    """
    Orchestrates the parallel pre-tokenization of a large text file.

    This function chunks a file, processes the chunks in parallel to get
    pre-token frequencies, and aggregates them into a single master frequency map.

    Args:
        input_path: The path to the text corpus.
        tokenizer: An instance of your RegexTokenizer.
        special_tokens: A list of special token strings.

    Returns:
        A dictionary mapping each unique pre-token (as a tuple of byte IDs)
        to its total frequency count across the entire corpus.
    """
    # --- Step 1: Find chunk boundaries ---
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=multiprocessing.cpu_count(),
            split_special_token=b"<|endoftext|>",
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_str = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk_str)

    print(f"Divided the file into {len(chunks)} chunks for parallel processing.")

    # --- Step 2: Set up and run the multiprocessing pool ---
    worker_func = partial(worker, tokenizer=tokenizer, special_tokens=special_tokens)

    with multiprocessing.Pool() as pool:
        list_of_counts = pool.map(worker_func, chunks)

    # --- Step 3: Aggregate the results ---
    # `list_of_counts` is now a list of Counter objects.
    # We aggregate them into a single Counter for the final result.
    total_counts = Counter()
    for counts in list_of_counts:
        total_counts.update(counts)

    return dict(total_counts)


# --- The Worker Function (Updated) ---
def worker(chunk: str, tokenizer: RegexTokenizer, special_tokens: list[str]) -> Counter:
    """
    This function runs in a separate process. It processes one text chunk,
    splits it by special tokens, and returns the frequency of each pre-token.
    """
    special_pattern = f"({ '|'.join(re.escape(token) for token in special_tokens) })"
    # ['', '<start>', ' lucas is great ', '<|endoftext|>', '']
    text_parts = re.split(special_pattern, chunk)

    # Use a Counter for efficient counting
    chunk_to_cnt = Counter()
    for part in text_parts:
        if not part:
            continue
        # Check if the part is a special token
        if part not in special_tokens:
            # It's normal text, apply the regex
            pre_tokens = re.findall(tokenizer.compiled_pattern, part)
            for p_token in pre_tokens:
                # The key must be a tuple to be hashable
                pre_token_tuple = tuple(p_token.encode("utf-8"))
                chunk_to_cnt[pre_token_tuple] += 1
    return chunk_to_cnt
