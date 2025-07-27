import pickle
import regex as re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Iterator
from collections import Counter
from .merge_common_pair import (
    get_stats,
    merge,
    update_cache,
    remove_from_cache,
    generate_global_pair_cnt,
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self._add_special_tokens(special_tokens)
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _add_special_tokens(self, tokens: list[str]):
        """
        Adds special tokens to the vocabulary in-place and rebuilds the inverse vocabulary.
        """
        if not tokens:
            return
        for token in tokens:
            if token.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = token.encode("utf-8")

    # OPTION 2: The from_files class method
    # This is used to load a tokenizer that you previously trained and saved.
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Constructs a Tokenizer from a saved vocabulary and merges file.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[bytes]:
        pre_tokens_as_bytes = []
        text_parts = [text]
        if self.special_tokens:
            special_pattern = (
                f"({ '|'.join(re.escape(token) for token in self.special_tokens) })"
            )
            # ['', '<start>', ' lucas is great ', '<|endoftext|>', '']
            text_parts = re.split(special_pattern, text)
        for part in text_parts:
            if not part:
                continue
            if part in self.special_tokens:
                pre_tokens_as_bytes.append(part.encode("utf-8"))
            else:
                pre_tokens = re.findall(self.compiled_pattern, part)
                for pre_token in pre_tokens:
                    pre_tokens_as_bytes.append(pre_token.encode("utf-8"))
        return pre_tokens_as_bytes

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        """
        # Your encoding logic goes here
        # ['', '<start>', ' lucas is great ', '<|endoftext|>', '']
        
        pre_tokens_as_bytes = self.pre_tokenize(text)
        token_ids = []
        
        for pre_token_as_bytes in pre_tokens_as_bytes:
            if pre_token_as_bytes in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[pre_token_as_bytes])
            else:
                token_ids = token_ids + self._encode_chunk(pre_token_as_bytes)
        return token_ids

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        # return the token ids
        token_bytes = [bytes([b]) for b in text_bytes]
        while len(token_bytes) >= 2:
            stats = Counter(zip(token_bytes[:-1], token_bytes[1:]))
            pair_to_merge = None
            for p in self.merges:
                if p in stats:
                    pair_to_merge = p
                    break  # Found the highest-priority merge, so we can stop searching.
            if pair_to_merge is None:
                break
            token_bytes = self._merge(token_bytes, pair_to_merge)
        return [self.inverse_vocab[token_byte] for token_byte in token_bytes]

    def _merge(self, text_bytes: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
        merged = []
        i = 0
        while i < len(text_bytes):
            # Check if we are at the end of the list to avoid index out of bounds
            if i < len(text_bytes) - 1 and (text_bytes[i], text_bytes[i + 1]) == pair:
                merged.append(pair[0] + pair[1])
                i += 2  # Skip over the two tokens that were just merged
            else:
                merged.append(text_bytes[i])
                i += 1
        return merged

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            encoded_ids = self.encode(text_chunk)
            for token_id in encoded_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.
        """
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        return text_bytes.decode("utf-8", errors="replace")
