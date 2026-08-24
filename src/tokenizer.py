"""Custom tokenizer encode/decode using the vocabulary file (bonus)."""

from __future__ import annotations

import functools
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .data_loader import LoadError, load_vocab


@functools.lru_cache(maxsize=32)
def get_cached_vocab(vocab_path: str) -> dict[str, Any]:
    """Load and cache a vocabulary file from disk.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        Vocabulary mapping token strings to IDs.
    """
    return load_vocab(vocab_path)


def my_encoder(llm: Small_LLM_Model, data: str) -> list[int]:
    """Encode text into token IDs using longest-match vocab lookup.

    Args:
        llm: The language model instance (used to locate vocab file).
        data: Text string to encode.

    Returns:
        List of token IDs.
    """
    try:
        vocab_path = llm.get_path_to_vocab_file()
        vocab = get_cached_vocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    ids: list[int] = []
    i = 0
    size = len(data)
    while i < size:
        find = False
        for j in range(size, i, -1):
            posible_token = data[i:j].replace(" ", "Ġ").replace("\n", "Ċ")
            if posible_token in vocab:
                token = vocab[posible_token]
                ids.append(token)
                find = True
                i = j
                break

        if not find:
            i += 1

    return ids


def my_decoder(ids: list[int]) -> str:
    """Decode token IDs back into a text string.

    Args:
        ids: List of token IDs to decode.

    Returns:
        Decoded text string.
    """
    try:
        vocab_path = llm.get_path_to_vocab_file()
        vocab = get_cached_vocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    revocab = {v: k for k, v in vocab.items()}

    result = ""
    for token_id in ids:
        result += revocab[token_id]

    result = result.replace("Ġ", " ").replace("Ċ", "\n")
    return result


llm = Small_LLM_Model()
vocab_path = llm.get_path_to_vocab_file()

vocab = load_vocab(vocab_path)

idss = llm.encode("Hello World!\n").tolist()[0]
print(idss)
ids = my_encoder(vocab, "Hello World!\n")

print(ids)

data = my_decoder(ids)

print(data)
