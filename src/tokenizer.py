"""Custom tokenizer encode/decode using the vocabulary file (bonus)."""

import functools
from typing import cast
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .data_loader import LoadError, load_vocab
# from typing import int


@functools.lru_cache(maxsize=32)
def get_cached_vocab(vocab_path: str) -> dict[str, int]:
    """Load and cache a vocabulary file from disk."""
    return cast(dict[str, int], load_vocab(vocab_path))


@functools.lru_cache(maxsize=32)
def get_cached_revocab(vocab_path: str) -> dict[int, str]:
    """Create and cache the reverse vocabulary (IDs to tokens)."""
    vocab = get_cached_vocab(vocab_path)
    return {v: k for k, v in vocab.items()}


def my_encoder(llm: Small_LLM_Model, data: str) -> list[int]:
    """Encode text into token IDs using longest-match vocab lookup."""
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


def my_decoder(llm: Small_LLM_Model, ids: list[int]) -> str:
    """Decode token IDs back into a text string."""
    try:
        vocab_path = llm.get_path_to_vocab_file()
        revocab = get_cached_revocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    result = ""
    for token_id in ids:
        result += revocab.get(token_id, "")

    result = result.replace("Ġ", " ").replace("Ċ", "\n")
    return result


if __name__ == "__main__":
    llm = Small_LLM_Model()

    idss = llm.encode("Hello World!").tolist()[0]
    print("Official IDs:", idss)

    ids = my_encoder(llm, "Hello World!")
    print("Custom IDs:  ", ids)

    data = my_decoder(llm, ids)
    print("Decoded text:", repr(data))
