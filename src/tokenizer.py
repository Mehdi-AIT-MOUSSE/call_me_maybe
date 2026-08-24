from llm_sdk import Small_LLM_Model
from .data_loader import load_vocab, LoadError
import functools

@functools.lru_cache(maxsize=32)
def get_cached_vocab(vocab_path):
    """Loads vocab from disk once, then caches it in memory."""
    return load_vocab(vocab_path)

def my_encoder(llm, data):
    try:
        vocab_path = llm.get_path_to_vocab_file()
        vocab = get_cached_vocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    ids = []
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

def my_decoder(ids):
    try:
            vocab_path = llm.get_path_to_vocab_file()
            vocab = get_cached_vocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    revocab = {v:k for k,v in vocab.items()}

    result = ""
    for id in ids:
        result += revocab[id]

    result = result.replace("Ġ", " ").replace("Ċ", "\n")
    return result

llm = Small_LLM_Model()
vocab_path = llm.get_path_to_vocab_file()

vocab = load_vocab(vocab_path)

idss = llm.encode("Hello World!\n").tolist()[0]
print(idss)
ids = my_encoder(vocab, "Hello World!\n")

print(ids)

data = my_decoder(vocab, ids)

print(data)
