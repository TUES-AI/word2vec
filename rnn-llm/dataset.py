import json
import os
import re
import urllib.request
from collections import Counter

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
UNK = "<unk>"


def download_dataset(path="data/input.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading TinyStories to {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def read_text(path="data/input.txt"):
    download_dataset(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text):
    return re.findall(r"<\|endoftext\|>|[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text.lower())


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    words = [word for word, count in counts.items() if count >= min_freq]
    words = [UNK] + sorted(words)
    stoi = {word: i for i, word in enumerate(words)}
    itos = words
    return stoi, itos


def encode(tokens, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(token, unk_id) for token in tokens]


def decode(ids, itos):
    words = [itos[i] if 0 <= i < len(itos) else UNK for i in ids]
    text = " ".join(words)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    return text


def save_vocab(stoi, itos, path="data/vocab.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)


def load_vocab(path="data/vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["stoi"], vocab["itos"]


def load_tokens(path="data/input.txt", max_tokens=None):
    tokens = tokenize(read_text(path))
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens
