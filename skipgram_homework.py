import argparse
import os
import re
import urllib.request
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
UNK = "<unk>"


def download_dataset(path="data/input.txt"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading dataset to {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def tokenize(text):
    return re.findall(r"<\|endoftext\|>|[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text.lower())


def load_tokens(path="data/input.txt", max_tokens=200_000):
    download_dataset(path)
    with open(path, "r", encoding="utf-8") as f:
        tokens = tokenize(f.read())
    return tokens[:max_tokens]


def build_vocab(tokens, vocab_size=5_000):
    counts = Counter(tokens)
    words = [word for word, _ in counts.most_common(vocab_size - 1)]
    itos = [UNK] + words
    stoi = {word: i for i, word in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(token, unk_id) for token in tokens]


class SkipGramDataset(Dataset):
    def __init__(self, token_ids, window_size):
        self.pairs = []

        # TODO: fill self.pairs with (center_word_id, context_word_id).
        # For each token, all words up to window_size positions left/right are context words.
        # Do not include the center word itself.
        raise NotImplementedError("build the skip-gram training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        center_id, context_id = self.pairs[index]
        return torch.tensor(center_id, dtype=torch.long), torch.tensor(context_id, dtype=torch.long)


class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # TODO: create these exact layers so the leaderboard can load your model:
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.output = nn.Linear(embedding_dim, vocab_size)
        raise NotImplementedError("build the skip-gram model")

    def forward(self, center_ids):
        # TODO: center_ids -> embedding -> logits over the vocabulary
        raise NotImplementedError("write the forward pass")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(model, loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0

        for center_ids, context_ids in loader:
            center_ids = center_ids.to(device)
            context_ids = context_ids.to(device)

            logits = model(center_ids)
            loss = loss_fn(logits, context_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = context_ids.numel()
            total_loss += loss.item() * batch_size
            total_items += batch_size

        print(f"epoch {epoch}/{epochs} loss {total_loss / total_items:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Skip-gram Word2Vec homework starter.")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--vocab-size", type=int, default=5_000)
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", default="submission.pt")
    args = parser.parse_args()

    tokens = load_tokens(args.data_path, args.max_tokens)
    split = int(len(tokens) * (1.0 - args.valid_fraction))
    train_tokens = tokens[:split]

    stoi, itos = build_vocab(train_tokens, args.vocab_size)
    train_ids = encode(train_tokens, stoi)

    dataset = SkipGramDataset(train_ids, args.window_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SkipGramWord2Vec(len(itos), args.embedding_dim)

    device = pick_device()
    print(f"tokens {len(train_tokens)} | vocab {len(itos)} | pairs {len(dataset)} | device {device}")
    train(model, loader, args.epochs, args.lr, device)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "config": {
                "model_type": "skipgram",
                "embedding_dim": args.embedding_dim,
                "vocab_size": len(itos),
                "window_size": args.window_size,
            },
        },
        args.save_path,
    )
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
