import argparse
import re
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

UNK = "<unk>"


def tokenize(text):
    return re.findall(r"<\|endoftext\|>|[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text.lower())


def build_vocab(tokens, vocab_size):
    counts = Counter(tokens)
    words = [word for word, _ in counts.most_common(vocab_size - 1)]
    itos = [UNK] + words
    stoi = {word: i for i, word in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(token, unk_id) for token in tokens]


class CBOWDataset(Dataset):
    def __init__(self, token_ids, window_size):
        self.token_ids = token_ids
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.token_ids) - 2 * self.window_size)

    def __getitem__(self, index):
        center = index + self.window_size
        left = self.token_ids[center - self.window_size:center]
        right = self.token_ids[center + 1:center + self.window_size + 1]

        context = torch.tensor(left + right, dtype=torch.long)
        target = torch.tensor(self.token_ids[center], dtype=torch.long)
        return context, target


class CBOWWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        context_vectors = self.embedding(context_ids)
        average_vector = context_vectors.mean(dim=1)
        return self.output(average_vector)


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
        total_examples = 0

        for context, target in loader:
            context = context.to(device)
            target = target.to(device)

            logits = model(context)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        print(f"epoch {epoch}/{epochs} loss {total_loss / total_examples:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple CBOW Word2Vec model.")
    parser.add_argument("text_path", nargs="?", default="rnn-llm/data/input.txt")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", default="cbow_submission.pt")
    args = parser.parse_args()

    with open(args.text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)[:args.max_tokens]
    split = int(len(tokens) * (1.0 - args.valid_fraction))
    train_tokens = tokens[:split]

    stoi, itos = build_vocab(train_tokens, args.vocab_size)
    token_ids = encode(train_tokens, stoi)

    dataset = CBOWDataset(token_ids, args.window_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CBOWWord2Vec(len(itos), args.embedding_dim)

    device = pick_device()
    print(f"tokens {len(train_tokens)} | vocab {len(itos)} | samples {len(dataset)} | device {device}")
    train(model, loader, args.epochs, args.lr, device)

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "config": {
                "model_type": "cbow",
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
