import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import build_vocab, encode, load_tokens, save_vocab


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids):
        x = self.in_embed(context_ids).mean(dim=1)
        return self.out(x)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, center_ids):
        x = self.in_embed(center_ids)
        return self.out(x)


class CBOWDataset(Dataset):
    def __init__(self, ids, window):
        self.ids = ids
        self.window = window

    def __len__(self):
        return max(0, len(self.ids) - 2 * self.window)

    def __getitem__(self, idx):
        center = idx + self.window
        context = self.ids[idx:center] + self.ids[center + 1:center + self.window + 1]
        return torch.tensor(context), torch.tensor(self.ids[center])


class SkipGramDataset(Dataset):
    def __init__(self, ids, window):
        self.pairs = []
        for i, center in enumerate(ids):
            left = max(0, i - window)
            right = min(len(ids), i + window + 1)
            for j in range(left, right):
                if i != j:
                    self.pairs.append((center, ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, target = self.pairs[idx]
        return torch.tensor(center), torch.tensor(target)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_word2vec(
    model_type="skipgram",
    data_path="data/input.txt",
    out_path=None,
    vocab_path="data/vocab.json",
    min_freq=1,
    max_tokens=50000,
    window=2,
    embed_dim=64,
    batch_size=256,
    epochs=3,
    lr=1e-3,
    device="auto",
):
    tokens = load_tokens(data_path, max_tokens=max_tokens)
    stoi, itos = build_vocab(tokens, min_freq=min_freq)
    ids = encode(tokens, stoi)
    save_vocab(stoi, itos, vocab_path)

    if model_type == "cbow":
        dataset = CBOWDataset(ids, window)
        model = CBOW(len(itos), embed_dim)
    elif model_type == "skipgram":
        dataset = SkipGramDataset(ids, window)
        model = SkipGram(len(itos), embed_dim)
    else:
        raise ValueError("model_type must be 'cbow' or 'skipgram'")

    device = get_device(device)
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_items += y.numel()
        print(f"{model_type} epoch {epoch}/{epochs} loss {total_loss / total_items:.4f}")

    if out_path is None:
        out_path = f"checkpoints/word2vec_{model_type}.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "embedding_weight": model.in_embed.weight.detach().cpu(),
            "stoi": stoi,
            "itos": itos,
            "min_freq": min_freq,
            "max_tokens": max_tokens,
            "window": window,
            "embed_dim": embed_dim,
        },
        out_path,
    )
    print(f"saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cbow", "skipgram", "both"], default="both")
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--vocab-path", default="data/vocab.json")
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    models = ["cbow", "skipgram"] if args.model == "both" else [args.model]
    for model_type in models:
        train_word2vec(
            model_type=model_type,
            data_path=args.data_path,
            out_path=os.path.join(args.out_dir, f"word2vec_{model_type}.pt"),
            vocab_path=args.vocab_path,
            min_freq=args.min_freq,
            max_tokens=args.max_tokens,
            window=args.window,
            embed_dim=args.embed_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
