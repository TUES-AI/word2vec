import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import encode, load_tokens
from rnn import RNNLanguageModel
from word2vec import get_device, train_word2vec


class NextWordDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--checkpoint", default="checkpoints/rnn_llm.pt")
    parser.add_argument("--word2vec-model", choices=["cbow", "skipgram"], default="skipgram")
    parser.add_argument("--word2vec-checkpoint", default=None)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    if args.word2vec_checkpoint is None:
        args.word2vec_checkpoint = f"checkpoints/word2vec_{args.word2vec_model}.pt"

    if not os.path.exists(args.word2vec_checkpoint):
        train_word2vec(
            model_type=args.word2vec_model,
            data_path=args.data_path,
            out_path=args.word2vec_checkpoint,
            min_freq=args.min_freq,
            max_tokens=args.max_tokens,
            window=args.window,
            embed_dim=args.embed_dim,
            device=str(device),
        )

    w2v = load_checkpoint(args.word2vec_checkpoint, device="cpu")
    stoi = w2v["stoi"]
    itos = w2v["itos"]
    embedding_weight = w2v["embedding_weight"]

    tokens = load_tokens(args.data_path, max_tokens=args.max_tokens)
    ids = encode(tokens, stoi)
    dataset = NextWordDataset(ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RNNLanguageModel(
        vocab_size=len(itos),
        embed_dim=embedding_weight.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_weight=embedding_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_items = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(itos)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_items += y.numel()
        print(f"rnn epoch {epoch}/{args.epochs} loss {total_loss / total_items:.4f}")

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "embed_dim": embedding_weight.shape[1],
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "word2vec_checkpoint": args.word2vec_checkpoint,
        },
        args.checkpoint,
    )
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
