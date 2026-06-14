import argparse

import torch
import torch.nn.functional as F

from dataset import decode, encode, tokenize
from rnn import RNNLanguageModel
from train_rnn import load_checkpoint
from word2vec import get_device


def sample_next(logits, temperature=1.0, top_k=20):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        probs = F.softmax(values, dim=-1)
        return indices[torch.multinomial(probs, 1)].item()
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/rnn_llm.pt")
    parser.add_argument("--prompt", default="to be or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    model = RNNLanguageModel(
        vocab_size=len(itos),
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = encode(tokenize(args.prompt), stoi)
    if not ids:
        ids = [stoi["<unk>"]]

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = ids[-ckpt["seq_len"]:]
            x = torch.tensor([context], device=device)
            logits, _ = model(x)
            next_id = sample_next(logits[0, -1], args.temperature, args.top_k)
            ids.append(next_id)

    print(decode(ids, itos))


if __name__ == "__main__":
    main()
