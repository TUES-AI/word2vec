import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open("./text8.txt", "r") as f:
    corpus = f.readline()

from collections import Counter

VOCAB_SIZE = 50_000

words = corpus.split()

counts      = Counter(words)
top_words   = [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]  # reserve 1 slot for <UNK>
vocab       = ["<UNK>"] + top_words
w2i         = {w: i for i, w in enumerate(vocab)}
i2w         = {i: w for w, i in w2i.items()}

words = [w if w in w2i else "<UNK>" for w in words]

print(f"Tokens: {len(words):,}  |  Vocab: {len(vocab):,}")

# text8 has no punctuation — treat every 1000 tokens as a "sentence"
CHUNK = 1000
sentences = [words[i:i + CHUNK] for i in range(0, len(words), CHUNK)]


class Word2VecDataset(Dataset):
    """Returns (context_ids, target_id) for CBOW or (target_id, context_ids) for Skip-gram."""

    def __init__(self, sentences, w2i, window=2, mode="cbow"):
        assert mode in ("cbow", "skipgram")
        self.mode    = mode
        self.w2i     = w2i
        self.window  = window
        self.samples = []

        for sent in sentences:
            ids = [w2i[w] for w in sent if w in w2i]
            for i in range(window, len(ids) - window):
                target  = ids[i]
                context = ids[i - window:i] + ids[i + 1:i + window + 1]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        context = torch.tensor(context, dtype=torch.long)  # [2*window]
        target  = torch.tensor(target,  dtype=torch.long)  # scalar
        if self.mode == "cbow":
            return context, target
        else:
            return target, context


class CBOW(nn.Module):
    """4 context indices -> averaged embedding -> 1 target logit vector."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, context):
        # context: [B, 2*window]  (indices)
        x = self.embed(context).mean(dim=1)  # [B, embed_dim]
        return self.linear(x)                # [B, vocab_size]


class SkipGram(nn.Module):
    """1 target index -> embedding -> logit vector per context position."""
    def __init__(self, vocab_size, embed_dim, window=2):
        super().__init__()
        self.window = window
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, target):
        # target: [B]  (indices)
        x      = self.embed(target)                                    # [B, embed_dim]
        logits = self.linear(x)                                        # [B, vocab_size]
        return logits.unsqueeze(1).expand(-1, 2 * self.window, -1)    # [B, 2*window, vocab_size]


def train(model, loader, optimizer, criterion, device, mode, epochs=5):
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for ctx, tgt in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            ctx = ctx.to(device)   # [B, 4, vocab]
            tgt = tgt.to(device)   # [B, vocab]

            if mode == "cbow":
                logits = model(ctx)                          # [B, vocab]
                loss   = criterion(logits, tgt)              # tgt: [B] indices
            else:
                logits = model(tgt)                          # [B, 4, vocab]
                # ctx: [B, 4] indices — loss over each context position
                loss = sum(criterion(logits[:, i], ctx[:, i]) for i in range(logits.size(1))) / logits.size(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}  loss: {total_loss / len(loader):.4f}")


if __name__ == "__main__":
    WINDOW     = 2
    BATCH_SIZE = 256
    EPOCHS     = 5
    LR         = 1e-3
    MODE       = "cbow"   # switch to "skipgram" to train skip-gram

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")

    ds     = Word2VecDataset(sentences, w2i, window=WINDOW, mode=MODE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Samples: {len(ds):,}")

    EMBED_DIM = 128

    if MODE == "cbow":
        model = CBOW(VOCAB_SIZE, EMBED_DIM)
    else:
        model = SkipGram(VOCAB_SIZE, EMBED_DIM, window=WINDOW)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train(model, loader, optimizer, criterion, device, mode=MODE, epochs=EPOCHS)

    # --- Statistics ---
    embeddings = model.embed.weight.detach().cpu()           # [vocab, embed_dim]
    norms      = embeddings.norm(dim=1)
    print(f"\nEmbedding norms  —  mean: {norms.mean():.4f}  std: {norms.std():.4f}  min: {norms.min():.4f}  max: {norms.max():.4f}")

    TOP_N = 200   # words to plot
    plot_words   = vocab[1:TOP_N + 1]  # skip <UNK>
    plot_indices = [w2i[w] for w in plot_words]
    plot_vecs    = embeddings[plot_indices].numpy()

    # --- PCA 2D ---
    pca   = PCA(n_components=2)
    vecs2d = pca.fit_transform(plot_vecs)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(vecs2d[:, 0], vecs2d[:, 1], s=10, alpha=0.6)
    for i, word in enumerate(plot_words):
        ax.annotate(word, (vecs2d[i, 0], vecs2d[i, 1]), fontsize=7, alpha=0.8)
    ax.set_title(f"Word2Vec ({MODE}) — PCA of top {TOP_N} words")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig("word2vec_pca.png", dpi=150)
    plt.show()
    print("Plot saved to word2vec_pca.png")
