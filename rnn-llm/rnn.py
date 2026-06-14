import torch
from torch import nn


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_weight=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_weight is not None:
            self.embedding.weight.data.copy_(embedding_weight)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        x = self.embedding(input_ids)
        x, hidden = self.rnn(x, hidden)
        logits = self.fc(x)
        return logits, hidden
