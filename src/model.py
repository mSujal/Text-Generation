"""LSTM MODEL DEFINITION"""

import torch
import torch.nn as nn

class PhilosophyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x : (batch_size, seq_length)
        embed = self.embedding(x) # (batch_size, seq_length, embed_size)
        out, hidden = self.lstm(embed, hidden) # (batch_size, seq_length, hidden_size)
        out = out.reshape(-1, self.hidden_size) # flatten for fc
        out = self.fc(out) # (batch_size*seq_length, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return(
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )