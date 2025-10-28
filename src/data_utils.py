"""DATA PREPROCESSING AND TOKENIZING"""

import torch
import config

class TextData:
    def __init__(self, filepath=config.DATA_PATH):
        with open(filepath, 'r', encoding="utf-8") as f:
            self.text = f.read()

        # tokenizing the text
        self.char = sorted(list(set(self.text)))
        self.vocab_size = len(self.char)
        self.ch_to_idx = {ch: i for i, ch in enumerate(self.char)}
        self.idx_to_ch = {i: ch for i, ch in enumerate(self.char)}
        self.data = torch.tensor([self.ch_to_idx[ch] for ch in self.text], dtype=torch.long)

    def get_batch(self, batch_size, seq_length):
        max_start_idx = len(self.data) - seq_length - 1
        start_idx = torch.randint(0, max_start_idx, (batch_size, ))

        # get shifted and unshifted data sequence
        # so that is 'this' is work x='t' y='h' and so on
        x = torch.stack([self.data[i:i+seq_length] for i in start_idx])
        y = torch.stack([self.data[i+1:i+seq_length+1] for i in start_idx])
        return x, y

