"""GENERATE TEXT USING THE MODEL WITH PROMPT"""

import torch
import config

def generate_philosophy(model, dataset, prompt, length=200, temperature=0.8):
    model.eval()

    with torch.no_grad():
        hidden = model.init_hidden(1, config.DEVICE)
        input_seq = torch.tensor(
            [dataset.ch_to_idx[ch] for ch in prompt],
            dtype=torch.long
        ).unsqueeze(0).to(config.DEVICE)

        # priming with prompt
        for i in range(len(prompt) - 1):
            output, hidden = model(input_seq[:, i:i+1], hidden)

        generated = prompt #starting of generated text is the prompt
        input_char = input_seq[:, -1:]

        # Generate
        for _ in range(length):
            output, hidden = model(input_char, hidden)
            output = output / temperature # controls the randomness of generated text
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs,1).item()
            generated += dataset.idx_to_ch[next_idx]
            input_char = torch.tensor([[next_idx]], dtype=torch.long, device=config.DEVICE)

    model.train()
    return generated