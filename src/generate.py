"""GENERATE TEXT USING THE MODEL WITH PROMPT"""

import torch
import config

def generate_philosophy(model, dataset, prompt, length=200, temperature=0.8):
    """
        Generate the text output of given length and temperature

        Args:
            model : trained model
            dataset: the encoded data
            prompt: priming word
            length: length of the text
            temperature: randomness
    """
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

        generated = prompt
        input_char = input_seq[:, -1:]

        # Generate
        for _ in range(length):
            output, hidden = model(input_char, hidden)
            output = output / temperature # controls the randomness of generated text
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs,1).item()
            generated += dataset.idx_to_ch[next_idx]
            input_char = torch.tensor([[next_idx]], dtype=torch.long, device=config.DEVICE)

        # Complete the last word if not completed
        if generated and not generated[-1] in [' ', '.', '!', '?', '\n']: # basic punctuations
            for _ in range(20): # max 20 character to complete
                output, hidden = model(input_char, hidden)
                prob = torch.softmax(output/temperature, dim=1)
                char_idx = torch.multinomial(prob, 1).item()
                char = dataset.idx_to_ch[char_idx]
                generated += char
                if char in [' ', '.', '!', '?']:
                    break
                input_char = torch.tensor([[char_idx]], dtype=torch.long, device=config.DEVICE)

    model.train()
    return generated