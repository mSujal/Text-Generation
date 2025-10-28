"""MAIN SCRIPT"""
from tkinter import PhotoImage

import config
from src.model import PhilosophyLSTM
from src.data_utils import TextData
from src.train import train_model
from src.utils import load_checkpoint
from src.generate import generate_philosophy

def main():
    # Load the data
    dataset = TextData(config.DATA_PATH)
    print(f"vocab size :  {dataset.vocab_size}")

    # creating model
    model = PhilosophyLSTM(
        dataset.vocab_size,
        config.EMBED_SIZE,
        config.HIDDEN_SIZE,
        config.NUM_LAYERS,
    ).to(config.DEVICE)

    # try loading the checkpoint if available
    start_epoch, _ = load_checkpoint(model, None, 'checkpoints/latest.pth')
    
    # train the model
    train_model(model, dataset, config, start_epoch)

    # generate the text
    # text prompts
    topics = ["Love", "Death", "Time", "Freedom", "Truth"]
    for topic in topics:
        print(f"\n{'='*50}")
        print(f"Topic : {topic}")
        print("="*50)
        generated = generate_philosophy(model, dataset, prompt=f"{topic}: ", length=200, temperature=0.8)
        print(generated)

if __name__ == "__main__":
    main()
