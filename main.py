"""MAIN SCRIPT"""

import torch
import os
import config
import argparse
from src.model import PhilosophyLSTM
from src.data_utils import TextData
from src.train import train_model
from src.utils import load_checkpoint
from src.generate import generate_philosophy

def main(args):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if args.mode in ['train', 'resume']: # if --mode train or --mode resume train the mode from scratch or resume
        if args.mode == 'train': # fresh train
            train_model(model=model, optimizer=optimizer, dataset=dataset, config=config, start_epoch=0)
        elif args.mode == 'resume': # continue training
            if not os.path.exists(config.CHECKPOINT_PATH):
                train_model(model=model, optimizer=optimizer, dataset=dataset, config=config, start_epoch=0)
            else:
                start_epoch, _, _, train_losses, val_losses = load_checkpoint(model, optimizer, filepath=config.CHECKPOINT_PATH)
                train_model(model=model, optimizer=optimizer, dataset=dataset, config=config, start_epoch=start_epoch,
                            train_losses=train_losses, val_losses=val_losses)

    elif args.mode == 'eval': # if --mode eval generate the text
        if os.path.exists(config.CHECKPOINT_PATH): # if saved checkpoint exists call the checkpoint to generate
            load_checkpoint(model=model, optimizer=None, filepath=config.CHECKPOINT_PATH)
            model.eval() # set to evaluation mode
            # generate the text
            # text prompts
            topics = ["Love", "Death", "Time", "Freedom", "Truth"]
            for topic in topics:
                print(f"\n{'=' * 50}")
                print(f"Topic : {topic}")
                print("=" * 50)
                generated = generate_philosophy(model, dataset, prompt=f"{topic}: ", length=200, temperature=0.8)
                print(generated)
        else: # no saved checkpoints available
            print("No checkpoints found in directory")
            training = input("Train the model? (y/n): ").lower().strip()
            if training in ['y', 'yes']:
                train_model(model=model, optimizer=optimizer, dataset=dataset, config=config, start_epoch=0)
                model.eval()
                topics = ["Love", "Death", "Time", "Freedom", "Truth"]
                for topic in topics:
                    print(f"\n{'=' * 50}")
                    print(f"Topic : {topic}")
                    print("=" * 50)
                    generated = generate_philosophy(model, dataset, prompt=f"{topic}: ", length=200, temperature=0.8)
                    print(generated)
            else:
                print("No trained model\nExiting...")
                return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Philosophy LSTM Text Generation")
    parser.add_argument('--mode',
                        choices=['train', 'resume', 'eval'],
                        default='resume',
                        help="train:  from scratch | resume: resume from checkpoint | eval: evaluate model or generate"
                        )
    args = parser.parse_args()
    main(args)