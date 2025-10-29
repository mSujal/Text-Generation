"""TRAINING AND EVALUATING THE MODEL"""

import torch
import torch.nn as nn
import config
from src.utils import save_checkpoint
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, save_path=config.SAVE_LOSS_PATH):
    """Plot the losses for visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(train_losses)), train_losses, label="Train Loss")
    ax.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def validate(model, dataset, config, criterion):
    """
        Validate the model on random batches

        Args:
            model: model to evaluate
            dataset: validation data to get the random batch from
            config: Hyperparameters
            criterion: loss function

        Returns:
            validation loss per validation batch
    """
    model.eval()
    val_loss = 0
    num_val_batch = 10 # using 10 batches for validation

    with torch.no_grad():
        hidden = model.init_hidden(config.BATCH_SIZE, config.DEVICE)

        for _ in range(num_val_batch):
            x, y = dataset.get_batch(config.BATCH_SIZE, config.SEQ_LENGTH)
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            output, hidden = model(x, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(output, y.view(-1))
            val_loss += loss.item()

    model.train()
    return val_loss / num_val_batch

def train_model(model, optimizer, dataset, config, start_epoch=0, train_losses=None, val_losses=None):
    """
        Train the model on dataset and plot the loss

        Args:
            model: model to be trained
            optimizer: minimize the loss
            dataset: training data
            config: hyperparameters
            start_epoch: starting epochs, default is 0 and custom for resuming
            train_losses, val_losses: loss history

    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    # incase the history loss is None
    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = val_losses if val_losses else []

    print("Training Started")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        hidden = model.init_hidden(config.BATCH_SIZE, config.DEVICE)
        epoch_loss = 0

        # training loop
        for batch_idx in range(config.NUM_BATCH_PER_EPOCH):
            x, y = dataset.get_batch(config.BATCH_SIZE, config.SEQ_LENGTH)
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # prediction and hidden state
            output, hidden = model(x, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            # calculation of loss, backprop and weight update
            loss = criterion(output, y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            # avoiding backprop through entire training history
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            epoch_loss += loss.item()

        # calculation of average training loss and validation losses
        avg_train_loss = epoch_loss / config.NUM_BATCH_PER_EPOCH
        train_losses.append(avg_train_loss)
        avg_val_loss = validate(model, dataset, config, criterion)
        val_losses.append(avg_val_loss)

        # Print progress and save checkpoint every 10 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Train_Loss: {avg_train_loss:.4f}, Val_loss : {avg_val_loss}")
            # save checkpoints
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model, optimizer, epoch, avg_train_loss, avg_val_loss,
                    filepath = f'{config.CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth'
                )

    # Plot the losses
    plot_losses(train_losses, val_losses)

