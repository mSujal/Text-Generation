"""CHECKPOINTS THE MODEL"""

import torch
import os
import config

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_losses=None, val_losses=None, filepath=config.CHECKPOINT_DIR):
    """Save model checkpoint"""
    # create dir if not present
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_losses': train_losses if train_losses is not None else [],
        'val_losses' : val_losses if val_losses is not None else []
    }
    torch.save(checkpoint, filepath)
    print(f"checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath=config.CHECKPOINT_PATH):
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        print("No checkpoints found, starting fresh")
        return 0, float('inf'), float('inf')
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint.get('val_loss', float('inf')) # if no val_loss just get infinity
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    print(f'Loaded checkpoint from epoch {epoch}')
    print(f'Train Loss : {train_loss:.4f}, Val_loss: {val_loss:.4f}')

    return epoch + 1, train_loss, val_loss, train_losses, val_losses # return next epoch to start from and losses
