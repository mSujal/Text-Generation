"""HYPERPARAMETERS"""

import torch

# Data
DATA_PATH = 'data/dataset.txt'
CHECKPOINT_DIR = 'checkpoints/'
CHECKPOINT_PATH = 'checkpoints/checkpoint_epoch_200.pth'


# Model
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
SEQ_LENGTH = 100
LEARNING_RATE = 0.01
NUM_EPOCHS = 200
NUM_BATCH_PER_EPOCH = 10

# Generation
GENERATION_LENGTH = 200
TEMPERATURE = 0.8

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
