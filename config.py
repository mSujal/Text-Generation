"""HYPERPARAMETERS"""

# Data
import torch

DATA_PATH = '../data/dataset.txt'
CHECKPOINT_DIR = 'checkpoints/'

# Model
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
SEQ_LENGTH = 100
LEARNING_RATE = 0.002
NUM_EPOCHS = 200
NUM_BATCHES_PER_EPOCHS = 20

# Generation
GENERATION_LENGTH = 200
TEMPERATURE = 0.8

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
