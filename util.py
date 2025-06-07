import random
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def encode_text(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    encoded_text = [char_to_idx[c] for c in text]

    return encoded_text, vocab_size, char_to_idx, idx_to_char

def create_sequences(data, seq_length):

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]) 
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


def plot_losses(train_losses, val_losses, fname):
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    train_losses = train_losses[1:] 
    val_losses = val_losses[1:]
    
    epochs = np.arange(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    plt.savefig(fname + ".png")