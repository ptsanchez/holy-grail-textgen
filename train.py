from util import *
from train import *
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm

from fate_lstm import LSTMModel

import uuid
from torch.optim.lr_scheduler import LambdaLR
import shutil
import csv

def train(model, device, train_dataloader, val_dataloader, config):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    unique_code_str = uuid.uuid4()

    models_folder = './test/'
    model_folder = models_folder+f"{config['model']}_{unique_code_str}/"
    try:
        os.mkdir(model_folder)
        print(f"Folder '{model_folder}' created successfully.")
    except FileExistsError:
        print(f"Folder '{model_folder}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
    original_config = "./configs/config.yaml"
    
    def lr_lambda(epoch):
        if (epoch < 6):
            return 1.0
        elif (epoch < 8):
            return 0.35
        else:
            return 0.1
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    early_stopper = EarlyStopper(config['patience'], config['early_stop_epochs'])

    unique_code = uuid.uuid4()
    unique_code_str = str(unique_code)


    train_loss = []
    val_loss = []

    init_val_loss = eval(model, device, val_dataloader, config)
    val_loss.append(init_val_loss)

    min_val_loss = float("inf")
    for t in range(config['epochs']):
        epoch_train_loss = []

        lowest_epoch_avg_loss = float("inf")
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {t}: lr = {current_lr}")

        for i, (input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            input = input.to(device)
            target = target.to(device)
        
            pred = model(input)
            loss = criterion(pred, target)

            if(len(train_loss) == 0):
                train_loss.append(loss.detach().cpu().numpy())

            epoch_train_loss.append(loss.detach().cpu().numpy())

            if i % 100 == 0:
                if lowest_epoch_avg_loss < np.mean(epoch_train_loss):
                    print(f"\033[91mEpoch {t}, Batch {i}: Loss: {np.mean(epoch_train_loss)}\033[0m")
                else:
                    print(f"\033[92mEpoch {t}, Batch {i}: Loss: {np.mean(epoch_train_loss)}\033[0m")
                    lowest_epoch_avg_loss = np.mean(epoch_train_loss)

            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(epoch_train_loss)
        avg_val_loss = eval(model, device, val_dataloader, config)

        scheduler.step()
        
        if(avg_val_loss < min_val_loss):
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_folder+f"{config['model']}_seqlen{config['seq_len']}_{unique_code_str}.pth")
            shutil.copyfile(original_config, model_folder+f"config_{unique_code_str}.yaml")
        
        if(early_stopper.early_stop(avg_val_loss, t)):
            plot_losses(train_loss, val_loss, model_folder+f"{config['model']}_seqlen{config['seq_len']}_{unique_code_str}")
            return
        
        print(f"Epoch {t}:")
        print(f"Avg Training Loss: {avg_train_loss}")
        print(f"Validation Loss: {avg_val_loss}")

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        
    plot_losses(train_loss, val_loss, model_folder+f"{config['model']}_seqlen{config['seq_len']}_{unique_code_str}")
    csv_filename = model_folder + f"{config['model']}_seqlen{config['seq_len']}_{unique_code_str}.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        
        for epoch_idx in range(len(train_loss)):
            writer.writerow([epoch_idx, train_loss[epoch_idx], val_loss[epoch_idx]])
    print("CSV file saved")
    
def eval(model, device, val_dataloader, config):
    criterion = nn.CrossEntropyLoss()
    val_loss = []
    for i, (input, target) in enumerate(val_dataloader):
        input = input.to(device)
        target = target.to(device)

        pred = model(input)
        loss = criterion(pred, target)

        val_loss.append(loss.detach().cpu().numpy())
        
    return np.mean(val_loss)

class EarlyStopper():
    def __init__(self, patience, early_stop_epoch):
        self.min_val_loss = float("inf")
        self.patience = patience
        self.early_stop_epoch = early_stop_epoch
        self.increasing_loss_count = 0
    def early_stop(self, val_loss, epoch):
        if(val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.increasing_loss_count = 0
        elif(epoch >= self.early_stop_epoch):
            self.increasing_loss_count += 1
            if(self.increasing_loss_count > self.patience):
                return True
        return False
            