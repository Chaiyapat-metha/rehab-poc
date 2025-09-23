# scripts/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on a given data loader and returns the average loss.
    This function is used for both validation and final test set evaluation.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            
            # The loss calculation is for 0-1 normalized angle differences
            loss = nn.MSELoss()(output, target)
            total_loss += loss.item()

    return total_loss / len(data_loader)