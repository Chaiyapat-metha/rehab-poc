# rehab-poc/backend/training/experiment_STGCN/scripts/evaluate.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluates the model and returns the average combined loss (L_total).
    """
    model.eval()
    total_loss = 0.0
    
    # Define loss functions for each task
    pos_loss_func = nn.MSELoss()         
    angle_loss_func = nn.MSELoss()       
    class_loss_func = nn.BCEWithLogitsLoss() 
    
    with torch.no_grad():
        for data, target_pos, target_angle, target_class in data_loader:
            data = data.to(device)
            target_pos = target_pos.to(device)
            target_angle = target_angle.to(device)
            target_class = target_class.to(device).view(-1, 1) # Reshape for BCE
            
            output = model(data)
            
            # 1. Calculate individual losses
            pos_loss = pos_loss_func(output['pos_output'], target_pos)
            angle_loss = angle_loss_func(output['angle_output'], target_angle)
            class_loss = class_loss_func(output['class_output'], target_class)
            
            # 2. Weighted sum of losses
            config = model.config
            loss = (config['loss']['w_pos'] * pos_loss + 
                    config['loss']['w_angle'] * angle_loss +
                    config['loss']['w_bce'] * class_loss)
            
            total_loss += loss.item()

    # NOTE: We return the average total loss for Early Stopping (required by train.py logic)
    return total_loss / len(data_loader)