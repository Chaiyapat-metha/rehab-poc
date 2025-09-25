# rehab-poc/backend/training/experiment_TCN_GRU_PoseAug/scripts/evaluate.py

import torch
import torch.nn as nn
import numpy as np

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on a given data loader and returns the average combined loss.
    """
    model.eval()
    total_loss = 0.0
    
    # Define loss functions
    pos_loss_func = nn.MSELoss()
    angle_loss_func = nn.MSELoss()
    class_loss_func = nn.BCEWithLogitsLoss() # New loss function
    
    with torch.no_grad():
        # แก้ไข Loop ให้รับ Target 4 ตัว
        for data, target_pos, target_angle, target_class in data_loader:
            data = data.to(device)
            target_pos = target_pos.to(device)
            target_angle = target_angle.to(device)
            # เพิ่ม unsqueeze(1) ให้ target_class เพื่อให้เข้ากับ BCEWithLogitsLoss
            target_class = target_class.to(device).unsqueeze(1) 
            
            output = model(data)
            
            # Calculate individual losses
            pos_loss = pos_loss_func(output['pos_output'], target_pos)
            angle_loss = angle_loss_func(output['angle_output'], target_angle)
            class_loss = class_loss_func(output['class_output'], target_class) # New Classification Loss
            
            # Weighted sum of losses
            config = model.config
            loss = (config['loss']['w_pos'] * pos_loss + 
                    config['loss']['w_angle'] * angle_loss +
                    config['loss']['w_bce'] * class_loss) # ต้องเพิ่ม w_bce ใน config
            
            total_loss += loss.item()

    return total_loss / len(data_loader)