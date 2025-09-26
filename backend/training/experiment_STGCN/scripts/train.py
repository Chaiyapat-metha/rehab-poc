# rehab-poc/backend/training/experiment_STGCN/scripts/train.py

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path

# Adjust path for local imports (STGCNModel and PoseDataset)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stgcn_model import STGCNModel # New Model
from data.dataset import PoseDataset
from evaluate import evaluate_model # Assume evaluate_model is defined in evaluate.py

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "stgcn_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, train_loader, optimizer, device):
    """
    Function to train the model for one epoch using the full Multi-task Loss (L_pos + L_angle + L_class).
    Weights are controlled by the model.config.
    """
    model.train()
    total_loss = 0.0
    
    # Define loss functions for each task
    pos_loss_func = nn.MSELoss()         # For L_pos
    angle_loss_func = nn.MSELoss()       # For L_angle
    class_loss_func = nn.BCEWithLogitsLoss() # For L_class (Classification)
    
    # Target and Data are unpacked: data, target_pos, target_angle, target_class 
    for data, target_pos, target_angle, target_class in train_loader:
        data = data.to(device)
        target_pos = target_pos.to(device)
        target_angle = target_angle.to(device)
        
        # Target Class must be (Batch Size, 1) for BCEWithLogitsLoss
        target_class = target_class.to(device).view(-1, 1) 
        
        optimizer.zero_grad()
        output = model(data) # Output is dict: {'pos_output', 'angle_output', 'class_output'}
        
        # 1. Calculate Positional Loss
        pos_loss = pos_loss_func(output['pos_output'], target_pos)
        
        # 2. Calculate Angle Loss
        angle_loss = angle_loss_func(output['angle_output'], target_angle)
        
        # 3. Calculate Classification Loss
        class_loss = class_loss_func(output['class_output'], target_class)
        
        # Weighted sum of all 3 losses
        config = model.config
        loss = (config['loss']['w_pos'] * pos_loss + 
                config['loss']['w_angle'] * angle_loss +
                config['loss']['w_bce'] * class_loss)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    db_config = config['database']
    try:
        full_dataset = PoseDataset(db_config, config)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
    except Exception as e:
        print(f"Error connecting to the database or loading data: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = STGCNModel(config).to(device) # Use STGCNModel
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_epoch = "None"
        
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_model(model, val_loader, device)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stop_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop_epoch = epoch + 1
                break
            
        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}')

    final_test_loss = evaluate_model(model, test_loader, device)
    model_config_baseline = {
        'model': {
            'encoder': config['model']['encoder'], # 'st_gcn'
            # FIX: ดึงค่าที่เกี่ยวข้องกับ ST-GCN มาแทน
            'stgcn': {'channels': config['model']['stgcn']['channels']},
            # FIX: ใช้ค่า Placeholder ที่ชัดเจนสำหรับคีย์ที่ไม่มี
            'bigru': {'layers': 0, 'hidden': 0}, 
            'attention': {'heads': 0}
        },
        'loss': config['loss'],
        'training': config['training']
    }
    
    aug_status = False 

    log_results(config,
                model_config_baseline,
                best_val_loss,
                final_test_loss,
                aug_status,
                early_stop_epoch
                )
    print("\nAll experiments finished!")

def log_results(config, model_config, val_loss, test_loss, aug_status, early_stop_epoch):
    """Logs the results of a single experiment to a CSV file."""
    log_dir = os.path.join('..', 'results')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, config['logging']['log_file'])
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as f:
        # FIX: แก้ไข Header ให้บันทึก ST-GCN/Loss Weights
        if not file_exists:
            f.write("encoder,stgcn_channels,w_pos,w_angle,w_bce,early_stop_epoch,val_loss,test_loss,timestamp\n")
        
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # FIX: ดึงค่าจากโครงสร้างใหม่ที่สร้างขึ้น
        f.write(f"{model_config['model']['encoder']},"
                f"{len(model_config['model']['stgcn']['channels'])}," # จำนวนชั้น STGCN
                f"{model_config['loss']['w_pos']},"
                f"{model_config['loss']['w_angle']},"
                f"{model_config['loss']['w_bce']},"
                f"{early_stop_epoch},"
                f"{val_loss:.4f},"
                f"{test_loss:.4f},"
                f"{timestamp}\n")
          
if __name__ == '__main__':
    main()