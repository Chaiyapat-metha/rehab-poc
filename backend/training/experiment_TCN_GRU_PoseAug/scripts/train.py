# scripts/train.py

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from itertools import product
import math
from pathlib import Path

# Adjust the path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tcn_gru import TCNGRUModel
from data.dataset import PoseDataset
from evaluate import evaluate_model

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "poseaug_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pos_loss_func = nn.MSELoss()
    angle_loss_func = nn.MSELoss()
    class_loss_func = nn.BCEWithLogitsLoss()

    for data, target_pos, target_angle, target_class in train_loader:
        data = data.to(device)
        # 1. การจัดการ Target Class
        target_pos = target_pos.to(device)
        target_angle = target_angle.to(device)
        target_class = target_class.to(device)
        target_class = target_class.view(-1, 1)

        optimizer.zero_grad()
        output = model(data)
        
        # 2. คำนวณ 3 Loss Heads
        pos_loss = pos_loss_func(output['pos_output'], target_pos)
        angle_loss = angle_loss_func(output['angle_output'], target_angle)
        class_loss = class_loss_func(output['class_output'], target_class)
        
        config = model.config
        # 3. รวม Loss แบบถ่วงน้ำหนัก (Weighted Sum)
        loss = config['loss']['w_pos'] * pos_loss + config['loss']['w_angle'] * angle_loss + config['loss']['w_bce'] * class_loss
        
        # 4. Backpropagation
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

    tcn_blocks_list = config['model']['tcn']['blocks']
    tcn_dilations_list = config['model']['tcn']['dilation_levels']
    pooling_types_list = config['model']['pooling']['type']
    gru_layers_list = config['model']['gru']['layers']
    gru_units_list = config['model']['gru']['hidden']
    aug_status_list = [False, True]
    
    combinations = list(product(tcn_blocks_list, tcn_dilations_list, pooling_types_list, 
                                gru_layers_list, gru_units_list, aug_status_list))
   
    print(f"Total combinations to train: {len(combinations)}") # 324 * 2 = 648 Combinations
    
    for i, (tcn_b, dilations, pool_t, gru_l, gru_u, aug_status) in enumerate(combinations):
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        print(f"Config: TCN_Blocks={tcn_b}, Dilation={dilations}, Pool={pool_t}, GRU_Layers={gru_l}, GRU_Units={gru_u}, Aug_Status={aug_status}")
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        model_config = config.copy()
        model_config['model']['tcn']['blocks'] = tcn_b
        model_config['model']['tcn']['dilation_levels'] = [dilations] * tcn_b 
        model_config['model']['pooling']['type'] = pool_t
        model_config['model']['gru']['layers'] = gru_l
        model_config['model']['gru']['hidden'] = gru_u
        model_config['augmentation']['poseaug']['enabled'] = aug_status 
        
        model = TCNGRUModel(model_config).to(device)
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
        log_results(config,
                    model_config,
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
        if not file_exists:
            f.write("tcn_blocks,tcn_dilations,pooling_type,gru_layers,gru_units,aug_status,early_stop_epoch,val_loss,test_loss,timestamp\n")
        
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{model_config['model']['tcn']['blocks']},"
                f"{model_config['model']['tcn']['dilation_levels']},"
                f"{model_config['model']['pooling']['type']},"
                f"{model_config['model']['gru']['layers']},"
                f"{model_config['model']['gru']['hidden']},"
                f"{aug_status},"
                f"{early_stop_epoch}," 
                f"{val_loss:.4f},"
                f"{test_loss:.4f},"
                f"{timestamp}\n")

if __name__ == '__main__':
    main()