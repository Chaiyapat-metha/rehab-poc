# rehab-poc/backend/training/experiment_BiGRU_Att/scripts/train.py

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product
import pandas as pd
from pathlib import Path
from typing import Tuple

# Adjust path for local imports (BiGRUAttentionModel and PoseDataset)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bigru_attention_model import BiGRUAttentionModel # New Model
from data.dataset import PoseDataset

# NOTE: Need to implement the evaluation logic in evaluate.py
from evaluate import evaluate_model 

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "exp_a_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_loss(output: dict, target_delta_theta: torch.Tensor, mask: torch.Tensor, config: dict) -> Tuple[torch.Tensor, dict]:
    """Calculates the total multi-component loss for Exp A."""
    
    # 1. L_angle_MSE (Regression Target)
    pred_delta_theta = output['delta_theta_output']
    
    # L_angle_MSE on valid joints (masked)
    angle_mse = torch.sum(mask * (pred_delta_theta - target_delta_theta)**2) / (torch.sum(mask) + 1e-6)
    
    # 2. L_uncertainty (Heteroscedastic)
    w_unc = config['loss']['w_unc']
    if w_unc > 0:
        log_var = output['log_variance_output']
        # L_unc = 0.5 * exp(-logvar) * (delta_theta_error)^2 + 0.5 * logvar
        uncertainty_loss = 0.5 * torch.exp(-log_var) * (pred_delta_theta - target_delta_theta)**2 + 0.5 * log_var
        uncertainty_loss = torch.sum(mask * uncertainty_loss) / (torch.sum(mask) + 1e-6)
    else:
        uncertainty_loss = torch.tensor(0.0, device=pred_delta_theta.device)
        
    # 3. L_vel (Temporal Smoothness Loss)
    # This requires the model output to contain predictions for the whole sequence (T), not just the last frame.
    # Since our current model only outputs the last frame, we'll use a placeholder for now.
    velocity_loss = torch.tensor(0.0, device=pred_delta_theta.device)
    
    # Total Loss
    total_loss = config['loss']['w_ang'] * angle_mse + w_unc * uncertainty_loss + config['loss']['w_vel'] * velocity_loss
    
    loss_components = {
        'L_total': total_loss.item(),
        'L_angle_mse': angle_mse.item(),
        'L_uncertainty': uncertainty_loss.item(),
        'L_velocity': velocity_loss.item(),
    }
    
    return total_loss, loss_components

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, target_delta_theta, mask in train_loader:
        data = data.to(device)
        target_delta_theta = target_delta_theta.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss, _ = calculate_loss(output, target_delta_theta, mask, model.config)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ... (rest of the main function, adjusted for Exp A sweep)
# The main loop needs to be updated to loop through loss weights as well.

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

    # --- Hyperparameter Grid Search (Loss Weights) ---
    w_ang_list = config['sweep']['w_ang']
    w_unc_list = config['sweep']['w_unc']
    w_vel_list = config['sweep']['w_vel']
    lr_list = config['sweep']['lr']
    
    # 48 combinations of loss weights
    loss_weight_combinations = list(product(w_ang_list, w_unc_list, w_vel_list)) 
    
    print(f"Total combinations to train: {len(loss_weight_combinations)}") 
    
    # Loop through Loss Weights and LR (Total 48 * 3 = 144 combinations)
    for i, (w_ang, w_unc, w_vel) in enumerate(loss_weight_combinations):
        for j, lr in enumerate(lr_list):
            
            # --- Setup for this specific run ---
            run_index = i * len(lr_list) + j + 1
            print(f"\n--- Running Combination {run_index}/{len(loss_weight_combinations) * len(lr_list)} ---")
            
            train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
            # Create Model Config for this run
            model_config = config.copy()
            model_config['loss']['w_ang'] = w_ang
            model_config['loss']['w_unc'] = w_unc
            model_config['loss']['w_vel'] = w_vel
            model_config['training']['lr'] = lr
            
            model = BiGRUAttentionModel(model_config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
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
                        early_stop_epoch
                        )
    
    print("\nAll experiments finished!")

def log_results(config, model_config, val_loss, test_loss, early_stop_epoch):
    """Logs the results of a single experiment to a CSV file."""
    log_dir = os.path.join('..', 'results')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, config['logging']['log_file'])
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as f:
        # บันทึก Loss Weights และ BiGRU/Attention Hyperparameters
        if not file_exists:
            f.write("w_ang,w_unc,w_vel,lr,bigru_layers,bigru_hidden,att_heads,early_stop_epoch,val_loss,test_loss,timestamp\n")
        
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ดึงค่าจาก model_config ที่ถูก set ในลูป (Loss Weights และ LR)
        f.write(f"{model_config['loss']['w_ang']},"
                f"{model_config['loss']['w_unc']},"
                f"{model_config['loss']['w_vel']},"
                f"{model_config['training']['lr']},"
                
                f"{model_config['model']['bigru']['layers']},"
                f"{model_config['model']['bigru']['hidden']},"
                f"{model_config['model']['attention']['heads']},"

                f"{early_stop_epoch}," 
                f"{val_loss:.4f},"
                f"{test_loss:.4f},"
                f"{timestamp}\n")
if __name__ == '__main__':
    main()