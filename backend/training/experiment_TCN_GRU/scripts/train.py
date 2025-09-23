# scripts/train.py
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import math

# Adjust the path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tcn_gru import TCNGRUModel
from data.dataset import PoseDataset
from evaluate import evaluate_model

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, train_loader, optimizer, device):
    """The training loop for a single epoch."""
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # NOTE: The loss calculation will be updated later
        loss = nn.MSELoss()(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # --- Data Loading and Splitting ---
    db_config = config['database']
    exercise_to_train = config['training']['exercise_name']
    
    try:
        full_dataset = PoseDataset(db_config, exercise_to_train, config)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
    except Exception as e:
        print(f"Error connecting to the database or loading data: {e}")
        return

    # --- Hyperparameter Grid Search ---
    tcn_blocks_list = config['model']['tcn']['blocks']
    tcn_dilations_list = config['model']['tcn']['dilation_levels']
    pooling_types_list = config['model']['pooling']['type']
    gru_layers_list = config['model']['gru']['layers']
    gru_units_list = config['model']['gru']['hidden']
    
    # Correctly generate the combinations for all hyperparameters
    combinations = list(product(tcn_blocks_list, tcn_dilations_list, pooling_types_list, gru_layers_list, gru_units_list))
    
    print(f"Total combinations to train: {len(combinations)}")
    
    for i, (dilations, tcn_b, pool_t, gru_l, gru_u) in enumerate(combinations):
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        print(f"Config: TCN_Blocks={tcn_b}, Dilation={dilations}, Pool={pool_t}, GRU_Layers={gru_l}, GRU_Units={gru_u}")
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        model_config = config.copy()
        model_config['model']['tcn']['blocks'] = tcn_b
        model_config['model']['tcn']['dilation_levels'] = [dilations] * tcn_b 
        model_config['model']['pooling']['type'] = pool_t
        model_config['model']['gru']['layers'] = gru_l
        model_config['model']['gru']['hidden'] = gru_u
        
        model = TCNGRUModel(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate_model(model, val_loader, device)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Optional: Save best model here
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stop_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')

        # Final Evaluation and Logging
        final_test_loss = evaluate_model(model, test_loader, device)
        log_results(config, model_config, best_val_loss, final_test_loss)
    
    print("\nAll experiments finished!")

def evaluate_model(model, data_loader, device):
    """A placeholder for the evaluation function."""
    # This will be updated to calculate the correct angle loss later
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = nn.MSELoss()(output, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def log_results(config, model_config, val_loss, test_loss):
    """Logs the results of a single experiment to a CSV file."""
    log_dir = os.path.join('..', 'results')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, config['logging']['log_file'])
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as f:
        if not file_exists:
            f.write("tcn_blocks,tcn_dilations,pooling_type,gru_layers,gru_units,val_loss,test_loss,timestamp\n")
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{model_config['model']['tcn']['blocks']},"
                f"{model_config['model']['tcn']['dilation_levels']},"
                f"{model_config['model']['pooling']['type']},"
                f"{model_config['model']['gru']['layers']},"
                f"{model_config['model']['gru']['hidden']},"
                f"{val_loss:.4f},"
                f"{test_loss:.4f},{timestamp}\n")

if __name__ == '__main__':
    # NOTE: You need to create a new config file: configs/experiment_config.yaml
    # and update the database and model sections.
    main()