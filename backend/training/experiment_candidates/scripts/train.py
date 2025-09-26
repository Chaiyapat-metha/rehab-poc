# rehab-poc/backend/training/experiment_candidates/scripts/train.py

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import product
import pandas as pd
import numpy as np
import random
import time 
from typing import Dict, Tuple
from pathlib import Path

# --- Path Setup ---
# Adjust path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- Import Custom Modules (Assumed to be implemented) ---
from models.model_archs import get_candidate_model
from data.dataset import PoseDataset
from evaluate import evaluate_model 
# import db # For optional tracking
# --- End Imports ---

# --- Fixed Global Configuration (Must match candidates_config.yaml) ---
SEED_SPLIT = 42
EXPERIMENT_SEEDS = [0, 11, 22, 33, 44]
CANDIDATE_ARCHS = [
    'TCN_GRU_BASELINE', 
    'BIGRU_ATTENTION', 
    'STGCN_BASELINE', 
    'STGCN_ATTENTION', 
    'STGCN_GRU'
]
# --- End Fixed Config ---


def set_seed(seed: int):
    """Sets seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "candidates_config.yaml"
    with open(config_path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

# --- Loss Calculation ---
def calculate_loss(output: dict, target_pos: torch.Tensor, target_angle: torch.Tensor, target_class: torch.Tensor, config: dict) -> Tuple[torch.Tensor, dict]:
    """Calculates the total multi-task loss (L_pos + L_angle + L_bce)."""
    
    pos_loss_func = nn.MSELoss()
    angle_loss_func = nn.MSELoss()
    class_loss_func = nn.BCEWithLogitsLoss()
    
    # 1. Individual Losses
    pos_loss = pos_loss_func(output['pos_output'], target_pos)
    angle_loss = angle_loss_func(output['angle_output'], target_angle)
    class_loss = class_loss_func(output['class_output'], target_class)
    
    # 2. Weighted Sum
    loss = (config['loss']['w_pos'] * pos_loss + 
            config['loss']['w_angle'] * angle_loss +
            config['loss']['w_bce'] * class_loss)
    
    loss_components = {
        'L_total': loss.item(),
        'L_pos': pos_loss.item(),
        'L_angle': angle_loss.item(),
        'L_bce': class_loss.item(),
    }
    
    return loss, loss_components

def train_one_epoch(model, train_loader, optimizer, device, config):
    """Runs a single training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    
    for data, target_pos, target_angle, target_class in train_loader:
        data = data.to(device)
        target_pos = target_pos.to(device)
        target_angle = target_angle.to(device)
        target_class = target_class.to(device).view(-1, 1) # (B, 1) for BCE

        optimizer.zero_grad()
        output = model(data)
        
        loss, _ = calculate_loss(output, target_pos, target_angle, target_class, config)
        
        loss.backward()
        
        # Apply Gradient Clipping (FIXED_HYPERPARAMS['grad_clip'] = 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# --- Logging and Metric Functions ---
def log_results(log_data: Dict):
    """Logs the complete result set to a CSV file."""
    config = log_data['config']
    metrics = log_data.get('metrics', {})
    arch_config = log_data.get('arch_config')

    # Prefer top-level peak_gpu value, otherwise fallback to metrics dict, else 0
    peak_gpu_mb = log_data.get('peak_gpu_mem_mb', metrics.get('peak_gpu_mem_mb', 0))

    log_dir = os.path.join(PROJECT_ROOT, config['logging']['log_dir'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, config['logging']['log_file'])

    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as f:
        # Define Header (All 10+ metrics/params)
        if not file_exists:
            f.write("seed,arch_name,total_params,w_pos,w_angle,w_bce,early_stop_epoch,val_loss_final,test_mpjpe_mean,test_mae_deg,test_roc_auc,test_f1,convergence_time_s,peak_gpu_mb,timestamp\n")

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use .get() to avoid KeyError if some metric is missing
        f.write(
            f"{log_data.get('seed','')},"  # seed
            f"{log_data.get('arch_name','')},"  # arch_name
            f"{log_data.get('num_params', 0)},"  # total_params
            f"{config['loss'].get('w_pos', '')},"  # w_pos
            f"{config['loss'].get('w_angle', '')},"  # w_angle
            f"{config['loss'].get('w_bce', '')},"  # w_bce
            f"{log_data.get('early_stop_epoch','')},"  # early_stop_epoch
            f"{log_data.get('best_val_loss', float('nan')):.4f},"  # val_loss_final
            f"{metrics.get('mpjpe_mean', float('nan')):.4f},"  # test_mpjpe_mean
            f"{metrics.get('mae_deg_mean', float('nan')):.2f},"  # test_mae_deg
            f"{metrics.get('roc_auc', float('nan')):.4f},"  # test_roc_auc
            f"{metrics.get('f1_score', float('nan')):.4f},"  # test_f1
            f"{log_data.get('convergence_time', 0):.2f},"  # convergence_time_s
            f"{peak_gpu_mb:.0f},"  # peak_gpu_mb
            f"{timestamp}\n"
        )


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} | CUDA: {torch.version.cuda} | Total Runs: {len(CANDIDATE_ARCHS) * len(EXPERIMENT_SEEDS)}')
    
    # --- Step 1: Fixed Data Split (SEED_SPLIT=42) ---
    set_seed(SEED_SPLIT)
    full_dataset = PoseDataset(config['database'], config) 
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Generate and save fixed split indices
    split_indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = split_indices[:train_size]
    val_indices = split_indices[train_size:train_size + val_size]
    test_indices = split_indices[train_size + val_size:]
    print(f"Fixed Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # --- Step 2: Main Experiment Loop (Archs * Seeds) ---
    
    for arch_name in CANDIDATE_ARCHS:
        for seed in EXPERIMENT_SEEDS:
            
            # 2.1. Set Unique Seed for Training Run
            set_seed(seed)
            print(f"\n--- Running {arch_name} | Seed: {seed} ---")
            
            # 2.2. Create DataLoaders using Fixed Indices
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_dataset, val_indices)
            test_subset = torch.utils.data.Subset(full_dataset, test_indices)

            train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=config['training']['batch_size'], shuffle=False)

            # 2.3. Build Model and Optimizer
            # NOTE: get_candidate_model must use the arch_configs block in the YAML file
            model = get_candidate_model(arch_name, config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # --- Training Loop and Early Stopping ---
            best_val_loss = float('inf')
            patience_counter = 0
            early_stop_epoch = config['training']['epochs']
            start_time = time.time()
            
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()

            for epoch in range(config['training']['epochs']):
                train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
                
                # NOTE: evaluate_model must be updated to return total_loss for monitoring
                val_loss, metrics_placeholder = evaluate_model(model, val_loader, device) 
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # NOTE: Save best model checkpoint here!
                else:
                    patience_counter += 1
                    if patience_counter >= config['training']['early_stop_patience']:
                        early_stop_epoch = epoch + 1
                        print(f"Early stopping at epoch {early_stop_epoch}")
                        break
                
                print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # --- Step 3: Final Evaluation and Logging ---
            convergence_time = time.time() - start_time
            
            # NOTE: Load best model weights before final evaluation (not implemented here)
            
            # Final Evaluation on Test Set
            final_loss, final_metrics = evaluate_model(model, test_loader, device)
            
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0

            log_data = {
                'seed': seed, 'arch_name': arch_name, 'num_params': num_params, 
                'config': config, 'arch_config': config['arch_configs'].get(arch_name),
                'best_val_loss': best_val_loss, 'test_mpjpe_mean': final_metrics['mpjpe_mean'],
                'test_mae_deg_mean': final_metrics['mae_deg_mean'], 'test_roc_auc': final_metrics['roc_auc'],
                'test_f1': final_metrics['f1_score'], 'early_stop_epoch': early_stop_epoch,
                'convergence_time': convergence_time, 'metrics': final_metrics,
                'peak_gpu_mem_mb': peak_gpu_mem_mb
            }
            log_results(log_data)
            
            print(f"✅ Run finished in {convergence_time:.2f}s. Final MAE: {final_metrics.get('mae_deg_mean', 0):.2f}°")

if __name__ == '__main__':
    main()