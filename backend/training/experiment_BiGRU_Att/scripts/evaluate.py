# rehab-poc/backend/training/experiment_BiGRU_Att/scripts/evaluate.py

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Dict

# --- Constants ---
RAD_TO_DEG = 180.0 / math.pi
ANGLE_THRESHOLD = 5.0 # PCK@5deg

def calculate_loss_components(output: dict, target_delta_theta: torch.Tensor, mask: torch.Tensor, config: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates the total multi-component loss for Exp A and returns the tensor loss and component dictionary."""
    
    pred_delta_theta = output['delta_theta_output']
    
    # 1. L_angle_MSE (Regression Target)
    error_sq = (pred_delta_theta - target_delta_theta)**2
    angle_mse = torch.sum(mask * error_sq) / (torch.sum(mask) + 1e-6)
    
    # 2. L_uncertainty (Heteroscedastic)
    w_unc = config['loss']['w_unc']
    if w_unc > 0 and 'log_variance_output' in output:
        log_var = output['log_variance_output']
        uncertainty_loss_per_joint = 0.5 * torch.exp(-log_var) * error_sq + 0.5 * log_var
        uncertainty_loss = torch.sum(mask * uncertainty_loss_per_joint) / (torch.sum(mask) + 1e-6)
    else:
        uncertainty_loss = torch.tensor(0.0, device=pred_delta_theta.device)
        
    # 3. L_vel (Temporal Smoothness Loss)
    velocity_loss = torch.tensor(0.0, device=pred_delta_theta.device)
    
    # Total Loss (returned as a tensor for backprop/total_loss calculation)
    total_loss_tensor = config['loss']['w_ang'] * angle_mse + w_unc * uncertainty_loss + config['loss']['w_vel'] * velocity_loss
    
    loss_components = {
        'L_total': total_loss_tensor.item(), # Return scalar item for the dictionary
        'L_angle_mse': angle_mse.item(),
        'L_uncertainty': uncertainty_loss.item(),
        'L_velocity': velocity_loss.item(),
    }
    
    return total_loss_tensor, loss_components


def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluates the model and computes key metrics, returning the primary validation loss (L_total).
    
    NOTE: Evaluation function should return a SINGLE value (e.g., L_total or L_angle_MSE)
    for the Early Stopping logic to work correctly. We choose L_total.
    """
    model.eval()
    total_loss = 0.0
    total_mae_deg = 0.0
    correct_pck_joints = 0
    total_valid_joints = 0
    
    all_loss_components = {'L_angle_mse': 0.0, 'L_uncertainty': 0.0, 'L_velocity': 0.0}
    
    with torch.no_grad():
        for data, target_delta_theta, mask in data_loader:
            data = data.to(device)
            target_delta_theta = target_delta_theta.to(device)
            mask = mask.to(device)
            
            output = model(data)
            
            # --- 1. Loss Calculation ---
            total_loss_tensor, loss_comps = calculate_loss_components(output, target_delta_theta, mask, model.config)
            total_loss += total_loss_tensor.item()
            
            # Aggregate loss components for logging
            for key in all_loss_components:
                 all_loss_components[key] += loss_comps[key]
            
            # --- 2. Metric Calculation (MAE and PCK) ---
            pred_delta_theta = output['delta_theta_output']
            
            # Angle Error (Absolute Difference)
            error_rad = torch.abs(pred_delta_theta - target_delta_theta)
            error_deg = error_rad * RAD_TO_DEG
            
            # Masked MAE (Mean Absolute Error)
            mae_deg_sum = torch.sum(mask * error_deg)
            num_valid_joints = torch.sum(mask)
            
            total_mae_deg += mae_deg_sum.item()
            total_valid_joints += num_valid_joints.item()
            
            # PCK (Percent Correct Keypoints)
            pck_correct = (error_deg <= ANGLE_THRESHOLD).float()
            correct_pck_joints += torch.sum(mask * pck_correct).item()

    # --- Final Metrics Calculation ---
    num_batches = len(data_loader)
    avg_total_loss = total_loss / num_batches
    
    avg_mae_deg = total_mae_deg / (total_valid_joints + 1e-6)
    pck_at_5deg = correct_pck_joints / (total_valid_joints + 1e-6) * 100.0

    # Log/Print Detailed Metrics (Optional, but highly recommended for analysis)
    print(f"\n[EVAL METRICS] Loss: {avg_total_loss:.4f} | MAE(deg): {avg_mae_deg:.2f}° | PCK@5deg: {pck_at_5deg:.2f}%")
    print(f"[COMPONENTS] L_ang: {all_loss_components['L_angle_mse']/num_batches:.6f}, L_unc: {all_loss_components['L_uncertainty']/num_batches:.6f}")

    # The function must return a single scalar for the early stopping logic.
    return avg_total_loss 

# NOTE: The 'evaluate_model' function in train.py currently requires only 'val_loss' (total loss). 
# We will use avg_total_loss for the early stopping.
# Detailed metrics (MAE, PCK) are printed to the console during evaluation.