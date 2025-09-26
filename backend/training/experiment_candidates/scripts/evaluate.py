# rehab-poc/backend/training/experiment_candidates/scripts/evaluate.py

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# --- Constants ---
RAD_TO_DEG = 180.0 / math.pi

# def calculate_metrics(output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: Dict) -> Dict[str, float]:
#     """Calculates all quality metrics (MPJPE, MAE, AUC, F1)."""
    
#     # 1. Prepare Tensors (Ensure on CPU for Scikit-learn)
#     pred_pos = output['pos_output'].cpu().numpy()
#     gt_pos = targets['pos'].cpu().numpy()
    
#     pred_angle = output['angle_output'].cpu().numpy()
#     gt_angle = targets['angle'].cpu().numpy()
    
#     pred_class_logits = output['class_output'].cpu()
#     gt_class = targets['class'].cpu().numpy()
    
#     # Sigmoid prediction for classification metrics
#     pred_class_prob = torch.sigmoid(pred_class_logits).numpy()
#     pred_class_binary = (pred_class_prob > 0.5).astype(int)
    
#     # --- A. Regression Metrics ---
#     # MPJPE (Position)
#     mpjpe = np.sqrt(np.sum((pred_pos - gt_pos)**2, axis=-1)).mean() # Assuming error is normalized
    
#     # Angle MAE (Convert from Radian to Degree)
#     mae_rad = mean_absolute_error(gt_angle, pred_angle)
#     mae_deg = mae_rad * RAD_TO_DEG
    
#     # --- B. Classification Metrics ---
#     try:
#         roc_auc = roc_auc_score(gt_class, pred_class_prob)
#     except ValueError:
#         # Handle case where one class is not present in the batch
#         roc_auc = 0.5
        
#     accuracy = accuracy_score(gt_class, pred_class_binary)
#     # Use 'binary' average as we only have one class output
#     f1 = f1_score(gt_class, pred_class_binary, average='binary', zero_division=0)
    
#     return {
#         'mpjpe_mean': mpjpe,
#         'mae_deg_mean': mae_deg,
#         'roc_auc': roc_auc,
#         'accuracy': accuracy,
#         'f1_score': f1
#     }

def _get_tensor_from_dict(d: Dict[str, torch.Tensor], candidates):
    """Return first matching tensor from dict for any name in candidates list."""
    for name in candidates:
        if name in d:
            return d[name]
    raise KeyError(f"None of the candidate keys {candidates} found in dict. Available keys: {list(d.keys())}")

def calculate_metrics(output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: Dict) -> Dict[str, float]:
    """Calculates all quality metrics (MPJPE, MAE, AUC, F1).
       Accepts either keys with suffix '_output' or plain keys ('pos','angle','class').
    """
    # Support both naming styles
    pred_pos_t = _get_tensor_from_dict(output, ['pos_output', 'pos'])
    gt_pos_t   = _get_tensor_from_dict(targets, ['pos', 'pos_output'])

    pred_angle_t = _get_tensor_from_dict(output, ['angle_output', 'angle'])
    gt_angle_t   = _get_tensor_from_dict(targets, ['angle', 'angle_output'])

    pred_class_t = _get_tensor_from_dict(output, ['class_output', 'class'])
    gt_class_t   = _get_tensor_from_dict(targets, ['class', 'class_output'])

    # Ensure tensors are on CPU / numpy for sklearn usage
    pred_pos = pred_pos_t.cpu().numpy()
    gt_pos = gt_pos_t.cpu().numpy()

    pred_angle = pred_angle_t.cpu().numpy()
    gt_angle = gt_angle_t.cpu().numpy()

    # For classification, keep logits then apply sigmoid
    pred_class_logits = pred_class_t.cpu()
    gt_class = gt_class_t.cpu().numpy().ravel()

    pred_class_prob = torch.sigmoid(pred_class_logits).numpy().ravel()
    pred_class_binary = (pred_class_prob > 0.5).astype(int)

    # --- A. Regression Metrics ---
    # MPJPE (Position) — sum squared per-sample over last axis then sqrt then mean
    # pred_pos and gt_pos expected shape: (N, D) where D = V*C (flattened joints)
    mpjpe = np.sqrt(np.sum((pred_pos - gt_pos) ** 2, axis=-1)).mean()

    # Angle MAE (Convert from Radian to Degree)
    mae_rad = mean_absolute_error(gt_angle, pred_angle)
    mae_deg = mae_rad * RAD_TO_DEG

    # --- B. Classification Metrics ---
    try:
        roc_auc = roc_auc_score(gt_class, pred_class_prob)
    except ValueError:
        # e.g., single-class batch
        roc_auc = 0.5

    accuracy = accuracy_score(gt_class, pred_class_binary)
    f1 = f1_score(gt_class, pred_class_binary, average='binary', zero_division=0)

    return {
        'mpjpe_mean': mpjpe,
        'mae_deg_mean': mae_deg,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, Dict]:
    """Evaluates the model and computes total loss and quality metrics."""
    model.eval()
    total_loss = 0.0
    all_outputs = {'pos': [], 'angle': [], 'class': []}
    all_targets = {'pos': [], 'angle': [], 'class': []}
    
    # Load loss configuration (for loss calculation only)
    config = model.config
    
    with torch.no_grad():
        for data, target_pos, target_angle, target_class in data_loader:
            data = data.to(device)
            target_pos = target_pos.to(device)
            target_angle = target_angle.to(device)
            target_class = target_class.to(device).view(-1, 1) # Reshape for BCE
            
            output = model(data)
            
            # 1. Total Loss Calculation (Using the same weighted sum as training)
            pos_loss = nn.MSELoss()(output['pos_output'], target_pos)
            angle_loss = nn.MSELoss()(output['angle_output'], target_angle)
            class_loss = nn.BCEWithLogitsLoss()(output['class_output'], target_class)
            
            loss = (config['loss']['w_pos'] * pos_loss + 
                    config['loss']['w_angle'] * angle_loss +
                    config['loss']['w_bce'] * class_loss)
            
            total_loss += loss.item()

            # 2. Store outputs and targets for final metric calculation
            all_outputs['pos'].append(output['pos_output'])
            all_outputs['angle'].append(output['angle_output'])
            all_outputs['class'].append(output['class_output'])

            all_targets['pos'].append(target_pos)
            all_targets['angle'].append(target_angle)
            all_targets['class'].append(target_class)

    avg_total_loss = total_loss / len(data_loader)
    
    # Concatenate all outputs/targets
    # final_outputs = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
    # final_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    
    final_outputs = {f"{k}_output": torch.cat(v, dim=0) for k, v in all_outputs.items()}
    final_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}

    # Calculate Quality Metrics
    quality_metrics = calculate_metrics(final_outputs, final_targets, config)

    # NOTE: Early stopping monitors the total loss
    return avg_total_loss, quality_metrics