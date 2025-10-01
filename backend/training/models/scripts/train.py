# File: .\backend\training\models\scripts\train.py

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
from tqdm import tqdm

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from backend.training.models.data.dataset import MultiTaskPoseDataset, window_collate_fn
from backend.training.models.models.model import MultiTaskModel
from backend.app.config import load_config

# กำหนด Path สำหรับ Checkpoints
CHECKPOINT_DIR = Path("./backend/training_outputs/weights")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_PRINTED = False 

def get_loss_function(loss_name):
    """Utility to map loss name string to PyTorch loss function."""
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "L1Loss":
        return nn.L1Loss()
    elif loss_name == "GaussianNLLLoss":
        # Reduction='none' is important for masking
        return nn.GaussianNLLLoss(reduction='none') 
    raise ValueError(f"Unknown loss function: {loss_name}")

def compute_total_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], model_cfg: Dict[str, Any], exercise_id: str) -> torch.Tensor:
    """
    Computes the weighted sum of multi-task losses (L_total = Σ (wᵢ * Lᵢ)).
    Handles label masking and NLL loss (Gaussian).
    """
    global DEBUG_PRINTED
    if not DEBUG_PRINTED:
        print("\n--- Targets Content Debug (Printed Once) ---")
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                print(f"Key: {k:15} | Type: Tensor | Shape: {v.shape}")
            elif isinstance(v, list):
                print(f"Key: {k:15} | Type: List ({len(v)})")
            else:
                print(f"Key: {k:15} | Type: {type(v).__name__}")
        print("------------------------------------------\n")
        DEBUG_PRINTED = True 
    # ----------------------------------------
    
    total_loss = torch.tensor(0.0, device=outputs['class_logits'].device)
    
    head_outputs = model_cfg['model']['heads'][exercise_id]['outputs']

    # --- 1. Classification Loss ---
    target_class_key = 'target_class'
    if target_class_key in targets and targets[target_class_key] is not None: 
        
        target = targets[target_class_key]
        loss_name = head_outputs['classification']['loss']

        if loss_name == "BCEWithLogitsLoss":
            target = target.float()
            if target.dim() == 1:
                target = target.unsqueeze(1)
                
        elif loss_name == "CrossEntropyLoss":
            # สำหรับ CEL (Multi-class): ต้องการ [B]
            target = target.squeeze().long() 
            if target.dim() == 0: 
                target = target.unsqueeze(0)
            
        if target.numel() > 0 and 'class_logits' in outputs:
            weight = head_outputs['classification']['weight']
            loss_fn = get_loss_function(loss_name)
            
            logits = outputs['class_logits']
            
            # บังคับให้ Logits มีมิติ [B, C]
            if loss_fn.__class__ is nn.BCEWithLogitsLoss and logits.dim() == 1:
                 logits = logits.unsqueeze(1)
            
            # ตอนนี้ logits คือ [B, C] และ target คือ [B] หรือ [B, 1] (ตาม Loss)
            loss = loss_fn(logits, target)
            total_loss += weight * loss
        
    # --- 2. Angle Regression Loss (Gaussian NLL Loss) ---
    if 'angle_mean' in outputs and 'target_angles' in targets:
        weight = head_outputs['regression_angles']['weight']
        loss_fn = get_loss_function(head_outputs['regression_angles']['loss'])
        
        # targets['target_angles'] contains NaNs from dataset.py for masked values
        target = targets['target_angles']
        
        # Create mask for valid (non-NaN) targets
        valid_mask = ~torch.isnan(target)
        
        # Calculate loss per element
        # mean, logvar shape: (B, N_angles)
        loss_per_element = loss_fn(outputs['angle_mean'], target, torch.exp(outputs['angle_logvar']))
        
        # Apply mask and compute mean loss
        masked_loss = loss_per_element * valid_mask.float()
        
        # Normalize by the number of valid (non-masked) elements
        valid_elements_count = valid_mask.sum().float()
        if valid_elements_count > 0:
            loss = masked_loss.sum() / valid_elements_count
            total_loss += weight * loss
            
    # --- 3. Positional Regression Loss ---
    if 'pos_pred' in outputs and 'target_pos' in targets:
        weight = head_outputs['regression_pos']['weight']
        loss_fn = get_loss_function(head_outputs['regression_pos']['loss'])
        loss = loss_fn(outputs['pos_pred'], targets['target_pos'])
        total_loss += weight * loss

    return total_loss

def evaluate_model(model: MultiTaskModel, val_loader: DataLoader, exercise_id: str, device: torch.device, model_cfg: Dict) -> Dict[str, float]:
    """Runs evaluation on the validation set and returns metrics."""
    model.eval()
    
    all_targets = {'class': [], 'angle': []}
    all_predictions = {'class': [], 'angle_mean': []}
    
    head_cfg = model_cfg['model']['heads'][exercise_id]
    num_classes = head_cfg['outputs']['classification'].get('num_classes', 2)
    is_multiclass = num_classes > 2
    
    with torch.no_grad():
        for data_batch, targets_batch in val_loader:
            data_batch = data_batch.to(device)
            if data_batch.numel() == 0:
                continue 
                
            outputs = model(data_batch, exercise_id)

            # 1. Classification Metrics
            if 'class_logits' in outputs and targets_batch.get('target_class') is not None:
                
                # Targets ต้องถูก Squeeze ให้เป็น 1D [B] ก่อนส่งไป CPU
                targets_cpu = targets_batch['target_class'].detach().cpu().numpy().squeeze() 
                
                # Logits: Multi-class [B, C] หรือ Binary [B, 1]
                logits = outputs['class_logits'].detach().cpu().numpy()
                
                if is_multiclass:
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                else:
                    # Probabilities: Logits [B, 1] -> Sigmoid -> [B]
                    probs = 1.0 / (1.0 + np.exp(-logits)).flatten() 
                
                # Targets CPU ต้องเป็น 1D [B] สำหรับ Scikit-learn
                if targets_cpu.ndim == 0:
                    targets_cpu = np.array([targets_cpu])
                
                all_targets['class'].extend(targets_cpu)
                all_predictions['class'].extend(probs) 
                
            # 2. Angle Metrics (Logic เดิม แต่แก้ไขการ extend ให้ flatten)
            if 'angle_mean' in outputs and targets_batch.get('target_angles') is not None:
                mean = outputs['angle_mean'].detach().cpu().numpy()
                target_angles = targets_batch['target_angles'].detach().cpu().numpy()
                
                # Mask out NaNs
                valid_mask = ~np.isnan(target_angles)
                valid_preds = mean[valid_mask]
                valid_targets = target_angles[valid_mask]
                
                all_predictions['angle_mean'].extend(valid_preds)
                all_targets['angle'].extend(valid_targets)
                
    metrics = {}
    
    # --- Compute Classification Metrics ---
    if all_targets['class']:
        targets_np = np.array(all_targets['class'])
        
        if is_multiclass:
            # Multi-class Metrics: Targets เป็น Class Index (0, 1, 2...)
            probs_np = np.array(all_predictions['class']).reshape(-1, num_classes)
            preds_class = np.argmax(probs_np, axis=1)
            
            metrics['val_acc'] = accuracy_score(targets_np, preds_class)
            metrics['val_auc'] = 0.5 # AUC ไม่ใช้โดยตรง
            metrics['val_f1'] = f1_score(targets_np, preds_class, average='weighted', zero_division=0)
            
        else: # Binary Metrics
            probs_np = np.array(all_predictions['class']).flatten()
            preds_binary = (probs_np > 0.5).astype(int)
            
            metrics['val_acc'] = accuracy_score(targets_np, preds_binary)
            
            if len(np.unique(targets_np)) > 1:
                metrics['val_auc'] = roc_auc_score(targets_np, probs_np)
            else:
                metrics['val_auc'] = 0.5 # ไม่สามารถคำนวณ AUC ได้ (ตั้งค่ากลาง)

            metrics['val_f1'] = f1_score(targets_np, preds_binary, zero_division=0)
            
    # --- Compute Angle Metrics (MAE) ---
    if all_targets['angle']:
        targets_angle_np = np.array(all_targets['angle'])
        preds_angle_np = np.array(all_predictions['angle_mean'])
        
        if targets_angle_np.size > 0:
            metrics['val_mae_angle'] = np.mean(np.abs(preds_angle_np - targets_angle_np))
        else:
            metrics['val_mae_angle'] = 9999.0 
            
    model.train()
    return metrics


def save_checkpoint(model: MultiTaskModel, epoch: int, is_best: bool, exercise_id: str):
    """Saves separate checkpoints for backbone and individual heads."""
    base_path = CHECKPOINT_DIR / exercise_id
    base_path.mkdir(parents=True, exist_ok=True)
    
    tag = 'best' if is_best else f'epoch{epoch:03d}'
    
    # 1. Save Backbone
    backbone_path = base_path / f"backbone_{tag}.pth"
    torch.save(model.backbone.state_dict(), backbone_path)
    
    # 2. Save Heads individually
    heads_dir = base_path / 'heads'
    heads_dir.mkdir(exist_ok=True)
    for name, head in model.heads.items():
        head_path = heads_dir / f"{name}_{tag}.pth"
        torch.save(head.state_dict(), head_path)
        
    print(f"-> Checkpoint saved for epoch {epoch} (tag: {tag})")
    
def train_model(exercise_id: str):
    """Main function to run the training pipeline."""
    configs = load_config()
    model_cfg = configs['model_config']
    
    # 1. Setup Model and Optimizer
    model = MultiTaskModel(model_cfg, configs['exercises'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg['training']['optimizer']['lr']))

    # 2. Setup DataLoaders (Note: Use Collate Function for windowing)
    train_dataset = MultiTaskPoseDataset(exercise_id, split='train')
    val_dataset = MultiTaskPoseDataset(exercise_id, split='val') 
    
    val_uuids = val_dataset.metadata['ingest_uuid'].unique()
    print(f"Validation UUIDs: {list(val_uuids)}")
    
    train_loader = DataLoader(train_dataset, batch_size=model_cfg['training']['batch_size'], shuffle=True, collate_fn=window_collate_fn) 
    val_loader = DataLoader(val_dataset, batch_size=model_cfg['training']['batch_size'], shuffle=False, collate_fn=window_collate_fn) 
    
    device = torch.device(model_cfg['inference']['device'])
    model.to(device)
    
    best_val_metric = float('inf') 
    
    # 3. Training Loop  
    for epoch in range(model_cfg['training']['epochs']):
        # --- TRAINING PHASE ---
        model.train()
        
        for data_batch, targets_batch in tqdm(train_loader):
            # 💡 NOTE: Data batch must be windowed (B, T, V, C) by the collate function
            data_batch = data_batch.to(device) 
            
            optimizer.zero_grad()
            
            # 4. Forward Pass
            outputs = model(data_batch, exercise_id)
            
            # 5. Compute Loss
            batch_total_loss = 0.0
            batch_size = data_batch.size(0)
            
            for i in range(batch_size):
                single_targets = {}
                single_outputs = {}
                
                # สกัด Targets
                for k, v in targets_batch.items():
                    if k in ['ingest_uuid', 'exercise_id']:
                        single_targets[k] = v[i] # Metadata ไม่ใช่ Tensor
                    elif v is not None and isinstance(v, torch.Tensor):
                        target_tensor = v[i].unsqueeze(0)
                        single_targets[k] = target_tensor.to(device)
                    else:
                        single_targets[k] = v

                # สกัด Outputs (Logic เดิม)
                for k, v in outputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        single_outputs[k] = v[i].unsqueeze(0)
                
                if single_targets.get('target_angles') is None and single_targets.get('target_pos') is None and single_targets.get('target_class') is None:
                    continue
                
                loss = compute_total_loss(single_outputs, single_targets, model_cfg, exercise_id)
                batch_total_loss += loss

            avg_batch_loss = batch_total_loss / batch_size
            
            # 6. Backward Pass
            avg_batch_loss.backward()
            optimizer.step()
        
        val_labels = []
        for _, targets_batch in val_loader:
            if targets_batch.get('target_class') is not None:
                val_labels.extend(targets_batch['target_class'].cpu().numpy().flatten().tolist())
        
        if val_labels:
            unique, counts = np.unique(val_labels, return_counts=True)
            print(f"\nVALIDATION CLASS BALANCE: {dict(zip(unique, counts))}")
        # ----------------------------------------
        
        # --- VALIDATION PHASE ---
        val_metrics = evaluate_model(model, val_loader, exercise_id, device, model_cfg)
        
        print(f"Epoch {epoch+1} Results: Train Loss={avg_batch_loss.item():.4f}, Val MAE={val_metrics.get('val_mae_angle', np.inf):.4f}, Val AUC={val_metrics.get('val_auc', 0.0):.4f}")

        # --- CHECKPOINTING ---
        current_val_metric = val_metrics.get('val_mae_angle', float('inf'))
        
        #ถ้า MAE เป็น inf ให้ใช้ (1 - Val_ACC) เป็น Metric
        if current_val_metric >= float('inf') or current_val_metric == 9999.0:
             current_val_metric = 1.0 - val_metrics.get('val_acc', 0.0)
        
        is_best = current_val_metric < best_val_metric
        
        if is_best:
            best_val_metric = current_val_metric
            save_checkpoint(model, epoch, is_best=True, exercise_id=exercise_id)
            
        save_checkpoint(model, epoch, is_best=False, exercise_id=exercise_id)
        
    print(f"\nTraining of {exercise_id} complete. Best Val MAE: {best_val_metric:.4f}")

if __name__ == '__main__':
    train_model('Jump_squats') 
    train_model('Stretching_upper_trapezius')