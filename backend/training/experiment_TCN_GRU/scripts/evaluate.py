# scripts/evaluate.py

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, data_loader, device, training_mode=True):
    """
    Evaluates the model on a given data loader and returns the average loss and metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)

            # The loss calculation is for 0-1 normalized angle differences
            loss = nn.BCEWithLogitsLoss()(output, target)
            total_loss += loss.item()
            
            # Convert logits to binary predictions
            preds = (torch.sigmoid(output) > 0.5).long().cpu().numpy()
            targets = target.long().cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)

    # Calculate overall metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='samples', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    avg_loss = total_loss / len(data_loader)
    
    if training_mode:
        return avg_loss, metrics
    else:
        return avg_loss, metrics