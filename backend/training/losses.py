import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, weights={'anomaly': 1.0, 'class': 1.0, 'reg': 1.0}):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights
        # Binary Cross-Entropy with Logits for anomaly (more stable)
        self.anomaly_loss = nn.BCEWithLogitsLoss()
        # Cross-Entropy for classification
        self.class_loss = nn.CrossEntropyLoss()
        # Mean Squared Error for regression
        self.reg_loss = nn.MSELoss()

    def forward(self, predictions, labels):
        # Anomaly Loss
        # Label is 1 if class is not CORRECT (0), else 0
        anomaly_labels = (labels['class'] != 0).float().unsqueeze(1)
        loss_anomaly = self.anomaly_loss(predictions['anomaly'], anomaly_labels)

        # Classification Loss
        loss_class = self.class_loss(predictions['classification'], labels['class'])
        
        # Regression Loss
        # We only calculate regression loss for incorrect poses
        # Create a mask for samples that are not "CORRECT"
        incorrect_mask = (labels['class'] != 0)
        if incorrect_mask.any():
            # Apply mask to predictions and labels before calculating loss
            loss_reg = self.reg_loss(
                predictions['regression'][incorrect_mask], 
                labels['regression'][incorrect_mask]
            )
        else:
            # If no incorrect samples in batch, regression loss is 0
            loss_reg = torch.tensor(0.0, device=predictions['regression'].device)

        # Weighted total loss
        total_loss = (self.weights['anomaly'] * loss_anomaly +
                      self.weights['class'] * loss_class +
                      self.weights['reg'] * loss_reg)
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "anomaly_loss": loss_anomaly.item(),
            "class_loss": loss_class.item(),
            "reg_loss": loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg
        }
