import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

# Setup path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.data_loader import RehabDataset
from training.models.tcn_gru_multitask import TCN_GRU_MultiTask
from training.losses import MultiTaskLoss

def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    train_params = config['training_params']
    model_params = config['model_params']
    output_params = config['output']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset and DataLoader
    print("Loading dataset...")
    dataset = RehabDataset(
        window_size=model_params['window_size'],
        step=train_params['step'],
        supervised=True # IMPORTANT
    )
    data_loader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True)

    # Model, Loss, Optimizer
    model = TCN_GRU_MultiTask(
        feature_dim=model_params['feature_dim'],
        num_classes=model_params['num_classes']
    ).to(device)
    
    criterion = MultiTaskLoss(weights=train_params['loss_weights'])
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    print("Starting supervised multi-task training...")
    for epoch in range(train_params['epochs']):
        model.train()
        epoch_losses = {"total_loss": 0, "anomaly_loss": 0, "class_loss": 0, "reg_loss": 0}
        
        for windows, labels in data_loader:
            windows = windows.to(device)
            # Move all label tensors to the device
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            predictions = model(windows)
            
            total_loss, loss_dict = criterion(predictions, labels)
            total_loss.backward()
            optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{train_params['epochs']}] | "
              f"Total Loss: {epoch_losses['total_loss'] / len(data_loader):.4f} | "
              f"Class Loss: {epoch_losses['class_loss'] / len(data_loader):.4f}")

    # Save the model
    output_dir = Path(output_params['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / output_params['model_name']
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN+GRU Multi-task model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config YAML file.")
    args = parser.parse_args()
    train(args.config)
