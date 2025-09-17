import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

# Import our custom modules
from .data_loader import RehabDataset
from .models.tcn_autoencoder import TCNAutoencoder

def train(config_path: str):
    # --- 1. Load Config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_params = config['training_params']
    model_params = config['model_params']
    output_params = config['output']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_params['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Prepare Dataset and DataLoader ---
    print("Loading dataset...")
    dataset = RehabDataset(
        window_size=model_params['window_size'], 
        step=config['data']['step']
    )
    if len(dataset) == 0:
        print("❌ Dataset is empty. Please run the video ingestion script first.")
        return
        
    train_loader = DataLoader(
        dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=True
    )

    # --- 3. Create Model, Loss function, Optimizer ---
    print("Initializing model...")
    model = TCNAutoencoder(
        feature_dim=model_params['feature_dim'],
        latent_dim=model_params['latent_dim'],
        window_size=model_params['window_size']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    
    print("🚀 Starting training...")
    # --- 4. Training Loop ---
    for epoch in range(train_params['epochs']):
        model.train()
        running_loss = 0.0
        
        # Wrap loader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_params['epochs']}")
        
        for data in progress_bar:
            inputs = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs) # Reconstruction loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{train_params['epochs']}], Average Loss: {epoch_loss:.6f}")

    # --- 5. Save the trained model (PyTorch format) ---
    model_path_pth = output_dir / output_params['model_name_pth']
    torch.save(model.state_dict(), model_path_pth)
    print(f"✅ Model saved to {model_path_pth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN Autoencoder for Anomaly Detection.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config YAML file.")
    args = parser.parse_args()
    train(args.config)
