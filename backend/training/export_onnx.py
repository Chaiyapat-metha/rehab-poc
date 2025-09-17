import torch
import argparse
import yaml
from pathlib import Path

# Setup path to import from app and other training modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the new supervised model architecture
from training.models.tcn_gru_multitask import TCN_GRU_MultiTask

def export_supervised_model(config_path: str):
    """
    Loads a trained TCN+GRU MultiTask model and exports it to the ONNX format.
    """
    print("--- Starting Supervised Model Export to ONNX ---")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    
    model_params = config['model_params']
    output_params = config['output']
    
    device = "cpu" # Exporting is usually done on CPU
    print(f"Using device: {device}")

    # 1. Instantiate the model architecture
    print("Instantiating TCN_GRU_MultiTask model...")
    model = TCN_GRU_MultiTask(
        feature_dim=model_params['feature_dim'],
        num_classes=model_params['num_classes']
    ).to(device)
    
    # 2. Load the trained weights (.pth file)
    trained_model_path = Path(output_params['dir']) / output_params['model_name']
    if not trained_model_path.exists():
        print(f"❌ Error: Trained model file not found at {trained_model_path}")
        print("Please run the training script first.")
        return
        
    print(f"Loading trained weights from: {trained_model_path}")
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval() # Set the model to evaluation mode

    # 3. Create a dummy input tensor with the correct shape
    batch_size = 1 # We export for single-item inference
    window_size = model_params['window_size']
    feature_dim = model_params['feature_dim']
    dummy_input = torch.randn(batch_size, window_size, feature_dim, device=device)
    print(f"Creating dummy input with shape: {dummy_input.shape}")

    # 4. Define the output path for the ONNX model
    onnx_output_dir = Path("weights")
    onnx_output_dir.mkdir(parents=True, exist_ok=True)
    onnx_model_name = Path(output_params['model_name']).stem + ".onnx"
    onnx_path = onnx_output_dir / onnx_model_name
    print(f"Exporting model to: {onnx_path}")

    # 5. Export the model
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['anomaly', 'classification', 'regression'], # Naming the outputs
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable batch size
                          'anomaly' : {0 : 'batch_size'},
                          'classification' : {0 : 'batch_size'},
                          'regression' : {0 : 'batch_size'}}
        )
        print("✅ Model exported successfully to ONNX format!")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained PyTorch model to ONNX.")
    # We reuse the supervised config file as it contains all necessary model parameters
    parser.add_argument("--config", type=str, required=True, 
                        default="training_configs/supervised_config.yaml",
                        help="Path to the supervised training config YAML file.")
    args = parser.parse_args()
    export_supervised_model(args.config)


# **หลังจากเทรนเสร็จ ให้คุณรันคำสั่งนี้เพื่อ Export:**
# ```bash
# python -m training.export_onnx --config training_configs/supervised_config.yaml
