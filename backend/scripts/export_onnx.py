# File: .\backend\scripts\export_onnx.py

import torch
import torch.nn as nn
import numpy as np
import onnx
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
    
    # Import Utilities and Model Components (Assuming absolute imports work from project root)
from backend.app.config import load_config
from backend.training.models.models.model import TCN_GRU_Backbone, Head 

# --- Configuration Paths ---
WEIGHTS_DIR = Path("./backend/weights")
CHECKPOINT_DIR = Path("./backend/training_outputs/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants for ONNX Input Shape
# NOTE: WINDOW_SIZE=16 (or 32, based on your dataset.py)
WINDOW_SIZE = 16 
V_JOINTS = 33
C_COORDS = 3
INPUT_SHAPE = (1, WINDOW_SIZE, V_JOINTS, C_COORDS) # B=1 (Dynamic), T=16, V=33, C=3

def export_to_onnx(model: nn.Module, model_path: Path, input_shape: Tuple[int, ...], input_names: List[str], output_names: List[str]):
    """Helper function to perform ONNX export with dynamic batch size."""
    # สร้าง Dummy Input Tensor
    dummy_input = torch.randn(input_shape)
    
    # Export ด้วย PyTorch
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=17, # Opset version ล่าสุดที่เสถียร
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            # กำหนดให้ Batch Size (มิติที่ 0) เป็น Dynamic
            input_names[0]: {0: 'batch_size'} 
        }
    )
    print(f"✅ Successfully exported to {model_path}")


def export_backbone(model_cfg: Dict, exercise_id: str):
    """Exports the shared TCN+GRU backbone."""
    print("--- Exporting Backbone (TCN+GRU) ---")
    
    # 1. สร้าง Model Instance
    backbone = TCN_GRU_Backbone(model_cfg['model'])
    
    # 2. โหลด Checkpoint Weight ที่ดีที่สุด
    backbone_best_path = CHECKPOINT_DIR / exercise_id / "backbone_best.pth"
    if not backbone_best_path.exists():
        print(f"❌ Error: Best backbone weight not found at {backbone_best_path}. Please run training first.")
        return
        
    backbone.load_state_dict(torch.load(backbone_best_path))
    backbone.eval()
    
    # 3. Export
    export_to_onnx(
        backbone,
        WEIGHTS_DIR / "backbone.onnx", # ชื่อไฟล์ที่ใช้ร่วมกัน
        INPUT_SHAPE,
        input_names=['input_data'],
        output_names=['shared_feature']
    )


def export_heads(model_cfg: Dict, exercises_cfg: Dict, exercise_id: str):
    """Exports individual heads for a specific exercise."""
    print(f"\n--- Exporting Modular Heads for {exercise_id} ---")
    
    heads_cfg = model_cfg['model']['heads'][exercise_id]['outputs']
    feature_dim = model_cfg['model']['backbone']['gru']['hidden']
    
    head_best_dir = CHECKPOINT_DIR / exercise_id / 'heads'
    if not head_best_dir.exists():
        print(f"❌ Error: Head weights directory not found for {exercise_id}. Cannot export heads.")
        return

    # Input Shape สำหรับ Heads คือ Shared Feature [B, FEATURE_DIM]
    HEAD_INPUT_SHAPE = (1, feature_dim) 

    # --- 1. Classification Head (e.g., jump_squats_class.onnx) ---
    if 'classification' in heads_cfg:
        head_name = f'{exercise_id}_class'
        num_classes = heads_cfg['classification']['num_classes']
        # Output dim is 1 for Binary (BCEWithLogitsLoss) or num_classes for Multi-class (CrossEntropyLoss)
        output_dim = 1 if num_classes == 2 else num_classes 
        head = Head(feature_dim, output_dim, heads_cfg['classification']['loss'])
        
        head_path = head_best_dir / f"{head_name}_best.pth"
        if head_path.exists():
            head.load_state_dict(torch.load(head_path))
            head.eval()
            
            export_to_onnx(
                head,
                WEIGHTS_DIR / f"{head_name}.onnx",
                HEAD_INPUT_SHAPE,
                input_names=['shared_feature'],
                output_names=['class_logit_output']
            )

    # --- 2. Angle Regression Head (e.g., jump_squats_angle.onnx) ---
    if 'regression_angles' in heads_cfg:
        head_name = f'{exercise_id}_angle'
        n_angles = len(exercises_cfg.get(exercise_id, {}).get('angle_output_order', []))
        # Head uses GaussianNLLLoss, so it predicts mean and logvar (use_logvar=True)
        head = Head(feature_dim, n_angles, heads_cfg['regression_angles']['loss'], use_logvar=True)

        head_path = head_best_dir / f"{head_name}_best.pth"
        if head_path.exists():
            head.load_state_dict(torch.load(head_path))
            head.eval()
            
            export_to_onnx(
                head,
                WEIGHTS_DIR / f"{head_name}.onnx",
                HEAD_INPUT_SHAPE,
                input_names=['shared_feature'],
                output_names=['angle_mean_output', 'angle_logvar_output'] # Output 2 tensors
            )

    # --- 3. Positional Regression Head (e.g., jump_squats_pos.onnx) ---
    if 'regression_pos' in heads_cfg:
        head_name = f'{exercise_id}_pos'
        output_dim = heads_cfg['regression_pos']['output_dim'] # 99 dims
        head = Head(feature_dim, output_dim, heads_cfg['regression_pos']['loss'])

        head_path = head_best_dir / f"{head_name}_best.pth"
        if head_path.exists():
            head.load_state_dict(torch.load(head_path))
            head.eval()
            
            export_to_onnx(
                head,
                WEIGHTS_DIR / f"{head_name}.onnx",
                HEAD_INPUT_SHAPE,
                input_names=['shared_feature'],
                output_names=['pos_output']
            )
        

def main(exercise_id_to_export: str):
    """Main export function."""
    configs = load_config()
    model_cfg = configs['model_config']
    exercises_cfg = configs['exercises']

    # 1. Export Backbone (ใช้ร่วมกันทุกท่า)
    export_backbone(model_cfg, exercise_id_to_export)
    
    # 2. Export Heads (สำหรับท่าที่ระบุ)
    export_heads(model_cfg, exercises_cfg, exercise_id_to_export)
    
    print("\nONNX Export pipeline finished. Weights are ready for FastAPI inference.")

if __name__ == '__main__':
    # 💡 IMPORTANT: เปลี่ยน Exercise ID ให้เป็นท่าที่คุณเทรนเสร็จแล้ว
    # เช่น 'Jump_squats' หรือ 'Stretching_upper_trapezius'
    main('Stretching_upper_trapezius')