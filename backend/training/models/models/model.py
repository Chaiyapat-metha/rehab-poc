# File: .\backend\training\models\models\model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

class TCNBlock(nn.Module):
    """
    TCN Block with Residual Connection and Dilated Convolution.
    The input/output is transposed to be (Batch, Channel, Time) for Conv1D.
    """
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout=0.0):
        super().__init__()
        
        # Padding สำหรับ Dilated Causal Convolution: 
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual Connection (Identity Mapping)
        self.residual = nn.Identity()
        if input_size != output_size:
            self.residual = nn.Conv1d(input_size, output_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x shape: (B, C, T)
        # 1. Save input for residual
        residual = self.residual(x)
        
        # 2. Convolution path
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 3. Add residual connection
        return self.relu(out + residual)


class TCN_GRU_Backbone(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__()
        backbone_cfg = model_cfg['backbone']
        tcn_cfg = backbone_cfg['tcn']
        gru_cfg = backbone_cfg['gru']
        
        input_dim = 33 * 3 # V*C
        embed_dim = gru_cfg['hidden'] // 2 if gru_cfg['bidirectional'] else gru_cfg['hidden']
        
        # 1. Pre-processing: Flatten (B,T,V,C) -> (B,T,V*C) -> Linear (V*C -> embed_dim)
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 2. TCN Block (1 block, dilation 2)
        # Note: TCN input is (B, C, T), output is (B, C, T). Need permute/transpose.
        self.tcn = TCNBlock(embed_dim, embed_dim, tcn_cfg['kernel_size'], tcn_cfg['dilation_levels'][0]) 
        
        # 3. GRU Layer (1 layer)
        self.gru = nn.GRU(
            embed_dim, 
            gru_cfg['hidden'], 
            num_layers=gru_cfg['layers'], 
            batch_first=True,
            bidirectional=gru_cfg['bidirectional']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x shape: (B, T, V, C)
        B, T, V, C = x.shape
        x = x.view(B, T, -1) # Flatten to (B, T, V*C)

        x = self.embedding(x) # (B, T, embed_dim)
        
        # TCN requires (B, C, T)
        x_tcn = x.permute(0, 2, 1) # (B, embed_dim, T)
        x_tcn = self.tcn(x_tcn) 
        x_gru_in = x_tcn.permute(0, 2, 1) # (B, T, embed_dim)

        output, _ = self.gru(x_gru_in) # output: (B, T, H)
        final_feature = output[:, -1, :] # Final feature: (B, H)
        return final_feature


class Head(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, loss_type: str, use_logvar: bool = False):
        super().__init__()
        self.loss_type = loss_type
        self.use_logvar = use_logvar
        
        output_factor = 2 if use_logvar else 1
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * output_factor)
        )
        
    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.mlp(feature)
        
        if self.use_logvar:
            # Predict mean and log variance
            mean, logvar = torch.chunk(output, 2, dim=-1)
            return mean, logvar
        else:
            return output, None

class MultiTaskModel(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], exercises_cfg: Dict[str, Any]):
        super().__init__()
        self.backbone = TCN_GRU_Backbone(model_cfg['model']) 
        self.heads = nn.ModuleDict()
        self.exercise_heads_map = {}
        
        feature_dim = model_cfg['model']['backbone']['gru']['hidden']
        
        for exercise_id, head_cfg in model_cfg['model']['heads'].items():
            outputs = head_cfg['outputs']
            
            # --- 1. Classification Head ---
            if 'classification' in outputs:
                num_classes = outputs['classification']['num_classes']
                head_name = f'{exercise_id}_class'
                self.heads[head_name] = Head(feature_dim, 1 if num_classes == 2 else num_classes, outputs['classification']['loss'])
                self.exercise_heads_map[f'{exercise_id}_class'] = head_name
                
            # --- 2. Angle Regression Head ---
            if 'regression_angles' in outputs:
                # Output dim = N_angles from exercises.yaml
                n_angles = len(exercises_cfg.get(exercise_id, {}).get('angle_output_order', []))
                head_name = f'{exercise_id}_angle'
                # Angle head uses GaussianNLLLoss, so it must predict logvar (uncertainty)
                self.heads[head_name] = Head(feature_dim, n_angles, outputs['regression_angles']['loss'], use_logvar=True)
                self.exercise_heads_map[f'{exercise_id}_angle'] = head_name

            # --- 3. Positional Regression Head ---
            if 'regression_pos' in outputs:
                output_dim = outputs['regression_pos']['output_dim'] # Should be 99
                head_name = f'{exercise_id}_pos'
                self.heads[head_name] = Head(feature_dim, output_dim, outputs['regression_pos']['loss'])
                self.exercise_heads_map[f'{exercise_id}_pos'] = head_name

    def forward(self, x: torch.Tensor, exercise_id: str) -> Dict[str, torch.Tensor]:
        """
        Runs the input through the backbone and the relevant head for the given exercise.
        """
        shared_feature = self.backbone(x)
        
        output = {}
        
        # Run Classification Head
        if f'{exercise_id}_class' in self.exercise_heads_map:
            head_name = self.exercise_heads_map[f'{exercise_id}_class']
            logits, _ = self.heads[head_name](shared_feature)
            output['class_logits'] = logits.squeeze(1) # (B,)
            
        # Run Angle Regression Head
        if f'{exercise_id}_angle' in self.exercise_heads_map:
            head_name = self.exercise_heads_map[f'{exercise_id}_angle']
            mean, logvar = self.heads[head_name](shared_feature)
            output['angle_mean'] = mean
            output['angle_logvar'] = logvar
            
        # Run Positional Regression Head
        if f'{exercise_id}_pos' in self.exercise_heads_map:
            head_name = self.exercise_heads_map[f'{exercise_id}_pos']
            pos_pred, _ = self.heads[head_name](shared_feature)
            output['pos_pred'] = pos_pred
            
        return output