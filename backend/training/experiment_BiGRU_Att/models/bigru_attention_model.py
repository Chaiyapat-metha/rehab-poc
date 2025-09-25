# rehab-poc/backend/training/experiment_BiGRU_Att/models/bigru_attention_model.py

import torch
import torch.nn as nn
import math

class MultiheadAttentionBlock(nn.Module):
    """Temporal Multihead Self-Attention applied to GRU outputs."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear layers for Query, Key, Value
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        # x shape: (B, T, D_model) where D_model is 512
        B, T, D_model = x.shape
        
        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, H_dim)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Weighted sum of values
        context = torch.matmul(attention_weights, V) # (B, H, T, H_dim)
        
        # Concatenate heads and pass through final linear layer
        context = context.transpose(1, 2).contiguous().view(B, T, D_model)
        
        output = self.output_linear(context)
        
        return output

class BiGRUAttentionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        input_dim = config['dataset']['joints'] * config['dataset']['input_dim'] # 99
        embed_dim = config['model']['frame_embed_dim'] # 128
        
        # 1. Frame Embedding (MLP)
        self.frame_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            # Small Residual Block (Optional, simplified to standard MLP for now)
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. BiGRU
        gru_cfg = config['model']['bigru']
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_cfg['hidden'], # 256
            num_layers=gru_cfg['layers'], # 2
            batch_first=True,
            bidirectional=True,
            dropout=gru_cfg['dropout']
        )
        
        # BiGRU output dimension is 2 * hidden_size = 512
        bigru_out_dim = gru_cfg['hidden'] * 2 
        
        # 3. Multihead Self-Attention
        att_cfg = config['model']['attention']
        self.attention = MultiheadAttentionBlock(
            d_model=bigru_out_dim, # 512
            num_heads=att_cfg['heads'] # 4
        )
        
        # --- 4. Output Heads ---
        # The heads operate on the attended feature vector from the last time step
        att_out_dim = bigru_out_dim # 512
        
        # Head 1: Delta Angle Regression (Main output)
        ang_cfg = config['heads']['delta_angle_regression']
        self.delta_angle_head = nn.Sequential(
            nn.Linear(att_out_dim, ang_cfg['hidden']),
            nn.ReLU(),
            nn.Linear(ang_cfg['hidden'], ang_cfg['output_dim']) # 6 Radians
        )
        
        # Head 2: Log Variance (Uncertainty)
        unc_cfg = config['heads']['log_variance_output']
        self.log_variance_head = nn.Sequential(
            nn.Linear(att_out_dim, unc_cfg['hidden']),
            nn.ReLU(),
            nn.Linear(unc_cfg['hidden'], unc_cfg['output_dim']) # 6 Log Variance
        )

    def forward(self, x: torch.Tensor):
        # x shape: (B, T, J, 3) -> (B, 32, 33, 3)
        B, T, J, C = x.shape
        
        # 1. Frame Embedding
        x = x.view(B, T, J * C) # Flatten J*C -> 99
        x = self.frame_embed(x) # Output: (B, T, 128)
        
        # 2. BiGRU
        # BiGRU does not require permutation (batch_first=True)
        gru_output, _ = self.bigru(x) # Output: (B, T, 512)
        
        # 3. Attention
        att_output = self.attention(gru_output) # Output: (B, T, 512)
        
        # 4. Global Pooling / Final Time Step (We use the last time step T-1)
        final_feature = att_output[:, -1, :] # Output: (B, 512)
        
        # 5. Output Heads
        delta_angle_pred = self.delta_angle_head(final_feature)
        log_var_pred = self.log_variance_head(final_feature)
        
        return {
            'delta_theta_output': delta_angle_pred,
            'log_variance_output': log_var_pred
        }