# rehab-poc/backend/training/experiment_candidates/models/model_archs.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import math

# --- Graph Utilities for ST-GCN ---
def get_adjacency_matrix(num_joints: int):
    """Creates a basic adjacency matrix based on the MediaPipe structure."""
    # This is a simplified version; a full GCN requires a complete 33-joint graph.
    edges = [
        (24, 23), (24, 26), (23, 25), (26, 28), (25, 27), # Hips/Legs
        (24, 12), (23, 11), (12, 14), (11, 13), (14, 16), (13, 15) # Torso/Arms
    ]
    A = np.zeros((num_joints, num_joints))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    A_normalized = A + np.eye(num_joints)
    return torch.tensor(A_normalized, dtype=torch.float32)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t, stride, A):
        super().__init__()
        # Graph Convolution (Spatial)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        # Temporal Convolution
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_t, 1), 
                             padding=((kernel_size_t - 1) // 2, 0), stride=(stride, 1))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = A
        
    def forward(self, x):
        # x shape: (N, C, T, V) where V=Joints
        
        # 1. Spatial GCN (GCN=x * A)
        x_gcn = torch.einsum('nctv, vk -> nctk', x, self.A.to(x.device))
        x_gcn = self.gcn(x_gcn)
        
        # 2. Temporal TCN
        x_tcn = self.tcn(x_gcn)
        
        return self.relu(self.bn(x_tcn))

# --- TCN Component for TCN+GRU ---
class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # Causal convolution logic is complex; using 'same' padding for simplicity and focus on dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + res

class MultiheadAttentionBlock(nn.Module):
    """Temporal Multihead Self-Attention applied to GRU outputs or STGCN features."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Ensures d_model is divisible by num_heads (a necessary check)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear layers for Query, Key, Value
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        # x shape: (B, T, D_model) 
        B, T, D_model = x.shape
        
        # 1. Linear Projection and Split Heads
        # Output shape: (B, H, T, H_dim)
        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 3. Apply Attention to Value
        context = torch.matmul(attention_weights, V) # (B, H, T, H_dim)
        
        # 4. Concatenate heads and final linear layer
        context = context.transpose(1, 2).contiguous().view(B, T, D_model)
        
        output = self.output_linear(context)
        
        return output
    
# --- Core Candidate Architectures ---

class BaseMultiHeadModel(nn.Module):
    """Base class for all models to standardize heads and output dictionary."""
    def __init__(self, feature_dim: int, heads_config: dict, full_config: dict):
        super().__init__()
        self.config = full_config 
        
        # All models use the same head structure on the final feature vector
        self.pos_head = nn.Linear(feature_dim, heads_config['pos_regression']['output_dim'])# 99
        self.angle_head = nn.Linear(feature_dim, heads_config['angle_regression']['output_dim']) # 6
        self.class_head = nn.Linear(feature_dim, heads_config['classification']['output_dim']) # 1

    def forward_heads(self, final_feature: torch.Tensor):
        return {
            'pos_output': self.pos_head(final_feature),
            'angle_output': self.angle_head(final_feature),
            'class_output': self.class_head(final_feature)
        }
    
# --- Implementation of Candidate 1: TCN+GRU (Baseline) ---
class TCNGRUModel(BaseMultiHeadModel):
    def __init__(self, arch_cfg, input_dim, heads_cfg, full_config):
        super().__init__(arch_cfg['gru']['hidden'], heads_cfg, full_config) # hidden=256
        
        self.input_embed = nn.Linear(input_dim, arch_cfg['gru']['hidden'])
        
        # TCN Layer (Fixed at 1 block, Dilation 1)
        self.tcn_block = TCNBlock(arch_cfg['gru']['hidden'], arch_cfg['gru']['hidden'], 
                                   arch_cfg['tcn']['kernel_size'], 1)
        gru_cfg = arch_cfg['gru']
        gru_dropout = gru_cfg.get('dropout', 0.0) 
        if gru_cfg['layers'] == 1:
            gru_dropout = 0.0 

        self.gru = nn.GRU(
            gru_cfg['hidden'],
            gru_cfg['hidden'],
            gru_cfg['layers'],
            batch_first=True,
            bidirectional=gru_cfg['bidirectional'],
            dropout=gru_dropout 
        )
        
    def forward(self, x: torch.Tensor):
        B, T, V, C = x.shape
        x = x.view(B, T, V * C) # (B, T, 99)
        
        x = self.input_embed(x) # (B, T, 256)
        
        # TCN (1 block)
        x = x.permute(0, 2, 1) # (B, 256, T)
        x = self.tcn_block(x)
        
        # GRU (No pooling)
        x = x.permute(0, 2, 1) # (B, T, 256)
        
        gru_output, _ = self.gru(x)
        final_feature = gru_output[:, -1, :] # Final time step (B, 256)
        
        return self.forward_heads(final_feature)


# --- Implementation of Candidate 2: BiGRU + Attention ---
class BiGRUAttentionModel(BaseMultiHeadModel):
    def __init__(self, arch_cfg, input_dim, heads_cfg, full_config):
        super().__init__(arch_cfg['gru']['hidden'] * 2, heads_cfg, full_config) # hidden=512
        
        self.input_embed = nn.Linear(input_dim, arch_cfg['gru']['hidden'])
        
        # BiGRU Layer (2 layers, Bidirectional)
        self.bigru = nn.GRU(arch_cfg['gru']['hidden'], arch_cfg['gru']['hidden'], arch_cfg['gru']['layers'], batch_first=True, bidirectional=arch_cfg['gru']['bidirectional'], dropout=arch_cfg['gru']['dropout'])
        
        # Multihead Attention
        bigru_out_dim = arch_cfg['gru']['hidden'] * 2
        self.attention = MultiheadAttentionBlock(bigru_out_dim, arch_cfg['attention']['heads'])

    def forward(self, x: torch.Tensor):
        B, T, V, C = x.shape
        x = x.view(B, T, V * C) # (B, T, 99)
        
        x = self.input_embed(x) # (B, T, 256)
        
        gru_output, _ = self.bigru(x) # (B, T, 512)
        
        att_output = self.attention(gru_output) # (B, T, 512)
        
        final_feature = att_output[:, -1, :] # Final time step (B, 512)
        
        return self.forward_heads(final_feature)

# --- Implementation of Candidate 3: ST-GCN + 3 Heads ---

# --- Utility Function to Build STGCN Network ---
def build_stgcn_network(arch_cfg: dict, heads_cfg: dict) -> nn.Sequential:
    """Builds the sequential ST-GCN layers (backbone only)."""
    num_joints = heads_cfg['pos_regression']['output_dim'] // 3
    A = get_adjacency_matrix(num_joints)
    
    stgcn_channels = arch_cfg['channels']
    kernel_size_t = arch_cfg['kernel_size_t']
    
    layers = []
    in_channels = 3 # Input is C=3, T, V
    for out_channels in stgcn_channels:
        layers.append(STGCNBlock(in_channels, out_channels, kernel_size_t, 1, A))
        in_channels = out_channels
    
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------------------
# --- Implementation of Candidate 3: ST-GCN + 3 Heads (BASELINE) ---
# ----------------------------------------------------------------------------------
class STGCNModel(BaseMultiHeadModel):
    def __init__(self, arch_cfg, input_dim, heads_cfg, full_config):
        # final_C = arch_cfg['channels'][-1] = 256
        super().__init__(arch_cfg['channels'][-1], heads_cfg, full_config) 

        # FIX 1: สร้าง Network Backbone โดยใช้ Utility Function
        self.stgcn_network = build_stgcn_network(arch_cfg, heads_cfg)

    def forward(self, x: torch.Tensor):
        B, T, V, C = x.shape
        
        # Preprocessing: (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.stgcn_network(x) # Output: (B, Final_C, T, V)
        
        # Global Temporal and Spatial Pooling
        final_feature = x.mean(dim=3).mean(dim=2) # Average over V and T: Output (B, Final_C)
        
        # FIX 2: ใช้ self.forward_heads
        return self.forward_heads(final_feature) 

# ----------------------------------------------------------------------------------
# --- Implementation of Candidate 4 (STGCN+Attention) ---
# ----------------------------------------------------------------------------------
class STGCNAttentionModel(BaseMultiHeadModel):
    def __init__(self, stgcn_cfg, bigru_att_cfg, input_dim, heads_cfg, full_config):
        super().__init__(stgcn_cfg['channels'][-1], heads_cfg, full_config) # 256
        
        # FIX: สร้าง Network Backbone โดยตรง
        self.stgcn_network = build_stgcn_network(stgcn_cfg, heads_cfg)
        self.attention = MultiheadAttentionBlock(stgcn_cfg['channels'][-1], bigru_att_cfg['attention']['heads'])

    def forward(self, x: torch.Tensor):
        B, T, V, C = x.shape
        
        # Pass through STGCN layers: (B, C_final, T, V)
        stgcn_features = self.stgcn_network(x.permute(0, 3, 1, 2).contiguous())
        
        # Apply temporal attention to global pooled features
        att_input = stgcn_features.mean(dim=3).permute(0, 2, 1) # (B, T, C_final)
        
        att_output = self.attention(att_input)
        final_feature = att_output.mean(dim=1) # Average over Time (B, C_final)
        
        return self.forward_heads(final_feature)

# ----------------------------------------------------------------------------------
# --- Implementation of Candidate 5 (STGCN-GRU) ---
# ----------------------------------------------------------------------------------
class STGRUModel(BaseMultiHeadModel):
    def __init__(self, stgcn_cfg, gru_cfg, input_dim, heads_cfg, full_config):
        hidden = gru_cfg.get('hidden') if isinstance(gru_cfg, dict) else None
        if hidden is None:
            raise KeyError(
                "GRU configuration missing required key 'hidden'. "
                f"Provided gru_cfg keys: {list(gru_cfg.keys()) if isinstance(gru_cfg, dict) else type(gru_cfg)}. "
                "Ensure you're passing the inner 'gru' sub-dict (e.g. arch_configs['TCN_GRU_BASELINE']['gru'])."
            )

        super().__init__(hidden, heads_cfg, full_config)
        
        # FIX: สร้าง Network Backbone โดยตรง
        self.stgcn_network = build_stgcn_network(stgcn_cfg, heads_cfg)
        self.gru = nn.GRU(stgcn_cfg['channels'][-1], gru_cfg['hidden'], gru_cfg['layers'], batch_first=True)

    def forward(self, x: torch.Tensor):
        B, T, V, C = x.shape
        
        # Pass through STGCN layers: (B, C_final, T, V)
        stgcn_features = self.stgcn_network(x.permute(0, 3, 1, 2).contiguous())
        
        # Input to GRU: (B, T, C_final)
        gru_input = stgcn_features.mean(dim=3).permute(0, 2, 1)
        
        gru_output, _ = self.gru(gru_input)
        final_feature = gru_output[:, -1, :] # Final time step (B, 256)
        
        return self.forward_heads(final_feature)

# --- Entry Point Function ---

def get_candidate_model(arch_name: str, config: dict) -> nn.Module:
    """Instantiates the specified model architecture."""
    
    # Base Configuration Dictionaries
    arch_configs = config['arch_configs']
    heads_config = config['heads']
    input_dim = config['dataset']['joints'] * config['dataset']['input_dim']
    
    if arch_name == 'TCN_GRU_BASELINE':
        return TCNGRUModel(arch_configs[arch_name], input_dim, heads_config, config)
    
    elif arch_name == 'BIGRU_ATTENTION':
        return BiGRUAttentionModel(arch_configs[arch_name], input_dim, heads_config, config)

    elif arch_name == 'STGCN_BASELINE':
        return STGCNModel(arch_configs['STGCN_COMMON'], input_dim, heads_config, config)

    elif arch_name == 'STGCN_ATTENTION':
        return STGCNAttentionModel(arch_configs['STGCN_COMMON'], arch_configs['BIGRU_ATTENTION'], input_dim, heads_config, config)

    elif arch_name == 'STGCN_GRU':
        # Resolve ST/GRU sub-configs safely (model_archs.py)
        stgcn_cfg = arch_configs.get('STGCN_COMMON', {})
        tcn_gru_block = arch_configs.get('TCN_GRU_BASELINE', {})

        # Prefer explicit 'gru' sub-block if present, otherwise assume the block itself IS the gru config
        gru_cfg = tcn_gru_block.get('gru', tcn_gru_block)

        return STGRUModel(stgcn_cfg, gru_cfg, input_dim, heads_config, config)
        
    raise ValueError(f"Unknown architecture: {arch_name}")