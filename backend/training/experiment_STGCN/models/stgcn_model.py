# rehab-poc/backend/training/experiment_STGCN/models/stgcn_model.py

import torch
import torch.nn as nn
import numpy as np
from typing import List

# --- Graph Utilities (ST-GCN A-Link Matrix) ---
def get_adjacency_matrix(num_joints: int):
    """
    Creates a basic adjacency matrix based on the MediaPipe/Pose structure.
    This defines the connections (bones) between joints.
    """
    # Simplified connections for the 33-joint body (Hip, Torso, Limbs)
    # This needs to be comprehensive for all 33 joints for the model to work properly.
    # For a quick start, we define the core limbs:
    
    # Core connections (parent to child indices)
    edges = [
        (24, 23), (24, 26), (23, 25), (26, 28), (25, 27), # Hips/Legs
        (24, 12), (23, 11), (12, 14), (11, 13), (14, 16), (13, 15) # Torso/Arms
        # In a full ST-GCN, you'd add Neck, Head, Feet, etc.
    ]
    
    A = np.zeros((num_joints, num_joints))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1 # Undirected graph
        
    # Standard ST-GCN uses A + Identity Matrix for self-connections
    A_normalized = A + np.eye(num_joints)
    
    # We will simply return the non-normalized A for simplicity here
    return torch.tensor(A_normalized, dtype=torch.float32)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t, stride, A):
        super().__init__()
        
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) # Graph Convolution (Spatial)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_t, 1), 
                             padding=((kernel_size_t - 1) // 2, 0), stride=(stride, 1)) # Temporal Convolution
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = A
        self.stride = stride
        
    def forward(self, x):
        # x shape: (N, C, T, V) where V=Joints
        
        # 1. Spatial GCN (GCN=x * A)
        x_gcn = torch.einsum('nctv, vk -> nctk', x, self.A.to(x.device)) # N, C, T, V @ V, V -> N, C, T, V
        x_gcn = self.gcn(x_gcn)
        
        # 2. Temporal TCN
        x_tcn = self.tcn(x_gcn)
        
        # 3. Residual Connection (if needed) and Activation
        return self.relu(self.bn(x_tcn))


class STGCNModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        num_joints = config['dataset']['joints'] # 33
        in_dim = config['dataset']['input_dim'] # 3
        
        A = get_adjacency_matrix(num_joints)
        
        # 1. Initial Transformation: (N, T, V, 3) -> (N, 3, T, V)
        self.data_bn = nn.BatchNorm1d(in_dim * num_joints)
        
        # 2. ST-GCN Blocks
        stgcn_channels = config['model']['stgcn']['channels']
        kernel_size_t = config['model']['stgcn']['kernel_size_t']
        final_channels = stgcn_channels[-1]
        
        layers = []
        in_channels = in_dim
        for out_channels in stgcn_channels:
            layers.append(STGCNBlock(in_channels, out_channels, kernel_size_t, 1, A))
            in_channels = out_channels
        self.stgcn_network = nn.Sequential(*layers)
        
        # 3. Output Pooling & Heads
        
        # Head 1: Positional Regression (L_pos)
        self.pos_head = nn.Linear(final_channels, config['heads']['pos_regression']['output_dim'])
        
        # Head 2: Angle Regression (L_angle)
        angle_dim = config['heads']['angle_regression']['output_dim']
        self.angle_head = nn.Sequential(
            nn.Linear(final_channels, 64),
            nn.ReLU(),
            nn.Linear(64, angle_dim)
        )
        
        # Head 3: Classification (L_class)
        class_dim = config['heads']['classification']['output_dim']
        self.class_head = nn.Sequential(
            nn.Linear(final_channels, 64),
            nn.ReLU(),
            nn.Linear(64, class_dim)
        )
    def forward(self, x: torch.Tensor):
        # x shape: (B, T, V, C)
        B, T, V, C = x.shape
        
        # 1. Preprocessing: (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 2. ST-GCN Forward
        x = self.stgcn_network(x) # Output: (B, Final_C, T_reduced, V)
        
        # 3. Global Temporal and Spatial Pooling (Shared Feature)
        final_feature = x.mean(dim=3).mean(dim=2) # Average over T and V: Output (B, Final_C)
        
        # 4. Output Heads
        return {
            'pos_output': self.pos_head(final_feature),
            'angle_output': self.angle_head(final_feature), # FIX: Added Angle Output
            'class_output': self.class_head(final_feature) # FIX: Added Class Output
        }