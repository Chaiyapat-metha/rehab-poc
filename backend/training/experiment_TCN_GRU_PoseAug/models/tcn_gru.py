# rehab-poc/backend/training/experiment_TCN_GRU_PoseAug/models/tcn_gru.py

import torch
import torch.nn as nn
from typing import List

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
        # Residual connection
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

class TCNGRUModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # --- Input Embedding ---
        input_dim = config['dataset']['joints'] * config['dataset']['input_dim']
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # --- TCN Part ---
        tcn_channels = config['model']['tcn']['channels']
        kernel_size = config['model']['tcn']['kernel']
        tcn_blocks = config['model']['tcn']['blocks']
        
        tcn_layers = []
        in_channels = 256
        for i in range(tcn_blocks):
            dilation = config['model']['tcn']['dilation_levels'][i % len(config['model']['tcn']['dilation_levels'])]
            out_channels = tcn_channels[min(i, len(tcn_channels) - 1)]
            tcn_layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))
            in_channels = out_channels
        self.tcn = nn.Sequential(*tcn_layers)
        
        # --- Pooling Part ---
        self.pooling = self._build_pooling(config['model']['pooling']['type'], config['model']['pooling']['pool_size'])
        
        # --- GRU Part ---
        gru_layers = config['model']['gru']['layers']
        gru_hidden = config['model']['gru']['hidden']
        gru_dropout = config['model']['gru']['dropout']
        
        self.gru = nn.GRU(
            input_size=tcn_channels[-1] if tcn_blocks > 0 else 256,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0
        )
        
        # --- Output Heads for Multi-task Regression ---
        pos_output_dim = config['heads']['pos_regression']['output_dim']
        angle_output_dim = config['heads']['angle_regression']['output_dim']
        class_output_dim = config['heads']['classification']['output_dim']
        
        gru_hidden = config['model']['gru']['hidden']
        class_hidden = config['heads']['classification']['hidden']
        
        self.pos_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, pos_output_dim) # Output 99 dimensions
        )

        self.angle_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, angle_output_dim) # Output 6 dimensions
        )

        self.class_head = nn.Sequential(
            nn.Linear(gru_hidden, class_hidden),
            nn.ReLU(),
            nn.Linear(class_hidden, class_output_dim) 
        )
                
    def _build_pooling(self, pool_type: str, pool_size: int):
        if pool_type == 'max':
            return nn.MaxPool1d(kernel_size=pool_size)
        elif pool_type == 'average':
            return nn.AvgPool1d(kernel_size=pool_size)
        else:
            return nn.Identity()

    def forward(self, x):
        # x shape: (batch, window_size, joints, dim)
        batch_size, window_size, num_joints, input_dim = x.shape
        
        # 1. Flatten: (batch, window_size, 99)
        x = x.view(batch_size, window_size, -1) 
        
        # 2. Input MLP: Embed 99 dims to 256 dims (The code from the previous fix)
        x = self.input_mlp(x) 
        # Output shape is now (batch, window_size, 256)
        
        # 3. Permute for TCN: (batch, channels, sequence_length)
        x = x.permute(0, 2, 1) # Output shape: (batch, 256, window_size)
        
        # --- TCN Convolution (Correctly operates over the 32 timesteps) ---
        x = self.tcn(x)
        
        # --- Pass through Pooling ---
        x = self.pooling(x)
        
        # 4. Permute back for GRU: (batch, sequence_length, channels)
        x = x.permute(0, 2, 1) 
        
        gru_output, _ = self.gru(x)
        
        last_timestep_output = gru_output[:, -1, :]
        pos_output = self.pos_head(last_timestep_output)
        angle_output = self.angle_head(last_timestep_output)
        class_output = self.class_head(last_timestep_output) 
        
        return {
            'pos_output': pos_output, 
            'angle_output': angle_output,
            'class_output': class_output # <--- คืนค่า Classification Output
        }