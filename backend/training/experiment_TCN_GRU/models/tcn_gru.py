# models/tcn_gru.py
import torch
import torch.nn as nn
from typing import List

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

class TCNGRUModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # --- Input Embedding ---
        input_dim = config['dataset']['input_dim']
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # --- TCN Part ---
        tcn_channels = config['model']['tcn']['channels']
        kernel_size = config['model']['tcn']['kernel']
        tcn_blocks = config['model']['tcn']['blocks']
        tcn_dilations = config['model']['tcn']['dilation_levels']
        
        tcn_layers = []
        in_channels = 256
        for i in range(tcn_blocks):
            dilation = tcn_dilations[i % len(tcn_dilations)]
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
        
        # --- Output Head for Angle Regression ---
        output_dim = config['heads']['angle_regression']['output_dim']
        self.angle_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid() # Use Sigmoid to scale output to 0-1 range
        )

    def _build_pooling(self, pool_type: str, pool_size: int):
        if pool_type == 'max':
            return nn.MaxPool1d(kernel_size=pool_size)
        elif pool_type == 'average':
            return nn.AvgPool1d(kernel_size=pool_size)
        else:
            return nn.Identity()

    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        
        # --- Input MLP ---
        x = self.input_mlp(x) # output: (batch, window_size, 256)
        
        # --- TCN ---
        x = x.permute(0, 2, 1) # Permute for TCN: (batch, channels, window_size)
        x = self.tcn(x)
        
        # --- Pooling ---
        x = self.pooling(x)
        
        # --- GRU ---
        x = x.permute(0, 2, 1) # Permute back for GRU: (batch, window_size, channels)
        
        gru_output, _ = self.gru(x)
        
        # --- Output Head ---
        last_timestep_output = gru_output[:, -1, :] # Take the last hidden state
        prediction = self.angle_head(last_timestep_output)
        
        return prediction