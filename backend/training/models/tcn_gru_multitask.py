import torch
import torch.nn as nn
# --- CORRECTED IMPORT ---
# The '.' tells Python to look in the current package/directory first.
from .tcn_autoencoder import TemporalBlock 

class TCN_GRU_MultiTask(nn.Module):
    def __init__(self, feature_dim, num_classes, tcn_channels=[32, 64, 128], gru_units=128):
        super(TCN_GRU_MultiTask, self).__init__()

        # TCN Encoder Part
        layers = []
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feature_dim if i == 0 else tcn_channels[i-1]
            out_channels = tcn_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size=3, stride=1, 
                                     dilation=dilation_size, padding=(3-1) * dilation_size)]
        self.tcn_encoder = nn.Sequential(*layers)

        # GRU Part
        self.gru = nn.GRU(tcn_channels[-1], gru_units, batch_first=True)

        # --- Multi-task Output Heads ---
        # 1. Anomaly Head (Binary Classification - is it wrong?)
        self.anomaly_head = nn.Sequential(
            nn.Linear(gru_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output for anomaly score
        )
        
        # 2. Classification Head (Multi-class - what kind of error?)
        self.class_head = nn.Sequential(
            nn.Linear(gru_units, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # 3. Regression Head (Regression - by how much?)
        self.regression_head = nn.Sequential(
            nn.Linear(gru_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output for severity score
        )

    def forward(self, x):
        # x shape: (batch, window_size, features)
        x = x.permute(0, 2, 1) # (batch, features, window_size) for TCN
        tcn_out = self.tcn_encoder(x)
        tcn_out = tcn_out.permute(0, 2, 1) # (batch, window_size, tcn_channels) for GRU
        
        gru_out, _ = self.gru(tcn_out)
        # We only need the last time step's output for prediction
        last_step_out = gru_out[:, -1, :]

        # Get predictions from each head
        anomaly_pred = self.anomaly_head(last_step_out)
        class_pred = self.class_head(last_step_out)
        regression_pred = self.regression_head(last_step_out)

        return {
            "anomaly": anomaly_pred,
            "classification": class_pred,
            "regression": regression_pred
        }

