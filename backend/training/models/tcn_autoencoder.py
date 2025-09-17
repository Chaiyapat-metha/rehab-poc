import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# --- Reusable TemporalBlock (Moved outside) ---
# This block is the fundamental building block for a TCN.
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0) # Removes padding from the end
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # The full block of operations
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection: if input/output channels differ, use a 1x1 conv to match them
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# --- TCN Autoencoder Architecture ---
# This class now uses the standalone TemporalBlock
class TCNAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, tcn_channels=[64, 32]):
        super(TCNAutoencoder, self).__init__()
        
        # --- Encoder ---
        encoder_layers = []
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feature_dim if i == 0 else tcn_channels[i-1]
            out_channels = tcn_channels[i]
            encoder_layers += [TemporalBlock(in_channels, out_channels, kernel_size=3, stride=1,
                                             dilation=dilation_size, padding=(3-1) * dilation_size)]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck = nn.Linear(tcn_channels[-1], latent_dim)

        # --- Decoder ---
        decoder_layers = []
        # We reverse the channels for the decoder
        decoder_channels = [latent_dim] + list(reversed(tcn_channels))
        
        # This part needs a more sophisticated design, often using Transposed Convolutions.
        # For simplicity in this PoC, we'll use a simple Linear decoder.
        # A full TCN decoder is more complex to implement correctly.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, tcn_channels[-1]),
            nn.ReLU(),
            nn.Linear(tcn_channels[-1], feature_dim)
            # Note: This is a simplified decoder. A proper TCN decoder would use Conv1dTranspose.
        )
        

    def forward(self, x):
        # x shape: (batch, window_size, features)
        x = x.permute(0, 2, 1) # (batch, features, window_size)
        
        # Encode
        encoded = self.encoder(x)
        encoded_permuted = encoded.permute(0, 2, 1) # (batch, window_size, channels)
        
        # We take the last time step's output to represent the whole window
        latent = self.bottleneck(encoded_permuted[:, -1, :])

        # Decode
        # We need to replicate the latent vector for each time step to decode back to a sequence
        replicated_latent = latent.unsqueeze(1).repeat(1, x.size(2), 1)
        decoded_permuted = self.decoder(replicated_latent)
        
        decoded = decoded_permuted.permute(0, 2, 1) # (batch, features, window_size)
        
        return decoded
