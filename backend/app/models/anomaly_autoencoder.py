import numpy as np
import onnxruntime as ort
from .base_model import BaseModel

class AutoencoderAnomaly(BaseModel):
    def __init__(self, window_size: int, feature_dim: int, latent_dim: int):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.session = None
        print(f"Initialized AutoencoderAnomaly with window: {window_size}, features: {feature_dim}")

    def load(self, path: str):
        print(f"Loading ONNX model from: {path}")
        # TODO: ตรวจสอบว่าไฟล์มีอยู่จริง
        self.session = ort.InferenceSession(path)

    def predict(self, input_data: np.ndarray) -> dict:
        """
        input_data shape: (batch_size, window_size, feature_dim)
        """
        if not self.session:
            raise RuntimeError("Model is not loaded. Call .load(path) first.")

        # ตรวจสอบ shape ของ input
        expected_shape = (-1, self.window_size, self.feature_dim)
        if input_data.ndim != 3 or input_data.shape[1:] != (self.window_size, self.feature_dim):
             raise ValueError(f"Input data shape mismatch. Expected (*, {self.window_size}, {self.feature_dim}), got {input_data.shape}")

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        reconstructed = self.session.run([output_name], {input_name: input_data.astype(np.float32)})[0]
        
        # คำนวณ reconstruction error
        error = np.mean((input_data - reconstructed)**2, axis=(1, 2))
        
        return {
            "anomaly_score": error.tolist(),
            "reconstructed_features": reconstructed.tolist()
        }