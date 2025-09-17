import numpy as np
import onnxruntime as ort
from .base_model import BaseModel

class SupervisedTcnGru(BaseModel):
    def __init__(self, window_size: int, feature_dim: int, num_classes: int):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.session = None
        print(f"Initialized SupervisedTcnGru with window: {window_size}, features: {feature_dim}")

    def load(self, path: str):
        print(f"Loading Supervised ONNX model from: {path}")
        try:
            self.session = ort.InferenceSession(path)
            print("✅ Supervised model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load supervised ONNX model: {e}")
            raise e

    def predict(self, input_data: np.ndarray) -> dict:
        """
        Runs inference and returns structured output from all three heads.
        input_data shape: (batch_size, window_size, feature_dim)
        """
        if not self.session:
            raise RuntimeError("Model is not loaded. Call .load(path) first.")

        input_name = self.session.get_inputs()[0].name
        output_names = [o.name for o in self.session.get_outputs()]
        
        # ONNX Runtime returns a list of numpy arrays
        # The order matches the output_names in the export script
        # ['anomaly', 'classification', 'regression']
        results = self.session.run(output_names, {input_name: input_data.astype(np.float32)})
        
        anomaly_pred = results[0]
        class_pred_logits = results[1]
        regression_pred = results[2]

        # Process the outputs
        # Apply sigmoid to anomaly score to get a probability
        anomaly_score = 1 / (1 + np.exp(-anomaly_pred))
        # Find the most likely class using argmax on the logits
        predicted_class_id = np.argmax(class_pred_logits, axis=1)

        return {
            "anomaly_score": anomaly_score[0][0], # un-batch and un-squeeze
            "predicted_class_id": predicted_class_id[0], # un-batch
            "predicted_class_logits": class_pred_logits[0].tolist(),
            "predicted_severity": regression_pred[0][0] # un-batch and un-squeeze
        }

