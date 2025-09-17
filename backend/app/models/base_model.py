from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        """รับ parameters จาก config"""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> dict:
        """
        รับ input เป็น numpy array (e.g., window of frames)
        และ return output เป็น dictionary ที่มีโครงสร้างชัดเจน
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """โหลด weights ของโมเดル"""
        pass

    def save(self, path: str):
        """(Optional) บันทึก weights ของโมเดล"""
        raise NotImplementedError