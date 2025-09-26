# rehab-poc/backend/training/experiment_STGCN/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import psycopg2
import math
import random
from typing import Tuple, Dict

# --- MediaPipe Keypoint Indices ---
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# --- Core Calculation Functions ---
def _calculate_angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates angle (in radians) between three points."""
    # (Simplified implementation for a multi-task scenario)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle 

def get_bone_angle_targets(keypoints: np.ndarray) -> np.ndarray:
    """Calculates the 6 bone angles (in radians) for the defined joints."""
    angles = np.zeros(6)
    angles[0] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW]) # L-Shoulder
    angles[1] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW]) # R-Shoulder
    angles[2] = _calculate_angle_3p(keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP], keypoints[LEFT_KNEE]) # L-Hip
    angles[3] = _calculate_angle_3p(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE]) # R-Hip
    angles[4] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE]) # L-Knee
    angles[5] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE]) # R-Knee
    return angles

class PoseDataset(Dataset):
    def __init__(self, db_config: dict, config: dict):
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        
        # ... (Database connection setup remains the same)
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

        # Query to fetch keypoints and labels
        cursor.execute("""
            SELECT keypoints, label
            FROM training_skeletons
            ORDER BY video_id, frame_no ASC;
        """)

        all_data = cursor.fetchall()
        
        if not all_data:
            raise ValueError("No data found in the training_skeletons table.")
        
        # Separate keypoints and labels
        keypoints_list = [item[0] for item in all_data]
        label_list = [1.0 if item[1] == 'wrong' else 0.0 for item in all_data] # Binary Label
        
        self.keypoints = np.array(keypoints_list, dtype=np.float32).reshape(-1, self.joints, self.input_dim)
        self.labels = np.array(label_list, dtype=np.float32)

        self.num_frames = self.keypoints.shape[0]
        
        # --- Create windows and apply initial normalization/augmentation ---
        self.windows = []
        self.window_labels = []
        stride = config['dataset']['stride']
        
        for i in range(0, self.num_frames - self.window_size + 1, stride):
            window = self.keypoints[i:i + self.window_size, :, :]
            
            # Apply initial normalization (Root-center and Scale)
            normalized_window = np.array([self._normalize_pose(frame) for frame in window])
            
            # Label based on the last frame
            window_label = self.labels[i + self.window_size - 1]
            
            self.windows.append(normalized_window)
            self.window_labels.append(window_label)

    def _normalize_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """Root-center by mid-hip and scale normalize by torso length."""
        normalized_pose = keypoints.copy()
        
        # 1. Root-center: Mid-Hip (Average of 23 and 24)
        mid_hip_coord = (normalized_pose[LEFT_HIP] + normalized_pose[RIGHT_HIP]) / 2
        normalized_pose -= mid_hip_coord
        
        # 2. Scale normalize: Torso length (Shoulder-to-Shoulder distance)
        torso_vector = normalized_pose[LEFT_SHOULDER] - normalized_pose[RIGHT_SHOULDER]
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length > 1e-6:
            normalized_pose /= torso_length
        
        return normalized_pose

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Data is already normalized and ready
        window = self.windows[idx].copy() 
        target_class_np = self.window_labels[idx]
        
        # We skip explicit augmentation here for the ST-GCN Baseline
        
        # 1. Target for L_pos: Last frame (Flattened 99 dim)
        target_pos = window[-1, :, :].reshape(-1)
        
        # 2. Target for L_angle: 6 calculated angles (Radians)
        target_angles = get_bone_angle_targets(window[-1, :, :])
        
        # 3. Target for L_class: Binary label (0 or 1)
        
        return (
            torch.from_numpy(window).float(),              # Input Data (B, T, V, C)
            torch.from_numpy(target_pos).float(),          # Target L_pos (B, 99)
            torch.from_numpy(target_angles).float(),       # Target L_angle (B, 6)
            torch.tensor(target_class_np).float()          # Target L_class (B, 1)
        )