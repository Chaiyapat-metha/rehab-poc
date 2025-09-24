# data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import psycopg2
import math
import random

# --- BlazePose Keypoint Indices (from MediaPipe Pose) ---
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
# We need to add all 33 keypoints here for PoseAug to work completely.
# For simplicity in this example, we'll only use the ones needed for the 6 angles.

# --- Core Calculation Functions ---
def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def _get_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return p2 - p1

def get_6_angles(keypoints: np.ndarray) -> np.ndarray:
    angles = np.zeros(6)
    angles[0] = _calculate_angle(keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST])
    angles[1] = _calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW])
    angles[2] = _calculate_angle(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW], keypoints[RIGHT_WRIST])
    angles[3] = _calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW])
    angles[4] = _calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
    angles[5] = _calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
    return angles

class PoseDataset(Dataset):
    def __init__(self, db_config, exercise_name, config):
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        
        try:
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

        # Query to fetch raw_frame_data as a bytea for PoseAug
        # We assume the bytea data can be converted back to a numpy array of shape (33, 3)
        self.cursor.execute("""
            SELECT f.raw_frame_data
            FROM frames AS f
            JOIN sessions AS s ON f.session_id = s.session_id
            WHERE s.exercise_name = %s
            ORDER BY f."time" ASC;
        """, (exercise_name,))
        
        self.all_data = self.cursor.fetchall()
        
        if not self.all_data:
            raise ValueError(f"No data found for exercise '{exercise_name}'.")
        
        # NOTE: This part needs a proper protobuf decoder
        # For now, we will use a placeholder for the raw data
        # In a real scenario, you'd decode f.raw_frame_data from protobuf to numpy
        num_frames = len(self.all_data)
        self.raw_frames = np.random.rand(num_frames, self.joints, self.input_dim).astype(np.float32)

        self.num_frames = self.raw_frames.shape[0]
        self.num_windows = self.num_frames // self.window_size

        if self.num_windows == 0:
            raise ValueError("Not enough data to create a single window.")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        
        window = self.raw_frames[start_idx:end_idx, :, :]
        
        # We need a reference clip for calculating the angle difference
        # In the absence of a proper reference, we use the first frame of the window
        reference_frame = window[0, :, :]
        reference_angles = get_6_angles(reference_frame)

        # Apply PoseAug if enabled
        if self.config['augmentation']['poseaug']['enabled']:
            window = self._pose_aug(window)
        
        # Calculate target labels (pos and angle)
        # Target for L_pos is the last frame of the window
        target_pos = window[-1, :, :]
        
        # Target for L_angle is the normalized angle difference of the last frame
        target_angles = get_6_angles(window[-1, :, :])
        angle_diff = np.abs(target_angles - reference_angles)
        normalized_angle_diff = angle_diff / 360.0
        
        return torch.from_numpy(window).float(), torch.from_numpy(target_pos).float(), torch.from_numpy(normalized_angle_diff).float()

    def _pose_aug(self, window):
        # We need to implement the full PoseAug logic here
        # This includes BA, BL, and RT
        # This is a complex function and needs careful implementation based on the paper
        return window # Placeholder for the actual implementation

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()