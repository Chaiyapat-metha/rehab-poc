# rehab-poc/backend/training/experiment_candidates/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import psycopg2
import math
import random
from typing import Tuple, Dict

# --- Keypoint Indices (33 Joints) ---
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28
# Define joints used for the 6 angles (as per the model spec)
ANGLE_JOINTS = [11, 12, 23, 24, 25, 26] # Placeholder for full set of joints involved in 6 angles

# --- Core Calculation Functions ---
def _calculate_angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates angle (in radians) between three points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle 

def get_6_angles(keypoints: np.ndarray) -> np.ndarray:
    """Calculates the 6 bone angles (in radians)."""
    angles = np.zeros(6)
    angles[0] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW])
    angles[1] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW])
    angles[2] = _calculate_angle_3p(keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP], keypoints[LEFT_KNEE])
    angles[3] = _calculate_angle_3p(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE])
    angles[4] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
    angles[5] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
    return angles

def get_angle_mask(keypoints: np.ndarray, visibility_threshold: float = 0.5) -> np.ndarray:
    """Generates a mask for the 6 angles based on joint visibility."""
    # NOTE: This function requires visibility data, which is currently not queried.
    # We create a placeholder mask based on the assumption that all keypoints are valid (mask=1)
    mask = np.ones(6, dtype=np.float32)
    return mask

# --- AUGMENTATION LOGIC (Simplified based on previous detailed request) ---
# NOTE: We assume Augmentation is applied in train_one_epoch for simplicity, 
# but for full on-the-fly augmentation, it should be here.
# For now, we put the logic here as part of the _augment method:

def _augment_window(window: np.ndarray, config: dict, is_train: bool) -> np.ndarray:
    if not is_train or not config['augmentation']['enabled']:
        return window

    aug_cfg = config['augmentation']
    augmented_window = window.copy()
    
    # 1. Random Rotation (Z-axis, Yaw)
    if random.random() < aug_cfg['rotation']['p_rot']:
        rot_z_deg = aug_cfg['rotation']['rot_z_deg']
        angle_rad = np.radians(random.uniform(-rot_z_deg, rot_z_deg))
        rot_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        # Apply rotation to every frame (f)
        augmented_window = np.einsum('fjc,ck->fjk', augmented_window, rot_matrix)
        
    # 2. Random Jittering
    if random.random() < aug_cfg['jitter']['p_jitter']:
        jitter_std = aug_cfg['jitter']['sigma_scale']
        jitter_noise = np.random.normal(0, jitter_std, augmented_window.shape)
        augmented_window += jitter_noise

    # 3. Random Occlusion Masking (Set to zero + Mask indicator if needed)
    if random.random() < aug_cfg['occlusion']['p_occ']:
        k_min, k_max = aug_cfg['occlusion']['k_range']
        num_joints_to_mask = random.randint(k_min, k_max)
        joints_to_mask = random.sample(range(config['dataset']['joints']), num_joints_to_mask)
        
        # Masking by setting the joint coordinates to zero
        augmented_window[:, joints_to_mask, :] = 0.0

    return augmented_window

class PoseDataset(Dataset):
    def __init__(self, db_config: dict, config: dict, is_train_set: bool = False):
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        self.is_train_set = is_train_set
        
        # --- FIX: Database Connection and Query must be inside the try block ---
        try:
            # 1. Establish connection and cursor
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # 2. Execute Query
            # NOTE: We need all three fields for multi-task and masking (keypoints, label, visibility)
            cursor.execute("""
                SELECT keypoints, label, visibility
                FROM training_skeletons 
                ORDER BY video_id, frame_no ASC;
            """)
            
            all_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
        except psycopg2.OperationalError as e:
             # Handle specific database connection failure gracefully
            raise ConnectionError(f"Database connection failed: {e}")
        except Exception as e:
            # Handle any other general error during data fetching
            raise Exception(f"Error fetching data from database: {e}")


        if not all_data:
            raise ValueError("No data found in the training_skeletons table.")
        
        # 3. Process Data
        # We need to access item[2] for visibility
        keypoints_list = [item[0] for item in all_data]
        label_list = [1.0 if item[1] == 'wrong' else 0.0 for item in all_data]
        # We assume item[2] returns a list of floats (visibility scores)
        visibility_list = [item[2] for item in all_data] 
        
        self.keypoints = np.array(keypoints_list, dtype=np.float32).reshape(-1, self.joints, self.input_dim)
        self.labels = np.array(label_list, dtype=np.float32)
        self.visibilities = np.array(visibility_list, dtype=np.float32)
        self.num_frames = self.keypoints.shape[0]
        
        self.windows = []
        self.window_labels = []
        stride = config['dataset']['stride']
        
        for i in range(0, self.num_frames - self.window_size + 1, stride):
            window = self.keypoints[i:i + self.window_size, :, :]
            
            # Apply initial normalization only once
            normalized_window = np.array([self._normalize_pose(frame) for frame in window])
            
            window_label = self.labels[i + self.window_size - 1]
            
            self.windows.append(normalized_window)
            self.window_labels.append(window_label)

    def _normalize_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """Root-center by mid-hip and scale normalize by torso length."""
        normalized_pose = keypoints.copy()
        
        mid_hip_coord = (normalized_pose[LEFT_HIP] + normalized_pose[RIGHT_HIP]) / 2
        normalized_pose -= mid_hip_coord
        
        torso_vector = normalized_pose[LEFT_SHOULDER] - normalized_pose[RIGHT_SHOULDER]
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length > 1e-6:
            normalized_pose /= torso_length
        
        return normalized_pose

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window_np = self.windows[idx].copy() 
        target_class_np = self.window_labels[idx]
        
        # Apply Augmentation (Only on Training Set)
        window_np = _augment_window(window_np, self.config, self.is_train_set)
        
        # 1. L_pos Target
        target_pos = window_np[-1, :, :].reshape(-1) 
        
        # 2. L_angle Target
        target_angles = get_6_angles(window_np[-1, :, :])
        
        # 3. Target for Classification (1 dim)
        target_class_np = self.window_labels[idx] 
        
        # 4. Target Mask (1 dim for the 6 angles - Placeholder here, but needs to be used in loss)
        # target_mask = get_angle_mask(...)

        return (
            torch.from_numpy(window_np).float(),         # 1. Input Data (T, V, C)
            torch.from_numpy(target_pos).float(),        # 2. L_pos Target
            torch.from_numpy(target_angles).float(),     # 3. L_angle Target
            torch.tensor(target_class_np).float()        # 4. L_class Target
        )