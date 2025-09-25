# rehab-poc/backend/training/experiment_BiGRU_Att/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import psycopg2
import math
import random
from typing import Tuple

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
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle # Returns in Radians

def get_bone_angle_targets(keypoints: np.ndarray) -> np.ndarray:
    """Calculates the bone angles (in radians) for the defined joints."""
    angles = np.zeros(6)
    # 6 Angles: L/R Shoulder, L/R Hip, L/R Knee
    
    # Left Shoulder Angle (between Hip-Shoulder-Elbow)
    angles[0] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW])
    # Right Shoulder Angle
    angles[1] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW])
    
    # Left Hip Angle (between Torso-Hip-Knee)
    angles[2] = _calculate_angle_3p(keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP], keypoints[LEFT_KNEE])
    # Right Hip Angle
    angles[3] = _calculate_angle_3p(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE])

    # Left Knee Angle (between Hip-Knee-Ankle)
    angles[4] = _calculate_angle_3p(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
    # Right Knee Angle
    angles[5] = _calculate_angle_3p(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
    
    return angles # Shape (6,)

class PoseDataset(Dataset):
    def __init__(self, db_config: dict, config: dict):
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        
        # NOTE: For Exp A, we need a reference pose (theta_ref) to calculate Delta_theta.
        # Since the database contains "Jump squats" (which are generally correct) 
        # we will use the average pose of all 'correct' frames as a simple theta_ref for now.
        
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT keypoints, label
                FROM training_skeletons
                ORDER BY video_id, frame_no ASC;
            """)
            all_data = cursor.fetchall()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

        if not all_data:
            raise ValueError("No data found in the training_skeletons table.")
        
        keypoints_list = [item[0] for item in all_data]
        self.raw_keypoints = np.array(keypoints_list, dtype=np.float32).reshape(-1, self.joints, self.input_dim)
        self.num_frames = self.raw_keypoints.shape[0]
        
        # --- Precompute Reference Angles (theta_ref) ---
        # Find all 'correct' frames
        correct_indices = [i for i, item in enumerate(all_data) if item[1] == 'correct']
        if not correct_indices:
             raise ValueError("No 'correct' frames found to establish a reference pose.")
        
        # Simple method: use the average of all 'correct' keypoints as the reference
        ref_keypoints = self.raw_keypoints[correct_indices].mean(axis=0)
        self.theta_ref = get_bone_angle_targets(self._normalize_pose(ref_keypoints))
        
        # --- Create windows and apply initial normalization/augmentation ---
        self.windows = []
        stride = config['dataset']['stride']
        
        for i in range(0, self.num_frames - self.window_size + 1, stride):
            window = self.raw_keypoints[i:i + self.window_size, :, :]
            
            # Apply initial normalization to the raw window data
            normalized_window = np.array([self._normalize_pose(frame) for frame in window])
            
            # Calculate Delta_theta for the last frame
            theta_live = get_bone_angle_targets(normalized_window[-1])
            delta_theta_gt = self.theta_ref - theta_live # Signed difference in Radians
            
            # Store the normalized window and its GT delta-theta
            self.windows.append({
                'data': normalized_window,
                'delta_theta_gt': delta_theta_gt
                # NOTE: We skip calculating masks for now, but should be added later
            })

    def _normalize_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """Root-center by mid-hip and scale normalize by torso length."""
        normalized_pose = keypoints.copy()
        
        # 1. FIX 2: Root-center: Calculate Mid-Hip (Average of 23 and 24)
        mid_hip_coord = (normalized_pose[LEFT_HIP] + normalized_pose[RIGHT_HIP]) / 2
        normalized_pose -= mid_hip_coord
        
        # 2. Scale normalize: Torso length (Distance between shoulders)
        # Assuming TORSO_JOINT_1 = LEFT_SHOULDER (11) and TORSO_JOINT_2 = RIGHT_SHOULDER (12)
        torso_vector = normalized_pose[LEFT_SHOULDER] - normalized_pose[RIGHT_SHOULDER]
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length > 1e-6:
            normalized_pose /= torso_length
        
        return normalized_pose

    def _augment(self, window: np.ndarray) -> np.ndarray:
        """
        Applies basic data augmentation to the skeleton window.
        - Random Rotation around Z-axis
        - Random Jittering of joints
        - Random Occlusion masking of joints
        """
        augmented_window = window.copy()

        # 1. Random Rotation (around Z-axis, which is the axis pointing out of the camera)
        rot_z_deg = self.config['augmentation']['basic']['rot_z_deg']
        if rot_z_deg > 0:
            angle_rad = np.radians(random.uniform(-rot_z_deg, rot_z_deg))
            
            # 2D Rotation matrix on X and Y axes
            rot_matrix = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            augmented_window = np.einsum('fji,ik->fjk', augmented_window, rot_matrix)
        
        # 2. Random Jittering (adding small noise to each joint)
        jitter_std = self.config['augmentation']['basic']['jitter_std']
        if jitter_std > 0:
            jitter_noise = np.random.normal(0, jitter_std, augmented_window.shape)
            augmented_window += jitter_noise

        # 3. Random Occlusion Masking
        p_mask = self.config['augmentation']['occlusion']['p_mask']
        if p_mask > 0:
            num_joints_to_mask = int(self.joints * p_mask)
            joints_to_mask = random.sample(range(self.joints), num_joints_to_mask)
            
            # Masking by setting the joint coordinates to zero
            augmented_window[:, joints_to_mask, :] = 0.0

        return augmented_window

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        window = item['data'].copy()
        delta_theta_gt = item['delta_theta_gt']
        
        # Apply augmentation only to training data (randomly flipped or modified)
        # We assume the PoseDataset is used for both train/val/test splits now.
        # The logic to only augment the train split is handled by DataLoader's shuffle/sampler.
        
        window = self._augment(window) # Apply basic augmentations
        
        # Final shape for model input: (T, J, 3)
        window_tensor = torch.from_numpy(window).float()
        
        # Target: Delta Theta GT (6 signed angle differences in radians)
        delta_theta_target = torch.from_numpy(delta_theta_gt).float()
        
        # For Exp A, we need the GT in Radians.
        # We also need to add a mask tensor (Placeholder: all ones)
        mask_tensor = torch.ones_like(delta_theta_target)
        
        return window_tensor, delta_theta_target, mask_tensor