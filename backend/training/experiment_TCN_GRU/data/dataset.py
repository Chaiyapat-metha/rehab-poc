# data/dataset.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import psycopg2
#import struct 

class PoseDataset(Dataset):
    def __init__(self, db_config, exercise_name, config):
        """
        Initializes the dataset by loading all relevant data from the database.
        No train/eval/test split is done here.
        """
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        
        # Connect to the database
        try:
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

        # Query to fetch all frames for a specific exercise_name
        self.cursor.execute("""
            SELECT f.feature_vector, f.labels
            FROM frames AS f
            JOIN sessions AS s ON f.session_id = s.session_id
            WHERE s.exercise_name = %s
            ORDER BY f."time" ASC;
        """, (exercise_name,))
        
        self.all_data = self.cursor.fetchall()
        
        if not self.all_data:
            raise ValueError(f"No data found for exercise '{exercise_name}'.")
        
        # Separate features and labels
        feature_vectors = np.array([item[0] for item in self.all_data], dtype=np.float32)
        angle_labels = np.array([item[1] for item in self.all_data], dtype=object)

        # Reshape features to a 2D array (num_frames, input_dim)
        expected_size = self.input_dim
        if feature_vectors.size % expected_size != 0:
            valid_size = (feature_vectors.size // expected_size) * expected_size
            feature_vectors = feature_vectors[:valid_size]
            angle_labels = angle_labels[:valid_size]
        
        self.features = feature_vectors.reshape(-1, expected_size)
        
        # IMPORTANT: Extract and process the angle labels for our new regression task
        # We need to map the JSONB data to a structured numpy array for training
        # This is a placeholder as the exact JSONB format is unknown.
        # We will assume labels['angle_error'] contains a list of 6 values.
        # NOTE: This part needs careful implementation based on your database schema
        processed_labels = []
        for label_json in angle_labels:
            try:
                # Assuming the JSONB object has a key like 'joint_angles_diff'
                angles_diff = label_json.get('joint_angles_diff', [0.0] * 6)
                processed_labels.append(angles_diff)
            except Exception as e:
                # Fallback to zeros if parsing fails
                processed_labels.append([0.0] * 6)
        
        self.labels = np.array(processed_labels, dtype=np.float32)
        
        # Ensure features and labels have the same number of frames
        min_frames = min(self.features.shape[0], self.labels.shape[0])
        self.features = self.features[:min_frames]
        self.labels = self.labels[:min_frames]
        
        self.num_frames = self.features.shape[0]
        self.num_windows = self.num_frames // self.window_size

        if self.num_windows == 0:
            raise ValueError("Not enough data to create a single window.")
        
    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        
        window_features = self.features[start_idx:end_idx, :]
        target_labels = self.labels[end_idx-1, :]
        
        # The labels from the DB are 6 normalized angle differences (0-1)
        # We convert them to a binary vector based on a threshold
        threshold = 0.5 # Example threshold. This should be tuned.
        binary_labels = (target_labels > threshold).astype(np.float32)
        return torch.from_numpy(window_features).float(), torch.from_numpy(binary_labels).float()
  
    # # Preprocessing and Augmentation methods remain the same
    # def _preprocess(self, window):
    #     pelvis_idx = 23 # Right hip
    #     left_shoulder_idx = 11
    #     right_shoulder_idx = 12
        
    #     pelvis = window[:, pelvis_idx, :]
        
    #     normalized_window = window - pelvis[:, None, :]
    #     shoulder_dist = np.linalg.norm(
    #         normalized_window[:, left_shoulder_idx, :] - normalized_window[:, right_shoulder_idx, :], axis=-1
    #     )
    #     normalized_window = normalized_window / (shoulder_dist[:, None, None] + 1e-8)
        
    #     return normalized_window
        
    # def _augment(self, window):
    #     """
    #     Applies basic data augmentation to the skeleton window.
    #     - Random Rotation around Z-axis
    #     - Random Jittering of joints
    #     - Random Occlusion masking of joints
    #     """
    #     augmented_window = window.copy()

    #     # 1. Random Rotation (around Z-axis, which is the axis pointing out of the camera)
    #     rot_z_deg = self.config['augmentation']['basic']['rot_z_deg']
    #     if rot_z_deg > 0:
    #         angle_rad = np.radians(random.uniform(-rot_z_deg, rot_z_deg))
            
    #         # 2D Rotation matrix on X and Y axes
    #         rot_matrix = np.array([
    #             [math.cos(angle_rad), -math.sin(angle_rad), 0],
    #             [math.sin(angle_rad), math.cos(angle_rad), 0],
    #             [0, 0, 1]
    #         ])
    #         augmented_window = np.einsum('fji,ik->fjk', augmented_window, rot_matrix)
        
    #     # 2. Random Jittering (adding small noise to each joint)
    #     jitter_std = self.config['augmentation']['basic']['jitter_std']
    #     if jitter_std > 0:
    #         jitter_noise = np.random.normal(0, jitter_std, augmented_window.shape)
    #         augmented_window += jitter_noise

    #     # 3. Random Occlusion Masking
    #     p_mask = self.config['augmentation']['occlusion']['p_mask']
    #     if p_mask > 0:
    #         num_joints_to_mask = int(self.joints * p_mask)
    #         joints_to_mask = random.sample(range(self.joints), num_joints_to_mask)
            
    #         # Masking by setting the joint coordinates to zero
    #         augmented_window[:, joints_to_mask, :] = 0.0

    #     return augmented_window

    def _preprocess(self, window):
        # We will not run this for now as it's designed for 3D coordinates.
        return window
        
    def _augment(self, window):
        # We will not run this for now.
        return window
    
    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()