# File: .\backend\training\models\models\augmentation.py

import numpy as np
import torch
import random
from typing import Dict, Any, Tuple, Optional, List
from backend.app.config import load_config

# Constants for coordinate indices (based on MediaPipe world coordinates)
X_IDX, Y_IDX, Z_IDX = 0, 1, 2
V_JOINTS = 33 # Total number of joints

class Augmentor:
    """
    Handles both global geometric augmentations and per-exercise logic 
    to create synthetic 'wrong' examples and update labels accordingly.
    """
    def __init__(self, seed: int = 42):
        self.config = load_config()
        self.global_cfg = self.config['augmentation'].get('global', {})
        self.per_exercise_cfg = self.config['augmentation'].get('per_exercise', {})
        random.seed(seed)
        np.random.seed(seed)
        
        self.JOINT_MAPPER = self._get_joint_mapper_static()

    @staticmethod
    def _get_joint_mapper_static() -> Dict[str, int]:
        """
        Provides a static mapping from exercise names (Upper Case) to MediaPipe indices.
        """
        return {
            'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            
            # Core Joints (ใช้ในการคำนวณ Angle)
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
        }

    # ----------------------------------------------------------------------
    # 1. Global Geometric Augmentations (Rotation, Jitter, Occlusion)
    # ----------------------------------------------------------------------

    def apply_global_augmentation(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Applies geometric augmentations (rotation, jitter, occlusion) 
        to the keypoints (T, V, C).
        """
        keypoints_aug = keypoints.copy()
        
        # 1. Rotation (Rotate around Y-axis / vertical axis)
        if random.random() < self.global_cfg.get('prob_rotate', 0.0):
            deg = random.uniform(*self.global_cfg.get('rotation_deg', [-15, 15]))
            rad = np.deg2rad(deg)
            # 2D Rotation matrix in XZ plane (around Y-axis)
            R = np.array([
                [np.cos(rad), 0, np.sin(rad)],
                [0, 1, 0],
                [-np.sin(rad), 0, np.cos(rad)]
            ])
            keypoints_aug = np.dot(keypoints_aug, R.T)
            
        # 2. Jitter (Add Gaussian noise)
        if random.random() < self.global_cfg.get('prob_jitter', 0.0):
            sigma = self.global_cfg.get('jitter_sigma_norm', 0.005)
            noise = np.random.normal(0, sigma, size=keypoints_aug.shape)
            keypoints_aug += noise
            
        # 3. Occlusion (Setting visibility/coords to zero - needs visibility info)
        # Note: เราไม่มี visibility channel ใน input (V, 3) แต่เราสามารถเซ็ตพิกัดเป็น 0 ได้
        if random.random() < self.global_cfg.get('prob_occlude', 0.0):
            num_occlude = self.global_cfg.get('occlude_joints_count', 5)
            # สุ่ม Joints ที่จะ Occlude
            occlude_indices = random.sample(range(V_JOINTS), num_occlude)
            keypoints_aug[:, occlude_indices, :] = 0.0 # Set coordinates to zero

        # NOTE: เมื่อมีการหมุน (Rotation) ต้องมั่นใจว่า Ground Truth Labels (Angles) 
        # จะต้องถูกคำนวณใหม่หากจำเป็น
        return keypoints_aug

    # ----------------------------------------------------------------------
    # 2. Per-Exercise Augmentation (Synthetic Wrong Examples)
    # ----------------------------------------------------------------------

    def apply_per_exercise_augmentation(self, 
                                        keypoints: np.ndarray, 
                                        exercise_id: str, 
                                        original_labels: Dict[str, Any]
                                        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies specific rules to create synthetic 'wrong' examples 
        and updates classification/regression labels.
        """
        keypoints_aug = keypoints.copy()
        updated_labels = original_labels.copy()
        
        if exercise_id not in self.per_exercise_cfg:
            return keypoints_aug, updated_labels

        cfg = self.per_exercise_cfg[exercise_id]
        
        # --- Jump Squat Rules ---
        if exercise_id == 'Jump_squats':
            # Simulate insufficient depth (shallower squat)
            if random.random() < cfg.get('prob_add_knee_bias', 0.0):
                deg_bias = random.uniform(*cfg.get('knee_bias_deg_range', [0, 0])) # Negative is shallower
                
                # Logic: เพิ่มค่า Y (ยกขึ้น) ให้กับสะโพกและเข่า เพื่อทำให้การงอน้อยลง
                y_offset = np.abs(deg_bias) * 0.01 
                
                hip_indices = [self.JOINT_MAPPER['LEFT_HIP'], self.JOINT_MAPPER['RIGHT_HIP']]
                knee_indices = [self.JOINT_MAPPER['LEFT_KNEE'], self.JOINT_MAPPER['RIGHT_KNEE']]

                # Apply offset to Hips and Knees (y-coord)
                keypoints_aug[:, hip_indices, Y_IDX] += y_offset
                keypoints_aug[:, knee_indices, Y_IDX] += y_offset
                
                # Set classification label to 'wrong_shallow_depth' (1)
                updated_labels['label_class'] = cfg.get('target_class_bias', 1) 
                updated_labels['is_synthetic_wrong'] = True

        # --- Stretching Upper Trapezius Rules ---
        elif exercise_id == 'Stretching_upper_trapezius':
             # Offset shoulder height (left_high or right_high)
            if random.random() < cfg.get('prob_shoulder_level_offset', 0.0):
                mm_offset = random.uniform(*cfg.get('shoulder_level_offset_mm_range', [0, 0])) / 1000.0 # mm to meters
                target_classes = cfg.get('target_class_offset', [1, 2])

                # Randomly select left or right high (class 1 or 2)
                class_label = random.choice(target_classes)
                
                if class_label == 1: # Left High
                    keypoints_aug[:, self.JOINT_MAPPER['LEFT_SHOULDER'], Y_IDX] += mm_offset
                elif class_label == 2: # Right High
                    keypoints_aug[:, self.JOINT_MAPPER['RIGHT_SHOULDER'], Y_IDX] += mm_offset

                updated_labels['label_class'] = class_label
                updated_labels['is_synthetic_wrong'] = True

        # ... (เพิ่ม Rules สำหรับท่าอื่นๆ) ...
        
        return keypoints_aug, updated_labels


# ----------------------------------------------------------------------
# 3. Data Flow Integration (Update video_processor.py)
# ----------------------------------------------------------------------

# NOTE: Augmentation.py ควรถูกใช้ใน:
# 1. Training loop (on-the-fly augmentation) 
# 2. Ingestion script (เพื่อสร้าง wrong examples ถาวรใน DB)

# **ถ้าใช้สำหรับ Ingestion (เพื่อสร้าง wrong samples ถาวรใน DB):**
# ต้องสร้างสคริปต์ใหม่ชื่อ augment_ingest.py ที่วนลูปดึง Correct data จาก DB, 
# เรียก Augmentor.apply_per_exercise_augmentation, 
# คำนวณ Labels ใหม่, และบันทึกกลับเข้า DB ด้วย ingest_skeleton_data_batch/ingest_label_data_batch