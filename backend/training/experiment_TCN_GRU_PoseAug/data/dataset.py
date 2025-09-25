# rehab-poc/backend/training/experiment_TCN_GRU_PoseAug/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import psycopg2

from models.pose_augmenter import initialize_pose_augmenter 

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
def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

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
    def __init__(self, db_config, config):
        self.config = config
        self.window_size = config['dataset']['window_size']
        self.joints = config['dataset']['joints']
        self.input_dim = config['dataset']['input_dim']
        
        try:
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")

        self.cursor.execute("""
            SELECT keypoints, label
            FROM training_skeletons
            ORDER BY video_id, frame_no ASC;
        """)

        all_data = self.cursor.fetchall()
        
        label_list = [1.0 if item[1] == 'wrong' else 0.0 for item in all_data]
        self.labels = np.array(label_list, dtype=np.float32)
        
        if not all_data:
            raise ValueError("No data found in the training_skeletons table.")
        
        keypoints_list = [item[0] for item in all_data] 
        self.keypoints = np.array(keypoints_list, dtype=np.float32).reshape(-1, self.joints, self.input_dim)
        
        self.num_frames = self.keypoints.shape[0]
        
        self.windows = []
        stride = config['dataset']['stride']
        for i in range(0, self.num_frames - self.window_size + 1, stride):
            self.windows.append(self.keypoints[i:i + self.window_size, :, :])

        # --- PoseAug Setup (New Implementation) ---
        # 1. Check if ANY augmentation is enabled (used for loading the augmenter)
        if config['augmentation']['poseaug']['enabled'] or config['augmentation'].get('sweep_aug_status', False):
            self.pose_augmenter = initialize_pose_augmenter(config)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.pose_augmenter = None
               
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 1. เตรียมข้อมูล: ดึง window ดิบออกมา
        window_np = self.windows[idx].copy()
        
        # 2. แปลงเป็น Tensor เพื่อเข้าสู่ PoseAugmenter
        # Shape: (1, T, J, C)
        window_tensor = torch.from_numpy(window_np).float().unsqueeze(0)
        
        # --- Logic การ Augmentation (ใช้แค่ชุดนี้) ---
        is_aug_enabled = self.config['augmentation']['poseaug']['enabled']
        
        if is_aug_enabled and self.pose_augmenter is not None:
            
            # ส่งไป GPU (ถ้ามี)
            device = self.device if hasattr(self, 'device') else torch.device('cpu')
            window_tensor = window_tensor.to(device)
            
            # Apply PoseAug และเก็บผลลัพธ์
            window_tensor = self.pose_augmenter(window_tensor)
            
            # ย้ายกลับมา CPU/NumPy เพื่อคำนวณ Target และคืนค่า
            window_np_final = window_tensor.squeeze(0).cpu().numpy()
        else:
            # ถ้า disabled ก็ใช้ข้อมูลดิบ
            window_np_final = window_np

        # 3. คำนวณ Targets จากข้อมูลสุดท้ายที่ผ่าน Augmentation แล้ว (window_np_final)
        
        # คำนวณ Index ของเฟรมสุดท้ายใน Window ที่เป็น Target
        last_frame_index = idx * self.config['dataset']['stride'] + self.config['dataset']['window_size'] - 1

        # Target L_pos: 99 มิติ
        target_pos = window_np_final[-1, :, :].reshape(-1)
        
        # Target L_angle: 6 มิติ
        target_angles = get_6_angles(window_np_final[-1, :, :])
        
        # Target L_class (สำหรับ Multi-task Classification)
        target_class = self.labels[last_frame_index]
        
        # 4. คืนค่า
        # เราควรคืนค่า Input Tensor (window_tensor) ที่ใช้เทรน, และ Targets (NumPy-based)
        return (
            torch.from_numpy(window_np_final).float(),  # Input Data (Shape: T, J, C)
            torch.from_numpy(target_pos).float(),       # Target Pos
            torch.from_numpy(target_angles).float(),    # Target Angle
            torch.tensor(target_class).float()          # Target Class
        )

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()