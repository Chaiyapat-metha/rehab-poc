# File: .\backend\training\models\data\dataset.py

import sys
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from backend.training.models.models.augmentation import Augmentor 
from backend.app.config import load_config
from backend.app.utils.db import DatabaseManager

# Configuration Constants
WINDOW_SIZE = 16 # T = 16
SPLIT_RATIO = [0.7, 0.15, 0.15] # Train, Val, Test
V_JOINTS = 33 # Total MediaPipe joints

class MultiTaskPoseDataset(Dataset):
    """
    Loads data records (windows) for multi-task learning.
    Handles train/val/test split based on original video ID (ingest_uuid).
    """
    def __init__(self, exercise_id: str, split: str = 'train'):
        self.db_manager = DatabaseManager()
        self.config = load_config()
        self.exercise_id = exercise_id
        self.split = split
        self.exercise_cfg = self.config['exercises'].get(exercise_id, {})
        
        # Determine label requirements
        self.angle_output_order = self.exercise_cfg.get('angle_output_order', [])
        self.has_pos_label = 'regression_pos' in self.config['model_config']['model']['heads'][exercise_id]['outputs']
        self.has_class_label = 'classification' in self.config['model_config']['model']['heads'][exercise_id]['outputs']

        # โหลด metadata และทำ Split
        self._load_and_split_metadata()
        
        # Instanciate augmentor for random masking
        self.augmentor = Augmentor() 
        self.global_cfg = self.augmentor.global_cfg

    def _load_and_split_metadata(self):
        """
        Loads all unique ingest sessions (UUIDs) and performs a Stratified Split 
        to ensure all classes are represented in the validation set.
        """
        # 1. Fetch metadata: (ingest_uuid, frame_idx, label_class)
        all_metadata = self.db_manager._fetch_all_ingest_metadata(self.exercise_id) 
        
        # 2. จัดกลุ่ม UUIDs ตาม Label Class (Stratified Unit)
        uuid_by_class = {}
        for class_label, group in all_metadata.groupby('label_class'):
            # Grouping key คือ UUIDs
            uuids = group['ingest_uuid'].unique().tolist()
            random.shuffle(uuids)
            uuid_by_class[class_label] = uuids

        train_ids, val_ids, test_ids = [], [], []

        # 3. ทำ Stratified Split (แบ่งสัดส่วนตาม UUID)
        for class_label, uids in uuid_by_class.items():
            n_total = len(uids)
            n_train = int(n_total * SPLIT_RATIO[0])
            n_val = int(n_total * SPLIT_RATIO[1])

            train_ids.extend(uids[:n_train])
            val_ids.extend(uids[n_train:n_train + n_val])
            test_ids.extend(uids[n_train + n_val:])
        
        # 4. เลือก UUIDs ที่ถูกต้องสำหรับ Split นี้
        if self.split == 'train':
            split_ids = train_ids
        elif self.split == 'val':
            
            # --- MULTICLASS VALIDATION SET ENFORCEMENT ---
            all_known_labels = sorted(all_metadata['label_class'].unique())
            val_set = set(val_ids) 
            
            for required_label in all_known_labels:
                # ค้นหา UUIDs ทั้งหมดที่มี Class นี้
                uuids_with_label = all_metadata[all_metadata['label_class'] == required_label]['ingest_uuid'].unique()
                
                # ตรวจสอบว่า Class นี้มีตัวแทนอยู่ใน Val Set หรือไม่
                if not val_set.intersection(uuids_with_label):
                    
                    # ถ้าขาด: ค้นหา UUID ที่มี Class นี้ซึ่งปัจจุบันอยู่ใน Train Set
                    c_in_train = list(set(uuids_with_label).intersection(train_ids))
                    
                    if c_in_train:
                        # ย้าย UUID ตัวแรกมาใส่ Val Set
                        uuid_to_move = c_in_train[0] 
                        
                        train_ids.remove(uuid_to_move)
                        val_ids.append(uuid_to_move)
                        val_set.add(uuid_to_move)
                        
                        print(f"DEBUG: MOVED UUID {uuid_to_move[:8]} (Class {required_label}) from TRAIN to VAL for balance.")
                    else:
                        print(f"WARNING: No available UUID for Class {required_label} in TRAIN set to balance VAL.")

            split_ids = val_ids # ใช้ val_ids ที่ถูกแก้ไขแล้ว
            
        elif self.split == 'test':
            split_ids = test_ids
        else:
            raise ValueError("Invalid split type.")
        
        random.shuffle(split_ids) 

        # 5. กรอง metadata ที่ใช้สำหรับ Windowing (ใช้ UUIDs)
        self.metadata = all_metadata[all_metadata['ingest_uuid'].isin(split_ids)]
        
        # 6. สร้าง self.window_indices (Window Start Indices)
        self.window_indices = []
        for uuid in split_ids:
            # Note: ใช้ self.metadata ที่ถูกกรองแล้ว
            session_meta = self.metadata[self.metadata['ingest_uuid'] == uuid].sort_values('frame_idx')

            min_frame = session_meta['frame_idx'].min()
            max_frame = session_meta['frame_idx'].max()
            
            # Window starts at frame_idx that allows for a full WINDOW_SIZE sequence
            for start_frame in range(min_frame, max_frame - WINDOW_SIZE + 2):
                if start_frame + WINDOW_SIZE <= max_frame + 1:
                    self.window_indices.append((uuid, start_frame))
                    
        if not self.window_indices:
             print(f"Warning: No valid windows (T={WINDOW_SIZE}) found for split: {self.split}")
             
    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        ingest_uuid, start_frame = self.window_indices[idx]

        # 1. Fetch Window Data
        frame_records = self.db_manager.fetch_window_data(ingest_uuid, start_frame, WINDOW_SIZE)
        
        # 💡 SKIP LOGIC: ถ้า Window ไม่ครบตามขนาด ให้ส่ง None (Collate Function จะกรองออก)
        if len(frame_records) != WINDOW_SIZE:
             return None, None 

        # 2. Input Tensor (T, V, C)
        keypoints_list = [torch.from_numpy(r['joints_array']).float().unsqueeze(0) for r in frame_records]
        keypoints_tensor = torch.cat(keypoints_list, dim=0) 
        
        # 3. Random Masking (Occlusion)
        if random.random() < self.global_cfg.get('prob_occlude', 0.0):
            num_occlude = self.global_cfg.get('occlude_joints_count', 5)
            occlude_indices = random.sample(range(V_JOINTS), num_occlude)
            keypoints_tensor[:, occlude_indices, :] = 0.0 

        # 4. Aggregate Targets (ใช้ข้อมูลจากเฟรมสุดท้ายของ Window)
        last_record = frame_records[-1]
        N_angles = len(self.angle_output_order)
    
        # 4a. Classification Target
        target_class = last_record.get('label_class')
        final_class_target = torch.tensor([target_class], dtype=torch.long) if target_class is not None else None
    
        # 4b. Angle Regression Target
        angles_vec = last_record.get('label_angles_vector')
        angles_tensor = torch.tensor(
            [a if a is not None else np.nan for a in angles_vec or [None] * N_angles], 
            dtype=torch.float
        )
        final_angles_target = angles_tensor 
    
        # 4c. Positional Regression Target
        pos_vec = last_record.get('label_pos_vector')
        pos_tensor = torch.tensor(
            pos_vec if pos_vec is not None else [np.nan] * 99, 
            dtype=torch.float
        )
        final_pos_target = pos_tensor 

        # 5. Final Target Tensors (1 Label ต่อ Window)
        final_targets = {
            'target_class': final_class_target,
            'target_angles': final_angles_target,
            'target_pos': final_pos_target, 
            'ingest_uuid': ingest_uuid,
            'exercise_id': self.exercise_id
        }

        # Input: (T, V, C), Output: (1 Target, N dims)
        return keypoints_tensor, final_targets
    
def window_collate_fn(batch: List[Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]]):
    """
    Collate function for windowed data (T, V, C) to create (B, T, V, C) batch.
    Filters out incomplete samples (None, None).
    """
    # 💡 FILTERING: กรองตัวอย่างที่เป็น (None, None) ออก
    batch = [item for item in batch if item[0] is not None]
    
    if not batch:
        return torch.tensor([]), {} 
    
    data_list = [item[0].unsqueeze(0) for item in batch] # (1, T, V, C)
    targets_list = [item[1] for item in batch]
    
    # 1. Input Data: (B, T, V, C)
    batch_data = torch.cat(data_list, dim=0) 
    
    # 2. Aggregate Targets (ต้องจัดการ Targets ที่เป็น None)
    
    # Helper function to safely concatenate tensors while handling None/empty tensors
    def safe_cat_targets(key: str):
        tensors = [t.get(key) for t in targets_list if t.get(key) is not None]
        if not tensors:
            return None
        # For targets, we need to add a dimension for the batch (B) if it's not already there
        tensors = [t.unsqueeze(0) if t.dim() == 1 else t for t in tensors]
        
        return torch.cat(tensors, dim=0)

    aggregated_targets = {
        'target_class': safe_cat_targets('target_class'),
        'target_angles': safe_cat_targets('target_angles'),
        'target_pos': safe_cat_targets('target_pos'),
        'ingest_uuid': [t['ingest_uuid'] for t in targets_list], 
        'exercise_id': [t['exercise_id'] for t in targets_list] 
    }
    
    return batch_data, aggregated_targets