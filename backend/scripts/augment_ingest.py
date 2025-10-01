# File: .\backend\scripts\augment_ingest.py

import uuid
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np

from backend.app.utils.db import DatabaseManager
from backend.training.models.models.augmentation import Augmentor
from backend.scripts.video_processor import VideoProcessor
from backend.app.proto_generated import rehab_pb2
from backend.app.config import load_config 

def augment_and_ingest_wrong_data(exercise_id: str, num_copies: int = 1):
    """
    Fetches 'correct' data for an exercise, applies per-exercise augmentation 
    to create 'wrong' samples, recalculates labels, and ingests them back into the DB.
    """
    db_manager = DatabaseManager()
    augmentor = Augmentor()
    processor = VideoProcessor() 
    configs = load_config()
    
    # 1. ดึงข้อมูล 'Correct' ทั้งหมดจาก DB
    print(f"Fetching correct data for {exercise_id}...")
    
    correct_records = db_manager.fetch_correct_data(exercise_id) 
    
    if not correct_records:
        print(f"No 'correct' data found for {exercise_id}. Skipping augmentation.")
        return

    print(f"Found {len(correct_records)} 'correct' frames. Generating {len(correct_records) * num_copies} synthetic wrong frames.")

    skeleton_batch = []
    label_batch = []
    
    for _ in range(num_copies):        
        for frame_idx, record in enumerate(tqdm(correct_records, desc=f"Augmenting Batch {_ + 1}")):
            
            keypoints_array = record['joints_array'] # Shape (V, 3)
            
            # 💡 NOTE: keypoints_array ใน record ถูกดึงมาจาก DB (V, 3) 
            # เราต้องเพิ่มมิติ T=1 ให้กับ Augmentor ถ้า Augmentor คาดหวัง (T, V, C)
            keypoints_array_tv = keypoints_array[np.newaxis, ...] # Shape (1, V, C)

            # 2. เตรียม Original Labels
            original_labels = {
                'label_class': record.get('label_class', 0), 
                'label_angles_vector': record.get('label_angles_vector', []),
                'label_pos_vector': record.get('label_pos_vector', []),
                'is_synthetic_wrong': False
            }

            # 3. Apply Per-Exercise Augmentation
            # keypoints_aug: (1, V, C), updated_labels: Dict
            keypoints_aug_tv, updated_labels = augmentor.apply_per_exercise_augmentation(
                keypoints_array_tv, # (T, V, C)
                exercise_id, 
                original_labels
            )
            
            # **ถ้าเฟรมนี้ถูก Augment เป็น 'wrong' ให้บันทึก**
            if updated_labels.get('is_synthetic_wrong', False):
                synthetic_uuid = str(uuid.uuid4())
                keypoints_aug = keypoints_aug_tv.squeeze(0) # กลับไป (V, C)
                
                # 4. คำนวณ Angles ใหม่จาก Skeleton ที่ถูก Augment
                new_angle_labels = VideoProcessor.calculate_angle_labels_from_array(
                    keypoints_aug, 
                    exercise_id, 
                    configs 
                )
                
                # 5. สร้าง Protobuf และ Queue Data
                frame_proto = VideoProcessor._create_frame_proto_from_array(keypoints_aug) 
                joints_bytes = frame_proto.SerializeToString()
                
                # 5a. Queue Skeleton Data
                frame_ts = datetime.now()
                frame_idx = record['frame_idx']
                
                skeleton_batch.append({
                    'ingest_uuid': synthetic_uuid,
                    'ingest_timestamp': frame_ts,
                    'frame_idx': 0,
                    'exercise_id': exercise_id,
                    'joints_data': joints_bytes,
                    'video_id_original': record.get('video_id_original', 'synthetic')
                })
                
                # 5b. Queue Label Data 
                label_proto = rehab_pb2.TrainingLabels(
                    exercise_id=exercise_id,
                    label_class=updated_labels['label_class'],
                    label_angles_vector=new_angle_labels, 
                    label_pos_vector=list(keypoints_aug.flatten()), 
                    is_valid_for_training=True
                )
                
                label_batch.append({
                    'ingest_uuid': synthetic_uuid,
                    'ingest_timestamp': frame_ts,
                    'frame_idx': 0,
                    'exercise_id': exercise_id,
                    'label_proto': label_proto
                })
                
    # 7. Flush Batches
    print("\nFlushing synthetic data batches...")
    db_manager.ingest_skeleton_data_batch(skeleton_batch)
    db_manager.ingest_label_data_batch(label_batch)
    print(f"Successfully ingested {len(skeleton_batch)} synthetic 'wrong' records for {exercise_id}.")


if __name__ == '__main__':
    augment_and_ingest_wrong_data('Jump_squats', num_copies=0) 
    augment_and_ingest_wrong_data('Stretching_upper_trapezius', num_copies=3)