# File: .\backend\scripts\video_processor.py

import cv2
import mediapipe as mp
import numpy as np
import math
from tqdm import tqdm 
from datetime import datetime
import uuid
from pathlib import Path
import sys
from typing import List, Dict, Optional

# Setup path to import from app module
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils.db import DatabaseManager
from app.proto_generated import rehab_pb2
from app.config import load_config
from training.models.models.augmentation import Augmentor 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VideoProcessor:
    def __init__(self, model_complexity=2):
        self.db_manager = DatabaseManager()
        self.config = load_config() 
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity, # 0=Lite, 1=Full, 2=Heavy/Large
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.skeleton_batch = []
        self.label_batch = []
        self.batch_size = 1000 # ขนาด Batch สำหรับ execute_batch

# ----------------------------------------------------------------------
# Helper Methods สำหรับการคำนวณ
# ----------------------------------------------------------------------

    def _get_joint_coords(self, results, joint_name: str) -> np.ndarray:
        """
        Retrieves normalized world coordinates (x, y, z) for a named joint.
        """
        landmarks = results.pose_world_landmarks.landmark
        try:
            lm_enum = mp_pose.PoseLandmark[joint_name]
            lm = landmarks[lm_enum.value]

            return np.array([lm.x, lm.y, lm.z])
        except (AttributeError, KeyError):
            # หากหา Joint Name ไม่เจอ (เช่น ชื่อไม่ตรงกับ MediaPipe Enum)
            print(f"Warning: Joint name '{joint_name}' not found in MediaPipe landmarks.")
            return np.zeros(3)


    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        vector_ba = p1 - p2
        vector_bc = p3 - p2
        
        cosine_angle = np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle_deg = np.degrees(np.arccos(cosine_angle))
        
        return angle_deg


    @staticmethod
    def calculate_angle_labels_from_array(keypoints_array: np.ndarray, exercise_id: str, config: Dict) -> List[Optional[float]]:
        """
        Calculates required angle and metric labels from a NumPy array (V, 3).
        This is the refactored version callable from augment_ingest.py.
        """
        # 1. โหลด config
        exercise_cfg = config['exercises'].get(exercise_id, {})
        if not exercise_cfg or keypoints_array.shape[0] != 33:
            return []
            
        metric_values = {}
        
        JOINT_MAPPER = Augmentor._get_joint_mapper_static() 

        for metric in exercise_cfg.get('metrics', []):
            name = metric['name']
            joints_list = metric['joints']
            
            try:
                if len(joints_list) == 3:
                    # Angle calculation (A-B-C)
                    p_coords = [keypoints_array[JOINT_MAPPER[j]] for j in joints_list]
                    value = VideoProcessor._calculate_angle(p_coords[0], p_coords[1], p_coords[2])
                    metric_values[name] = value
                    
                elif name == 'SHOULDER_LEVEL_DIFF' and len(joints_list) == 2:
                    # Vertical distance calculation (mm)
                    p_left = keypoints_array[JOINT_MAPPER[joints_list[0]]]
                    p_right = keypoints_array[JOINT_MAPPER[joints_list[1]]]
                    y_diff_m = abs(p_left[1] - p_right[1])
                    value = y_diff_m * 1000 # Convert meters to mm
                    metric_values[name] = value

            except Exception as e:
                print(f"Error calculating metric {name}: {e}") 
                metric_values[name] = np.nan 

        # 2. จัดเรียงผลลัพธ์ตาม 'angle_output_order'
        output_vector = []
        for name in exercise_cfg.get('angle_output_order', []):
            value = metric_values.get(name, np.nan)
            output_vector.append(float(value) if not np.isnan(value) else None) 

        return output_vector

    @staticmethod
    def _create_frame_proto_from_array(keypoints_array: np.ndarray) -> rehab_pb2.Frame:
        """
        Creates a rehab_pb2.Frame message from a NumPy array (V, C=3).
        """
        frame_proto = rehab_pb2.Frame()
        for i, (x, y, z) in enumerate(keypoints_array):
            joint = frame_proto.joints.add()
            joint.id = i
            joint.x = x
            joint.y = y
            joint.z = z
            joint.visibility = 1.0 # Assume visibility is full for calculated array
        return frame_proto

    def process_frame(self, frame, video_ingest_uuid: str, frame_idx: int, exercise_id: str, label_class: int, frame_ts: datetime, video_id_original: str):
        """
        Processes a single frame, calculates labels, and queues for batch ingestion.
        """
        # 1. MediaPipe Processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_world_landmarks:
            return

        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark], dtype=np.float32)
        
        # 2. สร้าง Skeleton Protobuf (joints_data)
        frame_proto = VideoProcessor._create_frame_proto_from_array(landmarks_array)
        joints_bytes = frame_proto.SerializeToString()# Serialize to bytes for DB

        # 3a. Angle Labels (Regression target)
        configs = load_config()
        angle_labels = VideoProcessor.calculate_angle_labels_from_array(landmarks_array, exercise_id, configs)
        
        # 3b. Position Labels (Regression target - 99 dim vector)
        pos_labels = landmarks_array.flatten().tolist()
        
        # 4. สร้าง Label Protobuf (ก่อน Queue)
        label_proto = rehab_pb2.TrainingLabels(
            exercise_id=exercise_id,
            label_class=label_class,
            label_angles_vector=angle_labels,
            label_pos_vector=pos_labels,
            is_valid_for_training=True
        )

        # 5. Queue Data for Batch Ingestion
        
        # 5a. Skeleton Data Batch 
        self.skeleton_batch.append({
            'ingest_uuid': video_ingest_uuid,
            'ingest_timestamp': frame_ts,
            'frame_idx': frame_idx,
            'exercise_id': exercise_id,
            'joints_data': joints_bytes,
            'video_id_original': video_id_original
        })

        # 5b. Label Data Batch 
        self.label_batch.append({
            'ingest_uuid': video_ingest_uuid,
            'ingest_timestamp': frame_ts,
            'frame_idx': frame_idx,
            'exercise_id': exercise_id,
            'label_proto': label_proto # ใช้ตัวแปรที่ถูกกำหนดค่าแล้ว
        })
        # 6. Flush if Batch is full
        if len(self.skeleton_batch) >= self.batch_size:
            self.flush_batches()


    def process_video(self, video_path: Path, exercise_id: str, label_class: int):
        """
        Processes an entire video file, showing a progress bar for frame processing.
        """
        video_ingest_uuid = str(uuid.uuid4())
        video_id = video_path.stem 
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"  Warning: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        with tqdm(total=total_frames, 
                  desc=f"      Frames ({video_path.name})", 
                  unit="frame", 
                  leave=False) as pbar:

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_ts = datetime.now() 
                self.process_frame(
                    frame, 
                    video_ingest_uuid, 
                    frame_idx, 
                    exercise_id, 
                    label_class, 
                    frame_ts,
                    video_id_original=video_id 
                )
                
                frame_idx += 1
                pbar.update(1) 
        cap.release()


    def flush_batches(self):
        """Ingests all queued data into the database."""
        if self.skeleton_batch:
            self.db_manager.ingest_skeleton_data_batch(self.skeleton_batch)
            self.skeleton_batch = []
            
        if self.label_batch:
            self.db_manager.ingest_label_data_batch(self.label_batch)
            self.label_batch = []

