# File: .\backend\app\utils\db.py

import os
import psycopg2
from psycopg2 import extras
import numpy as np 
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# 1. IMPORT PROTOC GENERATED FILES
from ..proto_generated import rehab_pb2 

# กำหนดค่าการเชื่อมต่อจาก Environment Variables หรือค่าเริ่มต้น
DB_NAME = os.getenv("POSTGRES_DB", "rehab_db")
DB_USER = os.getenv("POSTGRES_USER", "nonny")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nonny")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost") 
DB_PORT = os.getenv("POSTGRES_PORT", "5433")

class DatabaseManager:
    """
    Handles connections and CRUD operations for the TimescaleDB 'rehab_db'.
    Uses execute_batch for high-performance data ingestion.
    """
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            self.conn.autocommit = True  # ตั้งค่าเป็น True สำหรับการทำงานแบบไม่ต้อง commit บ่อยๆ (ระวังเรื่อง transaction)
            print("Database connection successful.")
        except psycopg2.OperationalError as e:
            print(f"Database connection failed: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    # ----------------------------------------------------------------------
    # 2. TRAINING DATA INGESTION (ปรับปรุงให้รองรับ Protobuf)
    # ----------------------------------------------------------------------

    def ingest_skeleton_data_batch(self, data_list: List[Dict[str, Any]]):
        """
        Ingests a batch of skeleton data.
        """
        query = """
            INSERT INTO skeleton_data (
                ingest_uuid, ingest_timestamp, frame_idx, 
                exercise_id, joints_data, video_id_original
            )
            VALUES (
                %(ingest_uuid)s, %(ingest_timestamp)s, %(frame_idx)s, 
                %(exercise_id)s, %(joints_data)s, %(video_id_original)s
            )
        """
        
        records_to_insert = []
        for d in data_list:
            # (Serialization logic เดิม, สมมติว่า joints_bytes ถูกกำหนดค่าใน d แล้ว)
            
            records_to_insert.append({
                'ingest_uuid': d['ingest_uuid'],       
                'ingest_timestamp': d['ingest_timestamp'],
                'frame_idx': d['frame_idx'],
                'exercise_id': d['exercise_id'],
                'joints_data': d['joints_data'], 
                'video_id_original': d['video_id_original']
            })
        
        with self.conn.cursor() as cursor:
            # ใช้ execute_batch เพื่อประสิทธิภาพสูงสุด
            extras.execute_batch(cursor, query, records_to_insert)
            print(f"Successfully ingested {len(records_to_insert)} skeleton records.")
            
    def ingest_label_data_batch(self, data_list: List[Dict[str, Any]]):
        """
        Ingests a batch of multi-task label data into the exercise_labels hypertable.
        Uses ingest_uuid and frame_idx as primary key components.
        """
        query = """
            INSERT INTO exercise_labels (
                ingest_uuid, ingest_timestamp, frame_idx, 
                exercise_id, label_class, label_angles_vector, label_pos_vector
            )
            VALUES (
                %(ingest_uuid)s, %(ingest_timestamp)s, %(frame_idx)s, 
                %(exercise_id)s, %(label_class)s, %(label_angles_vector)s, %(label_pos_vector)s
            )
        """
        
        records_to_insert = []
        for d in data_list:
            proto: rehab_pb2.TrainingLabels = d['label_proto']
            
            records_to_insert.append({
                'ingest_uuid': d['ingest_uuid'],
                'ingest_timestamp': d['ingest_timestamp'],
                'frame_idx': d['frame_idx'],
                'exercise_id': d['exercise_id'],
                'label_class': proto.label_class if proto.is_valid_for_training else None,
                'label_angles_vector': list(proto.label_angles_vector) if proto.is_valid_for_training else None,
                'label_pos_vector': list(proto.label_pos_vector) if proto.is_valid_for_training else None,
            })
        
        with self.conn.cursor() as cursor:
            extras.execute_batch(cursor, query, records_to_insert)
            print(f"Successfully ingested {len(records_to_insert)} label records.")


    # ----------------------------------------------------------------------
    # 3. TRAINING DATA RETRIEVAL (สำหรับ dataset.py)
    # ----------------------------------------------------------------------

    def fetch_correct_data(self, exercise_id: str) -> List[Dict[str, Any]]:
        """
        Fetches ALL 'correct' training data records for a specific exercise.
        Used by augment_ingest.py.
        """
                
        query = """
            SELECT
                s.joints_data,
                l.label_class,
                l.label_angles_vector,
                l.label_pos_vector,
                s.frame_idx 
            FROM skeleton_data s
            JOIN exercise_labels l 
                ON s.ingest_uuid = l.ingest_uuid AND s.frame_idx = l.frame_idx
            WHERE s.exercise_id = %s AND l.label_class = 0; -- ใช้ 0 ตรงๆ
        """
        
        with self.conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
            cursor.execute(query, (exercise_id,)) 
            raw_data = cursor.fetchall()
            
            processed_data = []
            for row in raw_data:
                joints_byte_data = row.pop('joints_data')
                
                # Deserialization Logic (ตามที่อยู่ใน fetch_training_data)
                pose_message = rehab_pb2.Frame()
                pose_message.ParseFromString(joints_byte_data)
                
                V = len(pose_message.joints)
                keypoints_array = np.zeros((V, 3), dtype=np.float32)
                for joint in pose_message.joints:
                    keypoints_array[joint.id, :] = [joint.x, joint.y, joint.z]
                    
                row['joints_array'] = keypoints_array
                processed_data.append(row)

            return processed_data
    
    def get_total_samples(self, exercise_id: str) -> int:
        """Counts total available labeled samples for an exercise."""
        query = "SELECT COUNT(*) FROM exercise_labels WHERE exercise_id = %s;"
        with self.conn.cursor() as cursor:
            cursor.execute(query, (exercise_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
    
    @staticmethod    
    def _frame_proto_to_numpy(joints_byte_data: bytes) -> Optional[np.ndarray]:
        """
        [HELPER] Converts serialized Protobuf byte data (BYTEA) into a (33, 3) NumPy array.
        """
        # 1. Deserialize byte data to Protobuf Message
        frame_message = rehab_pb2.Frame()
        try:
            frame_message.ParseFromString(joints_byte_data) 
        except Exception as e:
            print(f"Error parsing Protobuf byte data: {e}")
            return None

        # 2. Conversion Logic (Logic ที่เหลือเหมือนเดิม)
        V = len(frame_message.joints)
        if V == 0:
            return np.zeros((33, 3), dtype=np.float32)

        keypoints_array = np.zeros((33, 3), dtype=np.float32)
        for joint in frame_message.joints:
            if joint.id < 33:
                keypoints_array[joint.id, :] = [joint.x, joint.y, joint.z]

        return keypoints_array

    def _fetch_all_ingest_metadata(self, exercise_id: str) -> pd.DataFrame:
        """
        [Helper for dataset.py] Fetches all necessary metadata for train/val/test splitting.
        """
        query = """
            SELECT ingest_uuid::TEXT, frame_idx, label_class
            FROM exercise_labels
            WHERE exercise_id = %s
            ORDER BY ingest_uuid, frame_idx;
        """
        
        with self.conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
            cursor.execute(query, (exercise_id,))
            results = cursor.fetchall()
            
            df = pd.DataFrame(results)
            
            if 'ingest_uuid' not in df.columns:

                 if df.empty:
                      print("Warning: DataFrame is empty. No metadata found.")
                      return pd.DataFrame(columns=['ingest_uuid', 'frame_idx', 'label_class'])

                 raise KeyError("Pandas failed to assign 'ingest_uuid' column name from RealDictCursor results.")

            return df

    def fetch_window_data(self, ingest_uuid: str, start_frame_idx: int, window_size: int) -> List[Dict[str, Any]]:
        """
        Fetches a continuous sequence of frames and their labels for a time window (B, T, V, C).
        """
        end_frame_idx = start_frame_idx + window_size - 1
        
        query = """
            SELECT
                s.joints_data,
                l.label_class,
                l.label_angles_vector,
                l.label_pos_vector
            FROM skeleton_data s
            JOIN exercise_labels l 
                ON s.ingest_uuid = l.ingest_uuid AND s.frame_idx = l.frame_idx
            WHERE s.ingest_uuid = %s 
              AND s.frame_idx >= %s 
              AND s.frame_idx <= %s
            ORDER BY s.frame_idx ASC;
        """
        
        with self.conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
            cursor.execute(query, (ingest_uuid, start_frame_idx, end_frame_idx))
            raw_data = cursor.fetchall()
        
        processed_data = []
        for row in raw_data:
            joints_byte_data = row.pop('joints_data') # joints_byte_data คือ memoryview/bytes
            keypoints_array = self._frame_proto_to_numpy(joints_byte_data) 
            
            if keypoints_array is not None:
                row['joints_array'] = keypoints_array
                processed_data.append(row)

        return processed_data
    
    # ----------------------------------------------------------------------
    # 4. USER/THRESHOLD CONTROLLER FUNCTIONS
    # ----------------------------------------------------------------------

    def get_user_thresholds(self, user_id: str, exercise_id: str) -> Dict[str, float]:
        """Retrieves the current thresholds for all metrics of an exercise for a user."""
        query = """
            SELECT metric_name, current_threshold 
            FROM user_thresholds 
            WHERE user_id = %s AND exercise_id = %s;
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, (user_id, exercise_id))
            return {name: threshold for name, threshold in cursor.fetchall()}

    def update_user_thresholds(self, user_id: str, exercise_id: str, updates: Dict[str, float]): # 💡 ต้องรับ exercise_id
        """Updates multiple thresholds for a user (used by commit_thresholds in the controller)."""
        # 💡 FIX: ต้องมั่นใจว่าทุก record มี user_id และ exercise_id
        records = [
            {
                'user_id': user_id,
                'exercise_id': exercise_id, # เพิ่ม exercise_id เข้าไปใน record
                'metric_name': name,
                'current_threshold': value
            }
            for name, value in updates.items()
        ]
        
        query = """
            INSERT INTO user_thresholds (user_id, exercise_id, metric_name, current_threshold)
            VALUES (%(user_id)s, %(exercise_id)s, %(metric_name)s, %(current_threshold)s)
            ON CONFLICT (user_id, exercise_id, metric_name)
            DO UPDATE SET current_threshold = EXCLUDED.current_threshold;
        """
        with self.conn.cursor() as cursor:
            extras.execute_batch(cursor, query, records)
            
    def log_rep_result(self, record: Dict[str, Any]):
        """Logs a single repetition result for history tracking (used by ThresholdController)."""
        query = """
            INSERT INTO rep_history (user_id, exercise_id, rep_timestamp, is_success, metric_errors)
            VALUES (%(user_id)s, %(exercise_id)s, NOW(), %(is_success)s, %(metric_errors)s)
        """
        with self.conn.cursor() as cursor:
            record['metric_errors'] = psycopg2.Json(record['metric_errors']) 
            cursor.execute(query, record)
            
    def fetch_rep_history(self, user_id: str, exercise_id: str, window_k: int) -> List[Tuple[bool, Dict[str, float]]]:
        """Fetches the last K repetitions for success rate calculation."""
        query = """
            SELECT is_success, metric_errors
            FROM rep_history
            WHERE user_id = %s AND exercise_id = %s
            ORDER BY rep_timestamp DESC
            LIMIT %s;
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, (user_id, exercise_id, window_k))
            return cursor.fetchall()

    # ----------------------------------------------------------------------
    # 5. THRESHOLD PROPOSAL HISTORY
    # ----------------------------------------------------------------------
    def log_threshold_proposal(self, record: Dict[str, Any]):
        """Logs a threshold proposal result (used by ThresholdController)."""
        query = """
            INSERT INTO proposal_history (user_id, exercise_id, proposal_timestamp, proposed_thresholds)
            VALUES (%(user_id)s, %(exercise_id)s, NOW(), %(proposed_thresholds)s)
        """
        with self.conn.cursor() as cursor:
            record['proposed_thresholds'] = psycopg2.Json(record['proposed_thresholds'])
            cursor.execute(query, record)