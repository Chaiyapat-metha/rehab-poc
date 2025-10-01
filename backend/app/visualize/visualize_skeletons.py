# rehab-poc/backend/app/visualize/visualize_skeletons.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import psycopg2
import sys
from pathlib import Path
from typing import Optional

# Setup path to import from app module
# ใช้ Pathlib เพื่อจัดการ path ให้ถูกต้อง (ปรับให้เรียกใช้ db.py และ rehab_pb2)
sys.path.append(str(Path(__file__).resolve().parents[2])) 

# Import DatabaseManager และ Protobuf
from app.utils.db import DatabaseManager
from app.proto_generated import rehab_pb2

# --- Mediapipe Connections for 3D Plotting (Updated for full skeleton) ---
connections = [
    # Head and Shoulders (ส่วนบน)
    (0, 4), (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), # Face/Eyes
    (9, 10), # Mouth
    (1, 4), # Top Head (approximate center)
    (11, 12), # Shoulders
    (0, 10), (0, 9), # Nose to Mouth (often omitted, but added for completeness)
    
    # Core Trunk (เชื่อมจาก Shoulder ลง Hip)
    (11, 23), # Left Shoulder to Left Hip
    (12, 24), # Right Shoulder to Right Hip
    (23, 24), # Hips (เชื่อมสะโพกซ้าย-ขวา)

    # Arms (เหมือนเดิม)
    (11, 13), (13, 15), # Left Arm (Shoulder-Elbow-Wrist)
    (12, 14), (14, 16), # Right Arm

    # Legs (เหมือนเดิม)
    (23, 25), (25, 27), (27, 29), (29, 31), # Left Leg (Hip-Knee-Ankle-Heel-Toe)
    (24, 26), (26, 28), (28, 30), (30, 32),  # Right Leg
    
    # Feet (ปลายเท้า)
    (29, 31), (30, 32) # Toe to Heel connections (Indices might vary slightly depending on exact model)
]

# ----------------------------------------------------------------------
# HELPER: Deserialization Logic (นำมาจาก db.py/video_processor.py)
# ----------------------------------------------------------------------

def _frame_proto_to_numpy(joints_byte_data: bytes) -> Optional[np.ndarray]:
    """
    Deserializes the raw BYTEA data (Protobuf) from the DB into a (33, 3) NumPy array.
    """
    pose_message = rehab_pb2.Frame() 
    try:
        pose_message.ParseFromString(joints_byte_data)
    except Exception as e:
        print(f"Error deserializing Protobuf: {e}")
        return None

    V = len(pose_message.joints)
    if V == 0:
        return np.zeros((33, 3), dtype=np.float32)

    # Note: เราใช้ World Coordinates (x,y,z) ในการบันทึก
    data = np.zeros((33, 3), dtype=np.float32) 

    for joint in pose_message.joints:
        if joint.id < 33:
             data[joint.id, 0] = joint.x
             data[joint.id, 1] = joint.y
             data[joint.id, 2] = joint.z

    return data


# ----------------------------------------------------------------------
# DATABASE RETRIEVAL (อัปเดตให้ใช้ Schema ใหม่)
# ----------------------------------------------------------------------

def get_skeleton_data(ingest_uuid: str) -> Optional[np.ndarray]:
    """
    Fetches raw keypoint data for a specific ingest_uuid from the database.
    
    Args:
        ingest_uuid (str): The unique UUID of the video ingestion session.
        
    Returns:
        np.ndarray: A NumPy array of shape (num_frames, 33, 3).
    """
    db_manager = DatabaseManager() 
    
    # 💡 Query: ใช้ตาราง skeleton_data และคอลัมน์ ingest_uuid, joints_data
    sql = """
        SELECT joints_data
        FROM skeleton_data
        WHERE ingest_uuid = %s
        ORDER BY frame_idx ASC;
    """
    
    try:
        with db_manager.conn.cursor() as cursor:
            cursor.execute(sql, (ingest_uuid,))
            results = cursor.fetchall()

        if not results:
            print(f"Error: No data found for ingest UUID: {ingest_uuid}")
            return None
        
        # 💡 Deserialization: แปลง BYTEA (Protobuf) เป็น NumPy Array
        keypoints_list = []
        for row in results:
            joints_byte_data = row[0] # ดึง BYTEA
            frame_data = _frame_proto_to_numpy(joints_byte_data)
            if frame_data is not None:
                keypoints_list.append(frame_data)
        
        if not keypoints_list:
             print("Warning: All frames failed Protobuf deserialization.")
             return None
             
        # รวมเป็น (num_frames, 33, 3)
        keypoints_3d = np.stack(keypoints_list, axis=0) 
        
        print(f"✅ Successfully fetched {len(keypoints_3d)} frames from UUID: {ingest_uuid}")
        return keypoints_3d

    except Exception as e:
        print(f"❌ Database error: {e}")
        return None
    finally:
        db_manager.close()


# ----------------------------------------------------------------------
# VISUALIZATION FUNCTIONS (ใช้โค้ดเดิม)
# ----------------------------------------------------------------------

def plot_time_series(keypoints: np.ndarray, label: str):
    """Plots the time-series of X, Y, Z coordinates for all joints."""
    # ... (Logic เดิม, ไม่ต้องแก้ไข) ...
    print(f"-> Plotting time series for '{label}' video...")
    num_frames = keypoints.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    colors = plt.cm.get_cmap('tab20', 33)
    
    for i in range(33):
        axes[0].plot(keypoints[:, i, 0], alpha=0.5, color=colors(i), label=f"Joint {i}") # X
        axes[1].plot(keypoints[:, i, 1], alpha=0.5, color=colors(i), label=f"Joint {i}") # Y
        axes[2].plot(keypoints[:, i, 2], alpha=0.5, color=colors(i), label=f"Joint {i}") # Z
    
    axes[0].set_title('X Coordinate Over Time')
    axes[1].set_title('Y Coordinate Over Time')
    axes[2].set_title('Z Coordinate Over Time')
    axes[2].set_xlabel('Frame Number')
    
    fig.suptitle(f"Time-Series Plot for '{label}' video", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.show()

def animate_3d_skeleton(keypoints: np.ndarray, label: str):
    """Creates an animated 3D plot of the skeleton."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    all_coords = keypoints.reshape(-1, 3)
    ax.set_xlim([all_coords[:, 0].min(), all_coords[:, 0].max()])
    ax.set_ylim([all_coords[:, 1].min(), all_coords[:, 1].max()])
    ax.set_zlim([all_coords[:, 2].min(), all_coords[:, 2].max()])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Skeleton Animation for '{label}' video")
    
    points, = ax.plot([], [], [], 'o', color='blue')
    lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]

    def update(frame_idx):
        frame_data = keypoints[frame_idx]
        
        points._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        for i, connection in enumerate(connections):
            p1_idx, p2_idx = connection
            p1 = frame_data[p1_idx]
            p2 = frame_data[p2_idx]
            lines[i].set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
            
        return points, *lines

    num_frames = keypoints.shape[0]
    ani = FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=True)
    
    plt.show()

def main():
    """Main function to visualize a specific video."""
    # 💡 อัปเดต: ใช้ ingest_uuid ที่คุณเพิ่งบันทึกไป
    # คุณต้องแทนที่ UUID ตัวอย่างนี้ด้วย UUID จริงจาก DB ของคุณ
    ingest_uuid = '326e32ed-96fd-4adb-8975-a0f80c481bad' 
    label = 'Jump_squats_correct' 
    
    keypoints_data = get_skeleton_data(ingest_uuid)
    if keypoints_data is not None:
        plot_time_series(keypoints_data, label)
        animate_3d_skeleton(keypoints_data, label) 

if __name__ == '__main__':
    # สำหรับ Matplotlib 3D animation ต้องใช้ backend ที่เหมาะสม
    # plt.rcParams['animation.ffmpeg_args'] = ['-filter:v', 'crop=in_h:in_h', '-threads', '1']
    main()