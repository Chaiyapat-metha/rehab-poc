"""
องค์ประกอบหลักของ Feature Vector (175 features)
Feature Vector ขนาด 175 นี้เกิดจากการรวมข้อมูล 5 กลุ่มหลักเข้าด้วยกันครับ:

Joint Angles (6 features): โค้ดจะคำนวณมุมสำคัญของร่างกาย เช่น มุมข้อศอกและหัวเข่าทั้งสองข้าง ฟีเจอร์นี้ช่วยให้โมเดลเข้าใจองศาการงอของข้อต่อได้โดยตรง ซึ่งสำคัญมากสำหรับท่าออกกำลังกาย

Bone Vectors (54 features): เป็นฟีเจอร์ที่แสดงถึงทิศทางและความยาวของกระดูกแต่ละชิ้น (เช่น จากไหล่ไปข้อศอก หรือจากสะโพกไปหัวเข่า) โดยใช้เวกเตอร์ 3 มิติ (18 bones x 3 coords = 54 features)

Raw Keypoint Coordinates (99 features): คือพิกัด X, Y, Z ดิบของ keypoint ทั้งหมด 33 จุด (33 keypoints x 3 coords = 99 features) ฟีเจอร์นี้เป็นข้อมูลพื้นฐานที่ขาดไม่ได้

Symmetry Delta (1 feature): โค้ดจะคำนวณความแตกต่างของตำแหน่งระหว่างส่วนที่สมมาตรกันของร่างกาย เช่น ความห่างระหว่างข้อมือซ้ายและข้อมือขวา ฟีเจอร์นี้มีประโยชน์อย่างมากในการประเมินความสมมาตรของท่าทาง

Visibilities (33 features): เป็นค่าที่บอกว่า keypoint แต่ละจุดมองเห็นได้ชัดเจนแค่ไหน (เช่น 0.0-1.0) ฟีเจอร์นี้ช่วยให้โมเดลเข้าใจได้ว่าบางส่วนของร่างกายถูกบดบังไปหรือไม่

"""

import numpy as np
from typing import List, Dict, Tuple

from ..proto_generated import rehab_pb2

# --- BlazePose Keypoint Indices (from MediaPipe Pose) ---
# This list is comprehensive for all calculations.
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# --- Core Calculation Functions (Unchanged) ---
def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # (Implementation is the same)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def _get_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # (Implementation is the same)
    return p2 - p1


# --- NEW HELPER FUNCTION that auto_labeler needs ---
def get_angles_from_proto(frame_proto: rehab_pb2.Frame) -> Dict[str, float]:
    """
    Calculates only the joint angles from a frame_proto message.
    This is a dedicated function for the auto-labeler.
    Returns a dictionary of angle names to their values.
    """
    angles = {}
    try:
        # Convert protobuf joints to a numpy array for easier access
        keypoints = np.array([[j.x, j.y, j.z] for j in frame_proto.joints])

        # --- Calculate required angles ---
        # Left Arm
        angles['left_elbow_angle'] = _calculate_angle(keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST])
        angles['left_shoulder_angle'] = _calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW])
        
        # Right Arm
        angles['right_elbow_angle'] = _calculate_angle(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW], keypoints[RIGHT_WRIST])
        angles['right_shoulder_angle'] = _calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW])
        
        # Left Leg
        angles['left_knee_angle'] = _calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
        
        # Right Leg
        angles['right_knee_angle'] = _calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])

    except (IndexError, ValueError) as e:
        print(f"Warning: Could not calculate angles for a frame due to missing keypoints. Error: {e}")
        return {} # Return empty dict on failure
        
    return angles


# --- Main Feature Extraction Function (Refactored to use the new angle function) ---
def extract_features_from_frame(frame_proto: rehab_pb2.Frame) -> List[float]:
    """
    Calculates the full feature vector (175 features) from a frame_proto message.
    """
    all_features = []
    try:
        keypoints = np.array([[j.x, j.y, j.z] for j in frame_proto.joints])
        visibilities = np.array([j.visibility for j in frame_proto.joints])

        # 1. Joint Angles (10 features) - Reuse our new function
        angles_dict = get_angles_from_proto(frame_proto)
        angle_features = [
            angles_dict.get('left_elbow_angle', 0), angles_dict.get('left_shoulder_angle', 0),
            angles_dict.get('right_elbow_angle', 0), angles_dict.get('right_shoulder_angle', 0),
            angles_dict.get('left_knee_angle', 0), angles_dict.get('right_knee_angle', 0)
            # Add more angles here if needed, and remember to update EXPECTED_FEATURE_COUNT
        ]
        all_features.extend(angle_features)

        # 2. Bone Vectors (18 vectors * 3 coords = 54 features)
        # (Implementation is the same)
        bone_vectors = [
            _get_vector(keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW]),
            # ... add all other bone vectors here ...
            _get_vector(keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
        ]
        all_features.extend(np.array(bone_vectors).flatten())
        
        # 3. Raw Keypoint Coordinates (33 points * 3 coords = 99 features)
        all_features.extend(keypoints.flatten())

        # 4. Symmetry Deltas (placeholder, 1 feature)
        # (Implementation is the same)
        symmetry_delta = np.linalg.norm(keypoints[LEFT_WRIST] - keypoints[RIGHT_WRIST])
        all_features.append(symmetry_delta)

        # 5. Visibilities (33 features) - Add visibilities as features
        all_features.extend(visibilities)
        
    except Exception as e:
        print(f"Error extracting features, returning zero vector. Error: {e}")
        return [0.0] * 175 # Ensure a fixed size vector on error

    # Padding to ensure fixed length, in case some features failed
    EXPECTED_FEATURE_COUNT = 175 # Update this if you add more features
    if len(all_features) < EXPECTED_FEATURE_COUNT:
        all_features.extend([0.0] * (EXPECTED_FEATURE_COUNT - len(all_features)))
        
    return all_features[:EXPECTED_FEATURE_COUNT]
