import json
from tqdm import tqdm

# Setup path to import from app module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils import db, feature_engineering

# --- Define Error Types (Classification Labels) ---
# This mapping must be consistent across the project
ERROR_CLASSES = {
    "CORRECT": 0,
    "LEFT_ELBOW_ANGLE_LOW": 1,
    "RIGHT_ELBOW_ANGLE_LOW": 2,
    "LEFT_KNEE_ANGLE_LOW": 3,
    "RIGHT_KNEE_ANGLE_LOW": 4,
}
# A reverse map for easier debugging
CLASS_NAMES = {v: k for k, v in ERROR_CLASSES.items()}

# --- Define Simple Rules for a Specific Exercise ---
# EXAMPLE for "Incline rows with dumbbell"
# These thresholds are examples and need to be tuned.
EXERCISE_RULES = {
    'Incline rows with dumbbell': {
        'left_elbow_angle': {'min': 90, 'max': 160},
        'right_elbow_angle': {'min': 90, 'max': 160},
    }
    # TODO: Add rules for other exercises
}

def apply_rules(exercise_name: str, angles: dict) -> dict:
    """Applies predefined rules to a frame's angles and returns labels."""
    rules = EXERCISE_RULES.get(exercise_name)
    if not rules:
        return {"class": ERROR_CLASSES["CORRECT"], "severity": {}}

    # Default to correct
    label = {"class": ERROR_CLASSES["CORRECT"], "severity": {}}

    # Check Left Elbow
    if 'left_elbow_angle' in rules:
        angle = angles.get('left_elbow_angle', 180) # Default to a safe value
        if angle < rules['left_elbow_angle']['min']:
            label['class'] = ERROR_CLASSES['LEFT_ELBOW_ANGLE_LOW']
            label['severity']['left_elbow_angle'] = rules['left_elbow_angle']['min'] - angle
            return label # Return on first error found for simplicity

    # Check Right Elbow
    if 'right_elbow_angle' in rules:
        angle = angles.get('right_elbow_angle', 180)
        if angle < rules['right_elbow_angle']['min']:
            label['class'] = ERROR_CLASSES['RIGHT_ELBOW_ANGLE_LOW']
            label['severity']['right_elbow_angle'] = rules['right_elbow_angle']['min'] - angle
            return label

    return label

def main():
    print("Starting Auto-Labeling Process...")
    
    # 1. Get all frames that have not been labeled yet
    sql_fetch = """
    SELECT f.time, f.session_id, f.raw_frame_data, e.name as exercise_name
    FROM frames f
    JOIN sessions s ON f.session_id = s.session_id
    JOIN exercises e ON s.exercise_name = e.name
    WHERE f.labels IS NULL
    ORDER BY f.time;
    """
    
    sql_update = """
    UPDATE frames SET labels = %s WHERE time = %s AND session_id = %s;
    """

    with db.get_db_connection() as conn:
        with conn.cursor() as cur:
            print("Fetching unlabeled frames from database...")
            cur.execute(sql_fetch)
            unlabeled_frames = cur.fetchall()
            print(f"Found {len(unlabeled_frames)} frames to label.")

            if not unlabeled_frames:
                print("No new frames to label. Exiting.")
                return

            # 2. Process and update each frame
            for frame_data in tqdm(unlabeled_frames, desc="Labeling Frames"):
                time, session_id, raw_frame_data, exercise_name = frame_data
                
                # Deserialize protobuf to get keypoints
                frame_proto = db.rehab_pb2.Frame()
                frame_proto.ParseFromString(raw_frame_data)

                # Get angles needed for rule checking
                # We reuse the functions from feature_engineering
                angles = feature_engineering.get_angles_from_proto(frame_proto)

                # Apply rules to get the label
                label_dict = apply_rules(exercise_name, angles)

                # Update the frame in the database with the new label
                cur.execute(sql_update, (json.dumps(label_dict), time, session_id))
            
            # Commit all changes
            conn.commit()
    
    print(f"✅ Auto-labeling complete. {len(unlabeled_frames)} frames updated.")


if __name__ == "__main__":
    main()
