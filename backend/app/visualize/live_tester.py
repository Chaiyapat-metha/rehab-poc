import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from collections import deque
import time
import sys
from pathlib import Path

# --- Configuration (Minimal needed for MediaPipe) ---
WINDOW_SIZE = 16
EXERCISE_ID = "Jump_squats"
FEATURE_DIM = 256
MODEL_ROOT = Path(__file__).resolve().parents[2] / "weights"

# --- ONNX Paths (UPDATED) ---
MODEL_PATHS = {
    'BACKBONE': str(MODEL_ROOT / "backbone.onnx"),
    'CLASS_HEAD': str(MODEL_ROOT / f"{EXERCISE_ID}_class.onnx"),
    'ANGLE_HEAD': str(MODEL_ROOT / f"{EXERCISE_ID}_angle.onnx"),
    'POS_HEAD': str(MODEL_ROOT / f"{EXERCISE_ID}_pos.onnx") # ADDED POS HEAD
}

# --- Global Sessions ---
onnx_sessions = {}

def load_onnx_models():
    """Enables and loads ONNX models for inference."""
    print("Loading ONNX models...")
    try:
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        onnx_sessions['backbone'] = ort.InferenceSession(MODEL_PATHS['BACKBONE'], sess_options=session_options)
        onnx_sessions['class_head'] = ort.InferenceSession(MODEL_PATHS['CLASS_HEAD'], sess_options=session_options)
        onnx_sessions['angle_head'] = ort.InferenceSession(MODEL_PATHS['ANGLE_HEAD'], sess_options=session_options)
        onnx_sessions['pos_head'] = ort.InferenceSession(MODEL_PATHS['POS_HEAD'], sess_options=session_options) # LOAD POS HEAD
        
        print(f"Models loaded successfully for {EXERCISE_ID}.")
        return True
    except Exception as e:
        print(f"Error loading ONNX models. Check file paths and training outputs: {e}")
        return False

def prepare_input_tensor(window_data: deque) -> np.ndarray:
    """
    Converts a deque of 3D keypoint arrays (T, V, C) into the required ONNX input format (1, T, V, C).
    """
    # Stack deque into a NumPy array (T, V, C)
    window_array = np.array(window_data, dtype=np.float32)
    
    # Reshape to (1, T, V, C) for ONNX batch size 1
    input_tensor = window_array[np.newaxis, :, :, :]
    
    return input_tensor

def run_inference(input_tensor: np.ndarray):
    """Runs the model and returns classification probability and angle vector."""
    
    # 1. Backbone Inference
    shared_feature = onnx_sessions['backbone'].run(
        None, {"input_data": input_tensor}
    )[0].astype(np.float32)

    # 2. Head Inferences (Run all heads that share the feature)
    class_output = onnx_sessions['class_head'].run(
        None, {"shared_feature": shared_feature}
    )[0].flatten()[0] 

    angle_output = onnx_sessions['angle_head'].run(
        None, {"shared_feature": shared_feature}
    )
    
    pos_output = onnx_sessions['pos_head'].run(
        None, {"shared_feature": shared_feature}
    )[0].flatten() # 99 dimensions (V*C)

    # Sigmoid function for probability
    class_prob = 1.0 / (1.0 + np.exp(-class_output))
    angle_mean = angle_output[0].flatten() 

    return {
        "class_prob": class_prob,
        "angle_mean": angle_mean,
        "pos_mean_99d": pos_output # Predicted 3D Coordinates (99 dims)
    }

# --- Biomechanical Helper (Simulating Angle Recalculation from Predicted Pos) ---

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the angle (in degrees) between three points (p1-p2-p3) with p2 as pivot."""
    vector_ba = p1 - p2
    vector_bc = p3 - p2
    
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine_angle))


def calculate_kinematic_angles(pos_99d: np.ndarray) -> np.ndarray:
    """
    Recalculates the three key angles (L/R Knee, Hip Bend) from the predicted 99D position vector.
    This simulates the full diagnostic power of the multi-task model.
    """
    # Reshape 99D vector to (33, 3) joint coordinates
    coords_33_3 = pos_99d.reshape(33, 3)
    
    # Standard MediaPipe indices for Jump Squats
    indices = {
        'L_HIP': 23, 'R_HIP': 24,
        'L_KNEE': 25, 'R_KNEE': 26,
        'L_ANKLE': 27, 'R_ANKLE': 28,
        'L_SHOULDER': 11,
    }

    def get_coord(name):
        return coords_33_3[indices[name]]

    # 1. Left Knee Angle (HIP - KNEE - ANKLE)
    angle_lk = calculate_angle(get_coord('L_HIP'), get_coord('L_KNEE'), get_coord('L_ANKLE'))

    # 2. Right Knee Angle (HIP - KNEE - ANKLE)
    angle_rk = calculate_angle(get_coord('R_HIP'), get_coord('R_KNEE'), get_coord('R_ANKLE'))
    
    # 3. Hip Bend Angle (SHOULDER - HIP - KNEE)
    angle_hb = calculate_angle(get_coord('L_SHOULDER'), get_coord('L_HIP'), get_coord('L_KNEE'))

    return np.array([angle_lk, angle_rk, angle_hb])


def main_loop():
    """
    Main function to initialize camera and run pose detection and full ONNX inference.
    """
    
    if not load_onnx_models():
        return

    # Initialize MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Try multiple backends and indices for maximum compatibility
    camera_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
    camera_index = 0 
    cap = None

    for backend in camera_backends:
        cap = cv2.VideoCapture(camera_index, backend)
        
        # ... (Camera opening logic - omitted for brevity, assumed to be correct) ...
        # (The complex camera initialization loop needs to be preserved)
        if cap.isOpened():
             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
             cap.set(cv2.CAP_PROP_FPS, 30)
             
             success, image = cap.read()
             if success and image is not None:
                 print(f"Camera opened successfully using backend: {backend}")
                 cv2.waitKey(1) 
                 break
             else:
                 cap.release() 
                 cap = None
        
        if cap is None:
            cap = cv2.VideoCapture(1, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                success, image = cap.read()
                if success and image is not None:
                     print(f"Camera opened successfully using backend: {backend} (Index 1)")
                     cv2.waitKey(1)
                     break
                else:
                    cap.release()
                    cap = None

    if not cap or not cap.isOpened():
        print("Error: Could not open webcam using any available index or backend.")
        return
            
    pose_window = deque(maxlen=WINDOW_SIZE)
    frame_count = 0
    
    print("\n--- Starting Live Inference Loop (Press 'q' to stop) ---")

    while cap.isOpened():
        success, image = cap.read()
        if not success or image is None:
            continue

        image = cv2.flip(image, 1)

        # Process the image with MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose_detector.process(image_rgb)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Data Acquisition and Sanitization ---
        frame_3d_data = []

        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                if isinstance(landmark.x, float) and isinstance(landmark.y, float) and isinstance(landmark.z, float):
                    frame_3d_data.append([landmark.x, landmark.y, landmark.z])
                else:
                    frame_3d_data.append([0.0, 0.0, 0.0])
        
        if len(frame_3d_data) < 33:
            frame_3d_data.extend([[0.0, 0.0, 0.0]] * (33 - len(frame_3d_data)))

        frame_3d_array = np.array(frame_3d_data, dtype=np.float32)
        
        pose_window.append(frame_3d_array)
        
        # --- Visualization and Inference Trigger ---
        
        # 1. Draw Skeleton on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(220, 20, 60), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

        # 2. Run Inference when the window is full
        if len(pose_window) == WINDOW_SIZE:
            input_tensor = prepare_input_tensor(pose_window)
            
            results_inference = run_inference(input_tensor)
            
            # --- Recalculate Angles from Predicted Position ---
            predicted_angles = calculate_kinematic_angles(results_inference["pos_mean_99d"])

            # --- Print to Terminal ---
            print(f"Frame: {frame_count:04d} | Prob: {results_inference['class_prob']:.4f} | Angles (3): [{', '.join(f'{a:.2f}°' for a in predicted_angles)}]")
            
            # Display result status on the frame
            status_text = "CORRECT" if results_inference['class_prob'] > 0.5 else "WRONG"
            cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status_text == "CORRECT" else (0, 0, 255), 2, cv2.LINE_AA)

            # Important: Shift the window for continuous real-time processing
            pose_window.popleft() 
        
        # Display the frame
        cv2.imshow('ONNX Pose Live Tester (Press Q to Quit)', image)

        # Check for 'q' key press to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        frame_count += 1
        
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final print logic
    if 'results_inference' in locals() and frame_count > 0:
        print("\n--- Testing Complete ---")
        print(f"Total Frames Processed: {frame_count}")
        # Note: Printing the recalculated angle vector for clarity
        print(f"Final Prediction: Prob={results_inference['class_prob']:.4f}, Angles={predicted_angles}")
    elif frame_count > 0:
         print("\n--- Testing Complete (No Inference Run) ---")
         print(f"Total Frames Processed: {frame_count}")


if __name__ == "__main__":
    main_loop()
