import cv2
import mediapipe as mp
from pathlib import Path
import uuid
from tqdm import tqdm
import sys

# Setup path to import from app module
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import our custom modules
from app.utils import db, feature_engineering, asr
from app.proto_generated import rehab_pb2

# --- MediaPipe Pose Initializer ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except ImportError:
    print("Warning: mediapipe is not installed. Pose detection will not work.")
    pose_detector = None

def process_video_for_training(video_path: str, label: str):
    """
    Processes a single video file to extract raw pose keypoints and
    saves them to the training_skeletons table with a correct/wrong label.
    """
    if not pose_detector:
        print("Error: MediaPipe Pose is not initialized.")
        return

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    video_id = str(uuid.uuid4())
    print(f"\nProcessing '{video_path_obj.name}' with label '{label}' | Video ID: {video_id}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🤸‍♂️ Starting pose extraction from video frames ({total_frames} frames)...")
    
    for frame_no in tqdm(range(total_frames), desc=f"Analyzing Frames for '{label}'"):
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        
        if results.pose_world_landmarks:
            keypoints = []
            visibilities = []
            for landmark in results.pose_world_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
                visibilities.append(landmark.visibility)
            
            # Use the new db function to save to the training table
            db.save_training_skeleton_to_db(
                video_id=video_id,
                frame_no=frame_no,
                label=label,
                keypoints=keypoints,
                visibilities=visibilities
            )

    cap.release()
    print(f"✅ Finished processing '{video_path_obj.name}'.")
    
def process_video(video_path: str, user_id: str = "ground_truth_trainer"):
    """
    Processes a single video file to extract pose, features, and captions,
    then saves everything to the database.
    """
    if not pose_detector:
        print("Error: MediaPipe Pose is not initialized.")
        return

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    exercise_name = video_path_obj.stem
    # The category is the name of the parent folder (e.g., 'แขน')
    category = video_path_obj.parent.name
    session_id = f"session_{exercise_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
    
    print(f"\nProcessing '{exercise_name}' in category '{category}' | Session ID: {session_id}")

    # --- Step 0: Create session and exercise records in the database FIRST ---
    try:
        # Construct asset paths based on the file and its category folder
        thumbnail_url = f"/assets/thumbnails/{category}/{video_path_obj.stem}.jpg"
        preview_video_url = f"/assets/previews/{category}/{video_path_obj.stem}.gif"

        # The category is the name of the parent folder (e.g., 'แขน')
        category_name_map = {'คอ': 'neck', 'แขน': 'arm', 'ขา': 'leg', 'ลำตัว': 'torso', 'ทั้งตัว': 'full'}
        category_folder_name = video_path_obj.parent.name
        category_id = category_name_map.get(category_folder_name, 'unknown')

        db.create_or_update_exercise_and_session(
            session_id=session_id,
            user_id=user_id,
            exercise_name=exercise_name,
            category=category_id, # << PASS THE CATEGORY ID
            description=f"คำอธิบายสำหรับท่า {exercise_name}",
            duration="3:15", # Placeholder
            level="Intermediate", # Placeholder
            thumbnail_url=thumbnail_url,
            preview_video_url=preview_video_url
        )
        print(f"🗂️  Successfully created/updated exercise and session records in database.")
    except Exception as e:
        print(f"❌ Critical Error: Could not create session in database. Aborting. Error: {e}")
        return
    
    # --- Step 1: Transcribe Audio using Whisper ---
    print("🎤 Starting audio transcription with Whisper (this may take a while)...")
    try:
        captions = asr.transcribe_video_audio(video_path)
        if captions:
            db.save_captions_to_db(session_id, captions)
            print(f"📝 Transcription complete. Saved {len(captions)} caption segments.")
        else:
            print("📝 Transcription complete. No speech detected.")
    except Exception as e:
        print(f"⚠️ Could not transcribe audio, continuing without captions. Error: {e}")
    
    # --- Step 2: Process Video Frames for Pose ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"🤸‍♂️ Starting pose extraction from video frames ({total_frames} frames)...")
    
    for frame_no in tqdm(range(total_frames), desc=f"Analyzing Frames for '{exercise_name}'"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        
        if results.pose_world_landmarks:
            frame_proto = rehab_pb2.Frame()
            frame_proto.user_id = user_id
            frame_proto.session_id = session_id
            frame_proto.frame_no = frame_no
            frame_proto.timestamp = frame_no / fps
            
            for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                joint = frame_proto.joints.add()
                joint.id = i
                joint.x = landmark.x
                joint.y = landmark.y
                joint.z = landmark.z
                joint.visibility = landmark.visibility
            
            # Extract and add engineered features
            features = feature_engineering.extract_features_from_frame(frame_proto)
            frame_proto.features.extend(features)
            
            db.save_frame_to_db(frame_proto)

    cap.release()
    print(f"✅ Finished processing '{exercise_name}'.")

