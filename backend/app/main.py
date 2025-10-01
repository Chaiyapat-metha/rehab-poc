from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Required imports for utilities and external modules
from fastapi.middleware.cors import CORSMiddleware
from app.utils.threshold_controller import ThresholdController
from app.utils.feedback_engine import generate_feedback
from app.config import load_config
from app.llm.rag_chain import invoke_rag_chain 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Configuration & Initialization ---
config = load_config()
FEATURE_DIM = config['model_config']['model']['backbone']['gru']['hidden']
WINDOW_SIZE = 16 # Must match dataset.py config

app = FastAPI(title="Rehab Pose Correction API", version="1.0")
threshold_controller = ThresholdController()
onnx_sessions: Dict[str, Dict[str, ort.InferenceSession]] = {} # Cache ONNX sessions
active_websockets: Dict[str, WebSocket] = {}

# --- CORS Middleware (Crucial for Frontend) ---
origins = [
    "http://localhost",
    "http://localhost:5173", 
    "http://localhost:8011", 
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8011", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- Pydantic Models ---

class RagRequest(BaseModel):
    user_id: str = Field(..., example="trainer_chaiyapat")
    question: str = Field(..., example="สรุปผลการออกกำลังกาย Jump_squats ที่ผ่านมา พร้อมข้อแนะนำ")

class InferenceRequest(BaseModel):
    user_id: str = Field(..., example="trainer_chaiyapat")
    exercise_id: str = Field(..., example="Jump_squats")
    window_frames: List[Any] = Field(..., description="[T=16, V=33, C=3] window of joint coordinates")

# --- 1. Model Loading and Inference Logic ---

def load_onnx_models(exercise_id: str):
    """Loads and caches ONNX sessions for backbone and heads."""
    if exercise_id in onnx_sessions:
        return onnx_sessions[exercise_id]
        
    try:
        # Load Backbone (Shared)
        # 💡 FIX: Use BASE_DIR to ensure correct path resolution
        backbone_path = BASE_DIR / "backend" / "weights" / "backbone.onnx"
        backbone_sess = ort.InferenceSession(str(backbone_path))

        # Load Heads (Classification and Angles)
        class_path = BASE_DIR / "backend" / "weights" / f"{exercise_id}_class.onnx"
        angle_path = BASE_DIR / "backend" / "weights" / f"{exercise_id}_angle.onnx"
        
        class_sess = ort.InferenceSession(str(class_path))
        angle_sess = ort.InferenceSession(str(angle_path))

        onnx_sessions[exercise_id] = {
            "backbone": backbone_sess,
            "class_head": class_sess,
            "angle_head": angle_sess
        }
        return onnx_sessions[exercise_id]
    except Exception as e:
        # 💡 Improvement: Log the absolute path attempted
        logger.error(f"Failed to load ONNX models for {exercise_id}. Paths attempted: {backbone_path.resolve()}, {class_path.resolve()}")
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX models for {exercise_id}: {e}")

def run_inference(models: Dict[str, ort.InferenceSession], input_tensor: np.ndarray) -> Dict[str, Any]:
    """Runs input through the shared backbone and specific heads."""
    
    # 1. Backbone (Input: [1, T, V, C] -> Output: [1, 256])
    shared_feature = models['backbone'].run(
        None, {"input_data": input_tensor}
    )[0].astype(np.float32)

    # 2. Classification Head
    class_logit = models['class_head'].run(
        None, {"shared_feature": shared_feature}
    )[0].flatten()[0] 

    # 3. Angle Regression Head
    angle_outputs = models['angle_head'].run(
        None, {"shared_feature": shared_feature}
    )
    angle_mean = angle_outputs[0].flatten() 

    return {
        "class_logit": class_logit,
        "angle_mean": angle_mean,
        "class_prob": 1.0 / (1.0 + np.exp(-class_logit))
    }

# --- 2. API Endpoints (HTTP) ---

@app.get("/api/exercises", response_model=Dict[str, Any])
async def get_all_exercises():
    """Provides all exercise metadata grouped by category for the HomeScreen."""
    try:
        exercise_configs = config['exercises'] # Access cached config
        exercise_names = list(exercise_configs.keys())
        
        exercises_by_category = {}
        
        # 💡 CORRECT CATEGORIZATION MAP (Fixing the Side Plank issue)
        category_map = {
            'Incline_rows_with_dumbbell': 'arm', 'Punching_in_place': 'arm',
            'Side_plank_with_pull_through_left': 'torso', # CORRECTED
            'Side_plank_with_pull_through_right': 'torso', # CORRECTED
            'Stretching_forearm_muscles': 'arm', 'Stretching_lower_trapezius': 'arm',
            'Stretching_rhomboids': 'arm', 'Jump_squats': 'leg',
            'Lying_leg_raises': 'leg', 'Mountain_climbers': 'leg',
            'Stretching_upper_trapezius': 'neck', 'Triceps_dips_on_floor': 'full',
            'Pike_pushups': 'torso',
        }
        
        for name in exercise_names:
            category_id = category_map.get(name, 'misc')
            
            if category_id not in exercises_by_category:
                exercises_by_category[category_id] = []
                
            # Note: Frontend will use category_id (e.g., 'arm') to build the path
            exercises_by_category[category_id].append({
                "id": name,
                "name": name.replace('_', ' '),
                "thumbnail_url": f"/assets/thumbnails/{category_id}/{name}.jpg" 
            })
            
        return {"status": "success", "exercisesByCat": exercises_by_category}
        
    except Exception as e:
        logger.error(f"Failed to compile exercise list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compile exercise list: {e}")

@app.post("/api/rag/summarize", response_model=Dict[str, str])
async def rag_summarize(request: RagRequest):
    """Invokes the RAG chain to generate a session summary."""
    try:
        summary_text = invoke_rag_chain(request.user_id, request.question)
        return {"answer": summary_text}
    except Exception as e:
        logger.error(f"RAG Chain invocation failed: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการประมวลผลสรุปผลจาก AI (API Error)"}

# --- 3. Threshold Controller Endpoints ---

@app.get("/user/{user_id}/thresholds/{exercise_id}", response_model=Dict[str, float])
def get_thresholds(user_id: str, exercise_id: str):
    """Retrieves current personalized thresholds."""
    return threshold_controller.get_current_thresholds(user_id, exercise_id)

@app.post("/user/{user_id}/thresholds/propose/{exercise_id}", response_model=Dict[str, Any])
def propose_thresholds(user_id: str, exercise_id: str):
    """Calculates and proposes new thresholds."""
    proposals = threshold_controller.propose_new_thresholds(user_id, exercise_id)
    return {"proposed_thresholds": proposals}

@app.post("/user/{user_id}/thresholds/commit/{exercise_id}", response_model=Dict[str, Any])
def commit_thresholds(user_id: str, exercise_id: str, updates: Dict[str, float]):
    """Commits the proposed thresholds after user confirmation."""
    threshold_controller.commit_thresholds(user_id, exercise_id, updates)
    return {"status": "committed", "updates": updates}


# --- 4. WebSocket Endpoint (Real-time Pose Feedback) ---
@app.websocket("/ws/live/{exercise_id}")
async def websocket_endpoint(websocket: WebSocket, exercise_id: str, user_id: str = "trainer_chaiyapat"):
    await websocket.accept()
    
    models = None
    
    try:
        models = load_onnx_models(exercise_id)
    except Exception as e:
        logger.error(f"WS Model Load Error: {e}")
        await websocket.send_json({"status": "error", "display_text": "ไม่สามารถโหลดโมเดลได้", "tts_text": "ไม่สามารถเริ่มการประมวลผล"})
        return
        
    # 💡 NEW: Error state tracking (Req. 6 - Debounce)
    # Track consecutive prediction Windows (16 frames) where the model predicts 'wrong'
    CONSECUTIVE_ERROR_WINDOWS_THRESHOLD = 3 # 3 consecutive windows (3*16 = 48 frames)
    consecutive_error_windows = 0
    
    try:
        while True:
            # 1. Receive Pose Data from Frontend
            data = await websocket.receive_text()
            pose_data = json.loads(data)
            
            # 💡 FIX: Expecting 'window_frames' (T x V x C) from Frontend
            window_frames_data = pose_data.get('window_frames')
            
            if not window_frames_data or len(window_frames_data) != WINDOW_SIZE:
                 # Frontend is dropping frames or not warmed up yet
                 continue

            # Convert the list of list of lists to numpy array: [T, V, C]
            window_np = np.array(window_frames_data, dtype=np.float32) 
            
            if window_np.shape != (WINDOW_SIZE, 33, 3): 
                logger.error(f"Received incorrect shape: {window_np.shape}")
                continue
                
            # Add batch dimension: [1, T, V, C] for ONNX
            input_tensor = window_np[np.newaxis, ...] 

            # 2. Run Inference
            outputs = run_inference(models, input_tensor)
            
            # 3. Decision & Feedback Generation
            # We assume the default classification threshold is 0.5 if not specified in the model config.
            CLASSIFICATION_THRESHOLD = config['model_config']['inference'].get('min_success_threshold', 0.5) 
            is_correct = outputs['class_prob'] >= CLASSIFICATION_THRESHOLD
            
            # Get current thresholds for angle checking
            thresholds = threshold_controller.get_current_thresholds(user_id, exercise_id)
            angle_names = config['exercises'].get(exercise_id, {}).get('angle_output_order', [])
            angle_errors_map = {name: mean for name, mean in zip(angle_names, outputs['angle_mean'])}
            
            # Generate Feedback (assumes generate_feedback handles template mapping)
            feedback_result = generate_feedback(is_correct, angle_errors_map, thresholds, exercise_id)
            
            # 4. Debouncing Logic (Per-Rep policy: Announce only on sustained error)
            if not is_correct:
                consecutive_error_windows += 1
            else:
                consecutive_error_windows = 0

            # Only announce/send TTS text if the error is sustained
            should_announce = (consecutive_error_windows >= CONSECUTIVE_ERROR_WINDOWS_THRESHOLD)
            
            response = {
                "status": "correct" if is_correct else "error",
                "display_text": feedback_result['display_text'],
                "tts_text": feedback_result['tts_text'] if should_announce else None, # 💡 ส่ง TTS ก็ต่อเมื่อผิดติดต่อกัน
                "wrong_joints": feedback_result['wrong_joints'], 
            }
            
            # Reset consecutive counter immediately after fulfilling the announcement threshold
            if should_announce:
                 consecutive_error_windows = 0
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {exercise_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket processing: {e}")
        await websocket.send_json({"status": "error", "display_text": "เกิดข้อผิดพลาดในการประมวลผล", "tts_text": "เกิดข้อผิดพลาดในการประมวลผล"})
