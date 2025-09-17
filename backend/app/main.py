import asyncio
import logging
import uuid
from collections import defaultdict, deque

import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
from . import config
from .utils import db, feature_engineering
from .proto_generated import rehab_pb2
from scripts.video_processor import process_video
from .llm.rag_chain import invoke_rag_chain 
from pydantic import BaseModel

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(title="Rehab-PoC Backend")

# --- CORS Middleware ---
# Allows the frontend (running on a different port) to communicate with the backend.
origins = [
    "http://localhost",
    "http://localhost:5173", # Default Vite dev server port
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Constants for Real-time Inference ---
WINDOW_SIZE = config.app_config.get('supervised_model', {}).get('params', {}).get('window_size', 48)
user_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
CLASS_ID_TO_MESSAGE = {
    0: "เยี่ยมมากครับ! (ท่าถูกต้อง)",
    1: "ข้อศอกซ้ายงอน้อยเกินไปครับ",
    2: "ข้อศอกขวางอน้อยเกินไปครับ",
    3: "เข่าซ้ายย่อน้อยเกินไปครับ",
    4: "เข่าขวาย่อน้อยเกินไปครับ",
}

# --- Load Models on Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    try:
        logger.info("Loading AI models...")
        model_key = config.app_config.get('active_model', 'supervised_model')
        app.state.active_model = config.get_model_instance(model_key)
        logger.info(f"Active model '{model_key}' loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load AI model. Error: {e}")
        app.state.active_model = None
    try:
        app.state.db_pool = db.pool
        logger.info("Database connection pool initialized.")
    except Exception as e:
        logger.error(f"FATAL: Could not connect to the database. Error: {e}")
        app.state.db_pool = None
    logger.info("Application startup complete. Ready to accept connections.")


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to Rehab-PoC API"}

@app.get("/proposals/{user_id}")
async def get_user_proposals(user_id: str):
    """Fetches all pending threshold proposals for a user."""
    proposals = db.get_pending_proposals(user_id)
    return {"status": "success", "proposals": proposals}

@app.get("/api/exercises")
async def get_exercises():
    """Endpoint to fetch all exercise data for the frontend."""
    try:
        exercise_data = await asyncio.to_thread(db.get_all_exercises)
        return {"status": "success", "exercises": exercise_data}
    except Exception as e:
        logger.error(f"Error fetching exercises: {e}")
        return {"status": "error", "message": "Could not fetch exercise data."}

@app.get("/api/exercises")
async def get_exercises():
    """Endpoint to fetch all exercise data for the frontend, grouped by category."""
    try:
        # Fetch the flat list of exercises from the database
        exercise_list = await asyncio.to_thread(db.get_all_exercises)
        
        # --- THE FIX IS HERE: Group the data on the backend ---
        exercises_grouped_by_category = {}
        for exercise in exercise_list:
            category = exercise.get("category")
            if category:
                if category not in exercises_grouped_by_category:
                    exercises_grouped_by_category[category] = []
                exercises_grouped_by_category[category].append(exercise)
                
        return {"status": "success", "exercisesByCat": exercises_grouped_by_category}
        
    except Exception as e:
        logger.error(f"Error fetching and grouping exercises: {e}")
        return {"status": "error", "message": "Could not fetch exercise data."}

@app.post("/proposals/{proposal_id}/respond")
async def respond_to_proposal_endpoint(proposal_id: int, response: dict):
    """Updates the status of a proposal based on user response."""
    # response should be like {"response": "accepted"} or {"response": "rejected"}
    new_status = response.get("response")
    if new_status not in ["accepted", "rejected"]:
        return {"status": "error", "message": "Invalid response status."}
    
    db.respond_to_proposal(proposal_id, new_status)
    # TODO: If accepted, update the user's actual target table.
    
    return {"status": "success", "message": f"Proposal {proposal_id} updated to {new_status}."}

# # --- NEW: LLM Chat Endpoint ---
# @app.post("/chat")
# async def handle_chat(request: ChatRequest):
#     """
#     Handles a user's chat message by invoking the RAG chain.
#     """
#     logger.info(f"Received chat message from user {request.user_id}: '{request.message}'")
#     try:
#         # Run the potentially slow RAG chain in a separate thread
#         response_text = await asyncio.to_thread(
#             invoke_rag_chain, request.user_id, request.message
#         )
#         return {"status": "success", "reply": response_text}
#     except Exception as e:
#         logger.error(f"Error invoking RAG chain: {e}")
#         return {"status": "error", "reply": "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล"}
    
    
@app.websocket("/ws/live/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            # Receive raw keypoints from the frontend
            data = await websocket.receive_json()
            keypoints = data.get('keypoints', []) 
            
            # Convert incoming JSON to a Protobuf message
            frame_proto = rehab_pb2.Frame()
            for i, kp_data in enumerate(keypoints):
                joint = frame_proto.joints.add()
                joint.id = i
                joint.x = kp_data.get('x', 0)
                joint.y = kp_data.get('y', 0)
                joint.z = kp_data.get('z', 0)
                joint.visibility = kp_data.get('score', 0)

            # Extract features
            features = feature_engineering.extract_features_from_frame(frame_proto)
            
            # Add features to the user's buffer
            buffer = user_buffers[session_id]
            buffer.append(features)

            # If buffer is full, run inference
            if len(buffer) >= WINDOW_SIZE:
                if app.state.supervised_model:
                    window_np = np.array(buffer).astype(np.float32).reshape(1, WINDOW_SIZE, -1)
                    
                    # --- Run prediction with the new model ---
                    results = app.state.supervised_model.predict(window_np)
                    
                    class_id = int(results['predicted_class_id'])
                    message = CLASS_ID_TO_MESSAGE.get(class_id, "ไม่สามารถระบุข้อผิดพลาดได้")
                    
                    # Determine status for frontend UI
                    status = "correct" if class_id == 0 else "error"
                    
                    feedback = {
                        "status": status,
                        "message": message,
                        "severity": float(results['predicted_severity'])
                    }
                    
                    await websocket.send_json(feedback)
                else:
                    await websocket.send_json({"status": "error", "message": "โมเดลยังไม่พร้อมใช้งาน"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        del user_buffers[session_id] # Clean up buffer for disconnected user
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        del user_buffers[session_id]

