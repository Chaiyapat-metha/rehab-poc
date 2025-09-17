import psycopg2
import psycopg2.pool
import psycopg2.extras
from contextlib import contextmanager
import os
from datetime import datetime, timezone
from typing import List

# Import our Protobuf message definition
from ..proto_generated import rehab_pb2

# --- Database Connection Pool ---
# ใช้ค่าจาก environment variables ถ้ามี, หรือใช้ default สำหรับ local dev
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "rehab_db")
DB_USER = os.getenv("DB_USER", "nonny") # Changed to your user
DB_PASS = os.getenv("DB_PASS", "nonny") # Changed to your password

try:
    pool = psycopg2.pool.SimpleConnectionPool(
        1, 10,  # minconn, maxconn
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=5433
    )
except psycopg2.OperationalError as e:
    print(f"❌ CRITICAL: Could not connect to the database: {e}")
    print("Please ensure the Docker container for TimescaleDB is running and accessible.")
    pool = None

@contextmanager
def get_db_connection():
    """Provides a database connection from the pool."""
    if pool is None:
        raise ConnectionError("Database connection pool is not available.")
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def create_session_and_exercise(session_id: str, user_id: str, exercise_name: str, device_info: str = "Batch Video Ingest"):
    """
    Creates a user, an exercise, and then a session record.
    """
    user_sql = "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;"
    exercise_sql = "INSERT INTO exercises (name) VALUES (%s) ON CONFLICT (name) DO NOTHING;"
    
    # --- SQL Statement Updated Here ---
    session_sql = """
    INSERT INTO sessions (session_id, user_id, exercise_name, device_info, start_ts)
    VALUES (%s, %s, %s, %s, %s);
    """
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(user_sql, (user_id,))
            cur.execute(exercise_sql, (exercise_name,))
            
            start_time = datetime.now(timezone.utc)
            # --- Parameters Updated Here ---
            cur.execute(session_sql, (session_id, user_id, exercise_name, device_info, start_time))

def create_or_update_exercise_and_session(
    session_id: str,
    user_id: str,
    exercise_name: str,
    category: str, 
    description: str,
    duration: str,
    level: str,
    thumbnail_url: str,
    preview_video_url: str,
    device_info: str = "Batch Video Ingest"
):
    """
    Ensures a user and an exercise exist (creating or updating the exercise),
    then creates a new session record. This is a single transaction.
    """
    user_sql = "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING;"
    
    # This SQL now includes the 'category' column
    exercise_sql = """
    INSERT INTO exercises (name, category, description, duration, level, thumbnail_url, preview_video_url, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (name) DO UPDATE SET
        category = EXCLUDED.category,
        description = EXCLUDED.description,
        duration = EXCLUDED.duration,
        level = EXCLUDED.level,
        thumbnail_url = EXCLUDED.thumbnail_url,
        preview_video_url = EXCLUDED.preview_video_url;
    """
    
    session_sql = """
    INSERT INTO sessions (session_id, user_id, exercise_name, device_info, start_ts)
    VALUES (%s, %s, %s, %s, %s);
    """
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(user_sql, (user_id,))
            
            now = datetime.now(timezone.utc)
            # The parameters now include 'category'
            cur.execute(exercise_sql, (
                exercise_name, category, description, duration, level, thumbnail_url, preview_video_url, now
            ))
            
            cur.execute(session_sql, (session_id, user_id, exercise_name, device_info, now))

def save_frame_to_db(frame_proto: rehab_pb2.Frame):
    """Saves a single frame's data to the database."""
    sql = """
    INSERT INTO frames (time, session_id, frame_no, raw_frame_data, feature_vector)
    VALUES (%s, %s, %s, %s, %s);
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    datetime.fromtimestamp(frame_proto.timestamp, tz=timezone.utc),
                    frame_proto.session_id,
                    frame_proto.frame_no,
                    frame_proto.SerializeToString(),
                    list(frame_proto.features) 
                ))
    except Exception as e:
        print(f" -> Error saving frame {frame_proto.frame_no}: {e}")

def save_captions_to_db(session_id: str, captions: List[dict]):
    """Saves a list of caption segments to the database."""
    sql = """
    INSERT INTO captions (session_id, start_ts, end_ts, text)
    VALUES (%s, %s, %s, %s);
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for caption in captions:
                start_time = datetime.fromtimestamp(caption['start'], tz=timezone.utc)
                end_time = datetime.fromtimestamp(caption['end'], tz=timezone.utc)
                cur.execute(sql, (session_id, start_time, end_time, caption['text']))

def get_existing_exercise_names() -> set:
    """
    Queries the database and returns a set of all exercise names
    that have already been ingested.
    """
    sql = "SELECT name FROM exercises;"
    existing_names = set()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                results = cur.fetchall()
                # Convert list of tuples [('name1',), ('name2',)] to a set {'name1', 'name2'}
                existing_names = {row[0] for row in results}
    except Exception as e:
        print(f"Warning: Could not fetch existing exercise names. May re-process some videos. Error: {e}")
    
    return existing_names

def execute_query(sql: str, params: tuple = None) -> list:
    """A generic function to execute a read-only SQL query."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

def insert_threshold_proposal(proposal: dict):
    """Inserts a new threshold proposal into the database."""
    sql = """
    INSERT INTO threshold_proposals 
        (user_id, exercise_name, metric_key, current_value, proposed_value, status)
    VALUES (%s, %s, %s, %s, %s, 'pending')
    -- If a similar pending proposal exists, do nothing
    ON CONFLICT DO NOTHING; 
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                proposal['user_id'],
                proposal['exercise_name'],
                proposal['metric_key'],
                proposal['current_value'],
                proposal['proposed_value']
            ))
            
def get_pending_proposals(user_id: str) -> list:
    """Fetches all proposals with 'pending' status for a user."""
    sql = "SELECT proposal_id, exercise_name, metric_key, current_value, proposed_value FROM threshold_proposals WHERE user_id = %s AND status = 'pending' ORDER BY created_at DESC;"
    proposals = []
    with get_db_connection() as conn:
        # Use a dictionary cursor to get results as dicts
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (user_id,))
            for row in cur.fetchall():
                proposals.append(dict(row))
    return proposals

def respond_to_proposal(proposal_id: int, new_status: str):
    """Updates a proposal's status and responded_at timestamp."""
    sql = "UPDATE threshold_proposals SET status = %s, responded_at = %s WHERE proposal_id = %s;"
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (new_status, datetime.now(timezone.utc), proposal_id))
        
def get_all_exercises() -> list:
    """
    Fetches all exercises from the database, already structured correctly.
    """
    # SQL query is now simpler because we have a dedicated category column.
    sql = """
    WITH RankedExercises AS (
        SELECT
            name,
            category,
            description,
            duration,
            level,
            thumbnail_url,
            preview_video_url,
            ROW_NUMBER() OVER(PARTITION BY category ORDER BY created_at DESC) as rn
        FROM exercises
    )
    SELECT
        name,
        category,
        description,
        duration,
        level,
        thumbnail_url,
        preview_video_url
    FROM RankedExercises
    WHERE rn <= 10;
    """
    exercises = []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                exercises.append(dict(row))
    return exercises

def get_frames_for_exercise(exercise_name: str) -> list:
    """Fetches all frame data for a specific ground truth exercise session."""
    sql = """
    SELECT f.raw_frame_data
    FROM frames f
    JOIN sessions s ON f.session_id = s.session_id
    WHERE s.user_id = 'ground_truth_trainer' AND s.exercise_name = %s
    ORDER BY f.time;
    """
    frames = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (exercise_name,))
            results = cur.fetchall()
            for row in results:
                frame_proto = rehab_pb2.Frame()
                frame_proto.ParseFromString(row[0])
                # Convert protobuf to a simple dictionary for JSON serialization
                frame_dict = {
                    f"joint_{j.id}": {"x": j.x, "y": j.y, "z": j.z, "visibility": j.visibility}
                    for j in frame_proto.joints
                }
                frames.append(frame_dict)
    return frames