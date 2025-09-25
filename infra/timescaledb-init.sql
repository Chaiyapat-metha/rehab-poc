-- เปิดใช้งาน TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ตารางเก็บข้อมูลท่าออกกำลังกาย 
CREATE TABLE IF NOT EXISTS exercises (
    exercise_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL, 
    description TEXT,
    duration VARCHAR(50),
    level VARCHAR(50),
    thumbnail_url VARCHAR(255),
    preview_video_url VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ตารางเก็บข้อมูลผู้ใช้
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    rehab_stage INT,
    consent_flags JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ตารางเก็บข้อมูล Session
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    exercise_name VARCHAR(255) REFERENCES exercises(name),
    device_info TEXT,
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    calibration_data BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ตารางหลักสำหรับเก็บข้อมูล Frame (Hypertable)
CREATE TABLE IF NOT EXISTS frames (
    "time" TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    frame_no BIGINT,
    raw_frame_data BYTEA, -- Protobuf binary blob
    feature_vector REAL[], -- Engineered features for faster query
    labels JSONB,
    PRIMARY KEY (session_id, "time")
);

-- สร้าง Hypertable บนตาราง frames
-- Partitioning by session_id is a good practice for multi-tenant data
SELECT create_hypertable('frames', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS idx_session_id_time ON frames (session_id, "time" DESC);

-- New table to store raw skeleton data for training
CREATE TABLE IF NOT EXISTS training_skeletons (
    time TIMESTAMPTZ NOT NULL,
    video_id VARCHAR(255) NOT NULL,
    frame_no BIGINT,
    label VARCHAR(50) NOT NULL,
    keypoints REAL[], -- Raw 3D coordinates (33 joints * 3 coords = 99 values)
    visibility REAL[], -- Visibility scores for each joint
    PRIMARY KEY (video_id, time)
);

-- Create a Hypertable on the new table
SELECT create_hypertable('training_skeletons', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- ตารางเก็บผลลัพธ์จาก ASR
CREATE TABLE IF NOT EXISTS captions (
    caption_id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    start_ts TIMESTAMPTZ, -- Timestamp แบบเต็ม
    end_ts TIMESTAMPTZ,   -- Timestamp แบบเต็ม
    text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ตารางเก็บข้อมูลความคืบหน้า
CREATE TABLE IF NOT EXISTS progress (
    progress_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    week_start_date DATE,
    metrics JSONB, -- e.g., {"elbow_left_max_rom": 120.5, "avg_score": 0.85}
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ตารางสำหรับเก็บ "ข้อเสนอ" การปรับแก้เป้าหมาย
CREATE TABLE IF NOT EXISTS threshold_proposals (
    proposal_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    exercise_name VARCHAR(255) REFERENCES exercises(name),
    metric_key TEXT NOT NULL, -- e.g., 'left_elbow_angle_max'
    current_value REAL,
    proposed_value REAL NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, accepted, rejected
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    responded_at TIMESTAMPTZ
);