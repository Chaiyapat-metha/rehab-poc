--
-- File: infra/timescaledb-init.sql
-- Purpose: Initialize TimescaleDB schema for training and user data
--

-- 1. EXTENSIONS
-- ต้องเปิดใช้งาน TimescaleDB extension ก่อน
CREATE EXTENSION IF NOT EXISTS timescaledb;


-- ========================================================================
-- TRAINING DATABASE TABLES (Time-Series Data)
-- ========================================================================

-- ตารางหลักสำหรับเก็บข้อมูล Skeleton/Joints (B, T, V, C) 
CREATE TABLE skeleton_data (
    ingest_uuid UUID NOT NULL, 
    ingest_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    frame_idx INTEGER NOT NULL,
    exercise_id TEXT NOT NULL,
    joints_data BYTEA NOT NULL,
    video_id_original TEXT, 
    
    -- PK ควรเป็น UUID + Frame Index 
    PRIMARY KEY (ingest_uuid, frame_idx) 
);

-- แปลงตาราง skeleton_data ให้เป็น Hypertable โดยใช้ ingest_timestamp
SELECT create_hypertable('skeleton_data', 'ingest_timestamp');


-- ตารางสำหรับเก็บ Ground Truth Labels (สำหรับ Multi-task Training)
CREATE TABLE exercise_labels (
    ingest_uuid UUID NOT NULL,
    ingest_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    frame_idx INTEGER NOT NULL,
    exercise_id TEXT NOT NULL,
    
    -- Classification Head Label: 0=Correct, 1=Incorrect, 2=Other...
    label_class INTEGER,
    
    -- Regression Head Label: Vector ของมุม/ความคลาดเคลื่อนที่กำหนดใน exercises.yaml (เช่น 6-dim angle vector)
    label_angles_vector REAL[], 
    
    -- Positional Head Label: 99-dim vector (V*C)
    label_pos_vector REAL[],
    
    PRIMARY KEY (ingest_uuid, frame_idx)
);

-- แปลงตาราง exercise_labels ให้เป็น Hypertable โดยใช้ ingest_timestamp
SELECT create_hypertable('exercise_labels', 'ingest_timestamp');


-- ========================================================================
-- USER/HISTORY DATABASE TABLES (Key-Value/Relational Data)
-- ========================================================================

-- ตารางสำหรับเก็บประวัติการทำซ้ำ (Reps) ของผู้ใช้ สำหรับ ThresholdController
CREATE TABLE rep_history (
    rep_id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    session_id TEXT,
    rep_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    is_success BOOLEAN NOT NULL, -- ผลลัพธ์ Rep นี้ (ใช้ Threshold ปัจจุบัน)
    metric_errors JSONB -- เก็บรายละเอียดความผิดพลาดของมุม/ตำแหน่ง (เช่น {"LEFT_KNEE_ANGLE": 5.2})
);

-- ตารางสำหรับเก็บ Threshold ปัจจุบันของผู้ใช้แต่ละคน
CREATE TABLE user_thresholds (
    user_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    metric_name TEXT NOT NULL, -- เช่น LEFT_KNEE_ANGLE_ERROR
    current_threshold REAL NOT NULL, -- ค่า Threshold ปัจจุบัน (เช่น 15.0 องศา)
    PRIMARY KEY (user_id, exercise_id, metric_name)
);

-- ตารางสำหรับบันทึกการเสนอ/เปลี่ยนแปลง Threshold (ThresholdController)
CREATE TABLE threshold_history (
    history_id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    date_proposed TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    proposed_value REAL NOT NULL,
    previous_value REAL NOT NULL,
    user_accepted BOOLEAN -- NULL คือรอกาตัดสินใจ, TRUE/FALSE คือผลการตัดสินใจ
);

-- ตารางสำหรับบันทึกการเสนอ Thresholds (Proposal History)
CREATE TABLE proposal_history (
    proposal_id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    proposal_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    -- เก็บค่า Thresholds ที่ถูกเสนอทั้งหมดเป็น JSONB
    proposed_thresholds JSONB NOT NULL 
);