import React from 'react';

// --- SVG Icons ---
export const ICONS = {
    search: <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>,
    close: <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>,
    play: <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>,
    pause: <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>,
    mute: <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>,
    unmute: <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>,
    home: <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>,
    profile: <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>,
};

// --- Color Palette ---
export const PALETTE = { 
    light: "#dde1e6", 
    muted: "#6a7b91", 
    bg: "#040404", 
    primary: "#5898e8", 
    secondaryText: "#a0a7b2", 
    uiGrey: "#505050", 
    error: "#ff4d4f" 
};

// --- Utility: Naming and Path Mapping ---
const EXERCISE_MAPPING = {
    'Incline_rows_with_dumbbell': { name_th: 'อินไคลน์ โรว์ (ดัมเบลล์)', duration: '12 Reps', level: 'Basic', description: 'เน้นบริหารกล้ามเนื้อหลังส่วนบนและแขน', category: 'arm', type: 'video' },
    'Punching_in_place': { name_th: 'ต่อยมวยอยู่กับที่', duration: '1 นาที', level: 'Cardio', description: 'ช่วยเพิ่มการเผาผลาญและแรงปะทะ', category: 'arm', type: 'clip' },
    'Side_plank_with_pull_through_left': { name_th: 'ไซด์แพลงค์ (ดึงซ้าย)', duration: '10 Reps', level: 'Core', description: 'เสริมสร้างแกนกลางและหลังส่วนล่าง', category: 'torso', type: 'clip' },
    'Side_plank_with_pull_through_right': { name_th: 'ไซด์แพลงค์ (ดึงขวา)', duration: '10 Reps', level: 'Core', description: 'เสริมสร้างแกนกลางและหลังส่วนล่าง', category: 'torso', type: 'clip' },
    'Stretching_forearm_muscles': { name_th: 'ยืดกล้ามเนื้อแขน', duration: '30 วิ', level: 'Flexibility', description: 'ลดอาการเกร็งบริเวณแขนและข้อมือ', category: 'arm', type: 'clip' },
    'Stretching_lower_trapezius': { name_th: 'ยืดเทรปิเซียสส่วนล่าง', duration: '30 วิ', level: 'Flexibility', description: 'คลายกล้ามเนื้อไหล่และหลังส่วนบน', category: 'arm', type: 'clip' },
    'Stretching_rhomboids': { name_th: 'ยืดรอมบอยด์', duration: '30 วิ', level: 'Flexibility', description: 'ช่วยเปิดไหล่และบรรเทาอาการปวดคอ', category: 'arm', type: 'clip' },
    'Jump_squats': { name_th: 'จัมพ์ สควอทส์', duration: '15 Reps', level: 'Advanced', description: 'เพิ่มความแข็งแรงและแรงระเบิดให้ขาและสะโพก', category: 'leg', type: 'video' },
    'Lying_leg_raises': { name_th: 'นอนยกขา', duration: '15 Reps', level: 'Core', description: 'เน้นบริหารกล้ามเนื้อหน้าท้องส่วนล่าง', category: 'leg', type: 'video' },
    'Mountain_climbers': { name_th: 'เมาน์เทน ไคลมเบอร์', duration: '1 นาที', level: 'Cardio', description: 'เพิ่มอัตราการเต้นของหัวใจและบริหารทั้งตัว', category: 'leg', type: 'clip' },
    'Stretching_upper_trapezius': { name_th: 'ยืดกล้ามเนื้อคอ (เทรปิเซียส)', duration: '45 วิ', level: 'Flexibility', description: 'ลดอาการปวดคอและไหล่จากความตึงเครียด', category: 'neck', type: 'clip' },
    'Triceps_dips_on_floor': { name_th: 'ไทรเซ็ปส์ ดิปส์', duration: '10 Reps', level: 'Basic', description: 'เสริมสร้างกล้ามเนื้อแขนด้านหลัง', category: 'full', type: 'video' },
    'Pike_pushups': { name_th: 'ไพค์ พุชอัพ', duration: '8 Reps', level: 'Intermediate', description: 'เน้นบริหารหัวไหล่และแขนส่วนบน', category: 'torso', type: 'video' }
};

export const EXERCISES_MAP = EXERCISE_MAPPING;

// --- Mock Data (for categories display structure) ---
export const MOCK_DATA = {
    categories: [ 
        { id: 'arm', title: 'แขน' }, 
        { id: 'torso', title: 'ลำตัว' }, 
        { id: 'leg', title: 'ขา' }, 
        { id: 'neck', title: 'คอ' }, 
        { id: 'full', title: 'ทั้งตัว' } 
    ],
};
// --- MediaPipe Joint Constants ---
export const KEYPOINT_INDICES = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3, 
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
    // เพิ่มทั้งหมด 33 joints ตามความจำเป็น
};

// 💡 Connections (ใช้ชื่อตาม MediaPipe Keypoint Naming Convention)
export const JOINT_CONNECTIONS = [
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    // Arms
    ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
    // Legs
    ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
    // Feet
    ['left_ankle', 'left_heel'], ['left_ankle', 'left_foot_index'],
    ['right_ankle', 'right_heel'], ['right_ankle', 'right_foot_index']
];


// --- API Configuration ---
export const API_BASE_URL = 'http://localhost:8000'; 
export const USER_ID = "trainer_chaiyapat"; // ใช้ User ID ที่สอดคล้องกับ Backend

// --- Helper Function ---
export function hexToRgb(hex) {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : null;
}