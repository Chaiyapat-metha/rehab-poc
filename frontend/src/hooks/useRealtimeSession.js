import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE_URL, USER_ID } from '../constants.jsx';

const ANNOUNCEMENT_THROTTLE_MS = 3000; // 3 seconds debounce for TTS playback
const WINDOW_SIZE = 16; 

export function useRealtimeSession(exerciseId) {
    const [feedback, setFeedback] = useState({ display_text: 'กำลังเตรียมเซสชัน...', status: 'info', tts_text: null, wrong_joints: [] });
    const wsRef = useRef(null);
    const lastFeedbackTime = useRef(0);
    const isMounted = useRef(true);

    // 💡 Window buffer for pose data (to collect 16 frames before sending)
    const poseWindowBuffer = useRef([]);

    useEffect(() => {
        isMounted.current = true;
        const SESSION_ID = `${exerciseId}_${Date.now()}`;
        
        // Ensure protocol is 'ws' or 'wss'
        const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const backendUrl = `${wsProtocol}://${window.location.hostname}:8000/ws/live/${exerciseId}?user_id=${USER_ID}&session_id=${SESSION_ID}`;
        
        try {
            wsRef.current = new WebSocket(backendUrl);
        } catch (error) {
            console.error("WebSocket creation failed:", error);
            if(isMounted.current) setFeedback(prev => ({ ...prev, display_text: 'ไม่สามารถสร้างการเชื่อมต่อได้', status: 'error' }));
            return;
        }

        wsRef.current.onopen = () => {
            if (isMounted.current) setFeedback(prev => ({ ...prev, display_text: 'เชื่อมต่อแล้ว เริ่มได้เลย!', status: 'correct' }));
        };
        
        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (!isMounted.current) return;
            
            // 1. Update Visual Feedback immediately (display text and highlight)
             setFeedback(prev => ({ 
                ...prev, 
                display_text: data.display_text, 
                status: data.status,
                wrong_joints: data.wrong_joints || []
             }));
            
            // 2. TTS Throttling (Debounce the actual audio playback time)
            if (data.tts_text) { 
                const now = Date.now();
                // 💡 TTS Logic: Play audio only if 3 seconds have passed since the last announcement
                if (now - lastFeedbackTime.current > ANNOUNCEMENT_THROTTLE_MS) { 
                    setFeedback(prev => ({ 
                        ...prev, 
                        tts_text: data.tts_text, // Pass text to SessionScreen.jsx's useEffect
                    })); 
                    lastFeedbackTime.current = now;
                }
            } else {
                 setFeedback(prev => ({ ...prev, tts_text: null }));
            }
        };

        wsRef.current.onclose = () => {
             if (isMounted.current) setFeedback(prev => ({ ...prev, display_text: 'การเชื่อมต่อถูกตัด', status: 'error' }));
        };
        wsRef.current.onerror = (e) => {
            console.error("WebSocket Error:", e);
            if (isMounted.current) setFeedback(prev => ({ ...prev, display_text: 'เกิดข้อผิดพลาดในการเชื่อมต่อ', status: 'error' }));
        };

        return () => {
            isMounted.current = false;
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.close();
            }
        };
    }, [exerciseId]);

    const sendPoseData = useCallback((keypoints) => {
        // Collect frames into a buffer first
        poseWindowBuffer.current.push(keypoints);
        
        if (poseWindowBuffer.current.length >= WINDOW_SIZE) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                // 💡 FIX: Send the full window array with correct keys (T=16 frames)
                wsRef.current.send(JSON.stringify({ 
                    window_frames: poseWindowBuffer.current, 
                    user_id: USER_ID, 
                    exercise_id: exerciseId 
                }));
            }
            // 💡 FIX: Sliding window (shift by 1 frame) instead of hopping window
            poseWindowBuffer.current = poseWindowBuffer.current.slice(1);
        }
    }, [exerciseId]);

    return { feedback, sendPoseData };
}