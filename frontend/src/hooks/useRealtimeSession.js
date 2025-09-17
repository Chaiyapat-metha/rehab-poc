import { useState, useEffect, useRef, useCallback } from 'react';

const DEBOUNCE_TIME = 1500;
const ERROR_THRESHOLD = 3;

export function useRealtimeSession() {
    const [feedback, setFeedback] = useState({ message: 'กำลังเตรียมเซสชัน...', status: 'info' });
    const wsRef = useRef(null);
    const errorCounter = useRef(0);
    const feedbackTimer = useRef(null);
    const isMounted = useRef(true);

    useEffect(() => {
        isMounted.current = true;
        const SESSION_ID = "session_" + Math.random().toString(36).substr(2, 9);
        const backendUrl = `ws://${window.location.hostname}:8000/ws/live/${SESSION_ID}`;
        
        try {
            wsRef.current = new WebSocket(backendUrl);
        } catch (error) {
            console.error("WebSocket creation failed:", error);
            if(isMounted.current) setFeedback({ message: 'ไม่สามารถสร้างการเชื่อมต่อได้', status: 'error' });
            return;
        }

        wsRef.current.onopen = () => {
            if (isMounted.current) setFeedback({ message: 'เชื่อมต่อแล้ว เริ่มได้เลย!', status: 'correct' });
        };
        
        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (!isMounted.current) return;

            if (data.status === 'error') {
                errorCounter.current += 1;
            } else {
                errorCounter.current = 0;
                setFeedback({ message: data.message, status: 'correct' });
            }

            if (feedbackTimer.current) clearTimeout(feedbackTimer.current);

            if (errorCounter.current >= ERROR_THRESHOLD) {
                setFeedback({ message: data.message, status: 'error' });
                feedbackTimer.current = setTimeout(() => {
                    if (isMounted.current) errorCounter.current = 0;
                }, DEBOUNCE_TIME);
            }
        };

        wsRef.current.onclose = () => {
             if (isMounted.current) setFeedback({ message: 'การเชื่อมต่อถูกตัด', status: 'error' });
        };
        wsRef.current.onerror = (e) => {
            console.error("WebSocket Error:", e);
            if (isMounted.current) setFeedback({ message: 'เกิดข้อผิดพลาดในการเชื่อมต่อ', status: 'error' });
        };

        return () => {
            isMounted.current = false;
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.close();
            }
        };
    }, []);

    const sendPoseData = useCallback((keypoints) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ keypoints }));
        }
    }, []);

    return { feedback, sendPoseData };
}

