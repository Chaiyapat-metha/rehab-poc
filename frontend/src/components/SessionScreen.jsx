// import React, { useState, useEffect, useRef } from 'react';
// import { useRealtimeSession } from '../hooks/useRealtimeSession.js';
// import { ICONS, PALETTE, JOINT_CONNECTIONS, KEYPOINT_INDICES } from '../constants.jsx'; 

// // 💡 Helper function for TTS (Text-to-Speech)
// const speakFeedback = (text) => {
//     if ('speechSynthesis' in window) {
//         const utterance = new SpeechSynthesisUtterance(text);
//         utterance.lang = 'th-TH'; 
//         utterance.rate = 0.95; 
//         window.speechSynthesis.speak(utterance);
//     } else {
//         console.warn("Speech Synthesis API not supported.");
//     }
// };

// export default function SessionScreen({ exercise, onEnd }) {
//     // Note: useRealtimeSession ควรถูกปรับให้รับ exercise.id และส่ง USER_ID (จาก constants.jsx)
//     const { feedback, sendPoseData } = useRealtimeSession(exercise.id); 
//     const [isPlaying, setIsPlaying] = useState(true);
//     const [isMuted, setIsMuted] = useState(false);
//     const [isUserFullscreen, setIsUserFullscreen] = useState(false);
    
//     const videoRef = useRef(null);
//     const poseDetectorRef = useRef(null);
//     const animationFrameId = useRef(null);
//     const canvasRef = useRef(null); // Ref สำหรับ Canvas Overlay

//     // Effect to play TTS feedback
//     useEffect(() => {
//         // 💡 Logic เล่น TTS เมื่อมีข้อผิดพลาด หรือมีข้อความ TTS ใหม่จาก Backend
//         if (feedback.status !== 'info' && feedback.tts_text) { 
//             if (!isMuted) {
//                 speakFeedback(feedback.tts_text);
//             }
//         }
//     }, [feedback.tts_text, feedback.status, isMuted]); 

//     // 💡 Drawing Logic (ย้ายมาไว้ข้างนอกเพื่อให้ Scoping ชัดเจน)
//     const drawUserSkeleton = (ctx, keypoints, width, height, wrongJoints) => {
//         // ต้องมั่นใจว่า videoRef.current มีค่าก่อนใช้งาน
//         if (!videoRef.current || videoRef.current.videoWidth === 0) return;
        
//         const scaleX = width / videoRef.current.videoWidth;
//         const scaleY = height / videoRef.current.videoHeight;
        
//         ctx.strokeStyle = PALETTE.primary;
//         ctx.lineWidth = 4;

//         // วาดข้อต่อ
//         for (const kp of keypoints) {
//             const x = kp.x * scaleX;
//             const y = kp.y * scaleY;
//             const radius = 5;

//             // 💡 Highlight ข้อต่อที่ผิด
//             if (wrongJoints && wrongJoints.includes(KEYPOINT_INDICES[kp.name])) { 
//                 ctx.fillStyle = PALETTE.error; // สีแดง
//                 ctx.globalAlpha = 0.8;
//             } else {
//                 ctx.fillStyle = PALETTE.primary;
//                 ctx.globalAlpha = 0.5;
//             }
            
//             ctx.beginPath();
//             ctx.arc(x, y, radius, 0, 2 * Math.PI);
//             ctx.fill();
//         }
        
//         // วาดเส้นเชื่อมต่อ (Bones)
//         ctx.globalAlpha = 0.7;
//         for (const [startName, endName] of JOINT_CONNECTIONS) {
//             // NOTE: Keypoint Naming Convention ใน MediaPipe คือ snake_case (e.g., 'left_shoulder')
//             const startKp = keypoints.find(kp => kp.name === startName);
//             const endKp = keypoints.find(kp => kp.name === endName);

//             if (startKp && endKp) {
//                 ctx.beginPath();
//                 ctx.moveTo(startKp.x * scaleX, startKp.y * scaleY);
//                 ctx.lineTo(endKp.x * scaleX, endKp.y * scaleY);
//                 ctx.stroke();
//             }
//         }
//     };


//     // Effect for Camera and Pose Detection
//     useEffect(() => {
//         let isCancelled = false;
//         let poses; // ใช้ let นอก try block เพื่อ Scoping ที่ถูกต้อง

//         async function setupCameraAndPose() {
//             try {
//                 // ... (Camera setup logic เดิม) ...
//                 const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
//                 if (isCancelled || !videoRef.current) return;
                
//                 videoRef.current.srcObject = stream;
//                 videoRef.current.muted = true; 
//                 await new Promise(resolve => (videoRef.current.onloadedmetadata = resolve));
//                 videoRef.current.play();

//                 if (isCancelled || !window.poseDetection) {
//                     console.error("Pose Detection library not loaded.");
//                     return;
//                 }
//                 const model = window.poseDetection.SupportedModels.BlazePose;
//                 const detectorConfig = { runtime: 'mediapipe', solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose', modelType: 'full' };
//                 poseDetectorRef.current = await window.poseDetection.createDetector(model, detectorConfig);

//                 renderPose();
//             } catch (err) {
//                 console.error("Camera or Pose Detector setup failed:", err);
//             }
//         }

//         const renderPose = async () => {
//             if (isCancelled) return;
            
//             const video = videoRef.current;
//             const canvas = canvasRef.current;
//             const ctx = canvas ? canvas.getContext('2d') : null;
            
//             if (isPlaying && video && video.readyState >= 3 && video.videoWidth > 0 && poseDetectorRef.current && ctx) {
//                 try {
//                     poses = await poseDetectorRef.current.estimatePoses(video); // กำหนดค่า poses ที่นี่
//                 } catch (error) {
//                     console.error("Error during pose estimation:", error);
//                     // หากเกิด error เราจะข้าม Logic ที่ใช้ poses
//                 }

//                 // 💡 NEW: Logic วาดโครงกระดูกและส่งข้อมูล (ใช้ตัวแปร poses ที่ถูกกำหนดค่าแล้ว)
//                 if (poses && poses.length > 0 && poses[0].keypoints3D) {
                    
//                     const pose = poses[0];
//                     sendPoseData(pose.keypoints3D); // ส่ง World 3D coordinates
                    
//                     // วาดโครงกระดูก
//                     ctx.clearRect(0, 0, canvas.width, canvas.height);
//                     ctx.save();
                    
//                     const isFlipped = true; 
//                     if (isFlipped) {
//                         ctx.translate(canvas.width, 0);
//                         ctx.scale(-1, 1);
//                     }
                    
//                     drawUserSkeleton(ctx, pose.keypoints, canvas.width, canvas.height, feedback.wrong_joints);
                    
//                     ctx.restore();
//                 }
//             }
//             animationFrameId.current = requestAnimationFrame(renderPose);
//         };

//         setupCameraAndPose();

//         return () => {
//             // ... (Cleanup logic เดิม) ...
//             isCancelled = true;
//             if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
//             if (videoRef.current?.srcObject) {
//                 videoRef.current.srcObject.getTracks().forEach(track => track.stop());
//             }
//         };
//     }, [isPlaying, sendPoseData, feedback.wrong_joints]); 
    
//     // ... (FullscreenView และ PipView components เดิม) ...

//     return (
//         <div style={{ position: 'absolute', inset: 0, background: PALETTE.bg, color: PALETTE.light }}>
//             {/* 1. Video Clip / Avatar Model (Fullscreen/PIP) */}
//             <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
//                 {/* Top Half: Instruction Video Clip */}
//                 <div style={{ height: '50%', background: PALETTE.uiGrey, position: 'relative' }}>
//                     {exercise.clip_url ? (
//                         <video 
//                             src={exercise.clip_url} 
//                             autoPlay={isPlaying} 
//                             loop 
//                             playsInline 
//                             style={{ width: '100%', height: '100%', objectFit: 'cover' }}
//                             onError={(e) => { e.target.style.display='none'; }}
//                         />
//                     ) : (
//                         <div style={{ textAlign: 'center', color: PALETTE.light, padding: 50 }}>ไม่มีไฟล์วิดีโอสอน</div>
//                     )}
//                 </div>
                
//                 {/* Bottom Half: User Camera View + Overlay */}
//                 <div style={{ height: '50%', background: '#000', position: 'relative' }}>
//                     <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
//                     {/* 💡 CANVAS OVERLAY (วางทับ User Video) */}
//                     <canvas 
//                         ref={canvasRef}
//                         // transform: scaleX(-1) เพื่อ mirror canvas ให้ตรงกับ video
//                         style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'scaleX(-1)' }}
//                         width={640} 
//                         height={480} 
//                     />
//                 </div>
//             </div>

//             {/* 2. Control and Feedback UI (Overlay) */}
//             <div style={{ position: 'absolute', inset: 0 }}>
//                 {/* Feedback Box */}
//                  <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: 120, padding: '10px 20px', borderRadius: 12, border: `1px solid ${feedback.status === 'error' ? PALETTE.error : 'rgba(106, 123, 145, 0.5)'}`, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', minHeight: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', transition: 'all 0.3s ease', boxShadow: '0 4px 15px rgba(0,0,0,0.3)', color: feedback.status === 'error' ? PALETTE.error : PALETTE.light }}>
//                     <div style={{ fontWeight: 600 }}>{feedback.display_text}</div>
//                  </div>

//                 {/* Control Buttons */}
//                 <div style={{ position: 'absolute', left: 0, right: 0, bottom: 30, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 24 }}>
//                     <button onClick={() => setIsMuted(m => !m)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isMuted ? ICONS.mute : ICONS.unmute}</button>
//                     <button onClick={onEnd} style={{width: 72, height: 72, borderRadius: 36, border:'none', background: PALETTE.error, color: PALETTE.light, display:'flex', alignItems:'center', justifyContent:'center', fontSize: 16, fontWeight: 700, cursor: 'pointer'}}>End</button>
//                     <button onClick={() => setIsPlaying(p => !p)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isPlaying ? ICONS.pause : ICONS.play}</button>
//                 </div>
//             </div>
//         </div>
//     );
// }


// C:\Users\chaiyapat metha\Desktop\AI Project\rehab-poc\frontend\src\components\SessionScreen.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react'; 
import { useRealtimeSession } from '../hooks/useRealtimeSession.js';
import { ICONS, PALETTE, JOINT_CONNECTIONS, KEYPOINT_INDICES } from '../constants.jsx'; 

// 💡 Helper function for TTS (Text-to-Speech)
const speakFeedback = (text) => {
    if ('speechSynthesis' in window) {
        // Prevent overlapping speech
        window.speechSynthesis.cancel(); 
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'th-TH'; 
        utterance.rate = 0.95; 
        window.speechSynthesis.speak(utterance);
    } else {
        console.warn("Speech Synthesis API not supported.");
    }
};

export default function SessionScreen({ exercise, onEnd }) {
    const { feedback, sendPoseData } = useRealtimeSession(exercise.id); 
    const [isPlaying, setIsPlaying] = useState(true);
    const [isMuted, setIsMuted] = useState(false);
    const [isUserFullscreen, setIsUserFullscreen] = useState(false); 
    
    const videoRef = useRef(null);
    const poseDetectorRef = useRef(null);
    const animationFrameId = useRef(null);
    const canvasRef = useRef(null); // Ref สำหรับ Canvas Overlay

    // Effect to play TTS feedback
    useEffect(() => {
        if (feedback.tts_text) { 
            if (!isMuted) {
                speakFeedback(feedback.tts_text);
            }
        }
    }, [feedback.tts_text, isMuted]); 

    // 💡 Drawing Logic (ใช้สีขาว/เทาอ่อนตามมาตรฐาน MediaPipe)
    const drawUserSkeleton = (ctx, keypoints, width, height, wrongJoints) => {
         if (!videoRef.current || videoRef.current.videoWidth === 0) return;
        
        const scaleX = width / videoRef.current.videoWidth;
        const scaleY = height / videoRef.current.videoHeight;
        
        ctx.strokeStyle = '#FFFFFF'; // White lines
        ctx.lineWidth = 2; 

        // วาดข้อต่อ
        for (const kp of keypoints) {
            const x = kp.x * scaleX;
            const y = kp.y * scaleY;
            const radius = 4; 

            // 💡 Highlight ข้อต่อที่ผิด (Req. Highlight)
            if (wrongJoints && KEYPOINT_INDICES[kp.name] !== undefined && wrongJoints.includes(KEYPOINT_INDICES[kp.name])) { 
                ctx.fillStyle = PALETTE.error; // สีแดง
                ctx.globalAlpha = 1.0; 
            } else {
                ctx.fillStyle = '#DDDDDD'; // Grey/White dots
                ctx.globalAlpha = 0.6;
            }
            
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // วาดเส้นเชื่อมต่อ (Bones)
        ctx.globalAlpha = 0.5;
        for (const [startName, endName] of JOINT_CONNECTIONS) {
            const startKp = keypoints.find(kp => kp.name === startName);
            const endKp = keypoints.find(kp => kp.name === endName);

            if (startKp && endKp) {
                ctx.beginPath();
                ctx.moveTo(startKp.x * scaleX, startKp.y * scaleY);
                ctx.lineTo(endKp.x * scaleX, endKp.y * scaleY);
                ctx.stroke();
            }
        }
    };

    // Effect for Camera and Pose Detection
    useEffect(() => {
        let isCancelled = false;
        let poses; 

        async function setupCameraAndPose() {
            // ... (Camera setup logic remains the same) ...
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                if (isCancelled || !videoRef.current) return;
                
                videoRef.current.srcObject = stream;
                videoRef.current.muted = true; 
                await new Promise(resolve => (videoRef.current.onloadedmetadata = resolve));
                videoRef.current.play();

                if (isCancelled || !window.poseDetection) {
                    console.error("Pose Detection library not loaded.");
                    return;
                }
                
                const model = window.poseDetection.SupportedModels.BlazePose;
                const detectorConfig = { runtime: 'mediapipe', solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose', modelType: 'full' };
                poseDetectorRef.current = await window.poseDetection.createDetector(model, detectorConfig);

                renderPose();
            } catch (err) {
                console.error("Camera or Pose Detector setup failed:", err);
            }
        }

        const renderPose = async () => {
            if (isCancelled) return;
            
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas ? canvas.getContext('2d') : null;
            
            if (isPlaying && video && video.readyState >= 3 && video.videoWidth > 0 && poseDetectorRef.current && ctx) {
                try {
                    poses = await poseDetectorRef.current.estimatePoses(video); 
                } catch (error) {
                    console.error("Error during pose estimation:", error);
                }

                let keypoints3D_sanitized = null;
                
                if (poses && poses.length > 0 && poses[0].keypoints3D) {
                    const pose = poses[0];
                    
                    // 1. Sanitize World Coordinates (Zero Padding on null joints)
                    keypoints3D_sanitized = pose.keypoints3D.map(kp => {
                        // Final Sanitize Logic
                        if (kp && typeof kp.x === 'number') { 
                            return [kp.x, kp.y, kp.z]; 
                        } else {
                            return [0.0, 0.0, 0.0]; // Zero Padding
                        }
                    });
                    
                    // 2. Draw 2D Skeleton
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.save();
                    ctx.translate(canvas.width, 0);
                    ctx.scale(-1, 1);
                    drawUserSkeleton(ctx, pose.keypoints, canvas.width, canvas.height, feedback.wrong_joints);
                    ctx.restore();
                    
                } else {
                    // If Pose is not detected (Zero Frame)
                    keypoints3D_sanitized = Array(33).fill([0.0, 0.0, 0.0]);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
                
                // 3. Send Sanitized 3D Data to WebSocket (This is 1 frame, useRealtimeSession handles the 16-frame windowing)
                if (keypoints3D_sanitized) {
                    sendPoseData(keypoints3D_sanitized); 
                }
            }
            animationFrameId.current = requestAnimationFrame(renderPose);
        };

        setupCameraAndPose();

        return () => {
            isCancelled = true;
            if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
            if (videoRef.current?.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
        };
    }, [isPlaying, sendPoseData, feedback.wrong_joints]); 
    
    const handleVideoEnd = (e) => {
        const video = e.target;
        // หยุดวิดีโอทันทีเมื่อเล่นจบเพื่อให้ค้างอยู่ที่เฟรมสุดท้าย
        video.pause();
    };

    return (
        <div style={{ position: 'absolute', inset: 0, background: PALETTE.bg, color: PALETTE.light }}>
            {/* 1. Video Clip / Avatar Model */}
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
                {/* Top Half: Instruction Video Clip */}
                <div style={{ height: '50%', background: PALETTE.uiGrey, position: 'relative' }}>
                    {exercise.clip_url ? (
                        <video 
                            src={exercise.clip_url} 
                            autoPlay={isPlaying} 
                            // ❌ REMOVED: loop เพื่อให้เล่นแค่ครั้งเดียว
                            playsInline 
                            onEnded={handleVideoEnd} // 💡 ADDED: Handler สำหรับหยุดที่เฟรมสุดท้าย
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => { e.target.style.display='none'; }}
                        />
                    ) : (
                        <div style={{ textAlign: 'center', color: PALETTE.light, padding: 50 }}>ไม่มีไฟล์วิดีโอสอน</div>
                    )}
                </div>
                
                {/* Bottom Half: User Camera View + Overlay */}
                <div style={{ height: '50%', background: '#000', position: 'relative' }}>
                    {/* Video and Canvas must be the same size for overlay to work */}
                    <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                    <canvas 
                        ref={canvasRef}
                        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'scaleX(-1)' }}
                        width={640} 
                        height={480} 
                    />
                </div>
            </div>

            {/* 2. Control and Feedback UI (Overlay) */}
            <div style={{ position: 'absolute', inset: 0 }}>
                {/* Feedback Box */}
                <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: 120, padding: '10px 20px', borderRadius: 12, border: `1px solid ${feedback.status === 'error' ? PALETTE.error : 'rgba(106, 123, 145, 0.5)'}`, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', minHeight: 40, display: 'flex', alignItems:'center', justifyContent: 'center', textAlign: 'center', transition: 'all 0.3s ease', boxShadow: '0 4px 15px rgba(0,0,0,0.3)', color: feedback.status === 'error' ? PALETTE.error : PALETTE.light }}>
                    <div style={{ fontWeight: 600 }}>{feedback.display_text}</div>
                </div>

                {/* Control Buttons */}
                <div style={{ position: 'absolute', left: 0, right: 0, bottom: 30, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 24 }}>
                    <button onClick={() => setIsMuted(m => !m)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isMuted ? ICONS.mute : ICONS.unmute}</button>
                    <button onClick={onEnd} style={{width: 72, height: 72, borderRadius: 36, border:'none', background: PALETTE.error, color: PALETTE.light, display:'flex', alignItems:'center', justifyContent:'center', fontSize: 16, fontWeight: 700, cursor: 'pointer'}}>End</button>
                    <button onClick={() => setIsPlaying(p => !p)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isPlaying ? ICONS.pause : ICONS.play}</button>
                </div>
            </div>
        </div>
    );
}
