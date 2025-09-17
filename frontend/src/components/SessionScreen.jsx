import React, { useState, useEffect, useRef } from 'react';
import { useRealtimeSession } from '../hooks/useRealtimeSession.js';
import { ICONS, PALETTE } from '../constants.jsx';
import AvatarViewer from './AvatarViewer.jsx';

export default function SessionScreen({ exercise, onEnd }) {
    const { feedback, sendPoseData } = useRealtimeSession();
    const [isPlaying, setIsPlaying] = useState(true);
    const [isMuted, setIsMuted] = useState(false);
    const [isUserFullscreen, setIsUserFullscreen] = useState(false);
    const [avatarAnimationData, setAvatarAnimationData] = useState(null);
    
    const videoRef = useRef(null);
    const poseDetectorRef = useRef(null);
    const animationFrameId = useRef(null);

    // Effect to fetch animation data for the avatar
    useEffect(() => {
        if (!exercise) return;
        async function fetchAnimationData() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/exercises/${exercise.name}/frames`);
                const data = await response.json();
                if (data.status === 'success') {
                    setAvatarAnimationData(data.frames);
                }
            } catch (err) {
                console.error("Failed to fetch avatar animation data:", err);
            }
        }
        fetchAnimationData();
    }, [exercise]);

    
    // Effect for Camera and Pose Detection
    useEffect(() => {
        let isCancelled = false;

        async function setupCameraAndPose() {
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
            if (isPlaying && video && video.readyState >= 3 && video.videoWidth > 0 && poseDetectorRef.current) {
                try {
                    const poses = await poseDetectorRef.current.estimatePoses(video);
                    if (!isCancelled && poses && poses.length > 0 && poses[0].keypoints3D) {
                        sendPoseData(poses[0].keypoints3D);
                    }
                } catch (error) {
                    console.error("Error during pose estimation:", error);
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
    }, [isPlaying, sendPoseData]); 
    
    const FullscreenView = ({ children, onClick }) => (
        <div onClick={onClick} style={{ position: 'absolute', inset: 0, background: '#000', color: PALETTE.light, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', overflow: 'hidden' }}>
            {children}
        </div>
    );

    const PipView = ({ children, onClick }) => (
        <div onClick={onClick} style={{ position: 'absolute', right: 16, top: 60, width: 110, height: 160, borderRadius: 12, background: PALETTE.bg, color: PALETTE.light, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 6px 18px rgba(0,0,0,0.4)', cursor: 'pointer', overflow: 'hidden', border: `2px solid ${PALETTE.muted}` }}>
            {children}
        </div>
    );

    return (
        <div style={{ position: 'absolute', inset: 0, background: PALETTE.bg, color: PALETTE.light }}>
            {isUserFullscreen ? (
                <>
                    <FullscreenView onClick={() => setIsUserFullscreen(false)}>
                        <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                    </FullscreenView>
                    <PipView onClick={() => setIsUserFullscreen(false)}>
                         <AvatarViewer animationData={avatarAnimationData} isPlaying={isPlaying} />
                    </PipView>
                </>
            ) : (
                <>
                    <FullscreenView onClick={() => setIsUserFullscreen(true)}>
                        <AvatarViewer animationData={avatarAnimationData} isPlaying={isPlaying} />
                    </FullscreenView>
                    <PipView onClick={() => setIsUserFullscreen(true)}>
                        <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                    </PipView>
                </>
            )}
            <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: 120, padding: '10px 20px', borderRadius: 12, border: `1px solid ${feedback.status === 'error' ? PALETTE.error : 'rgba(106, 123, 145, 0.5)'}`, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', minHeight: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', transition: 'all 0.3s ease', boxShadow: '0 4px 15px rgba(0,0,0,0.3)', color: feedback.status === 'error' ? PALETTE.error : PALETTE.light }}>
                <div style={{ fontWeight: 600 }}>{feedback.message}</div>
            </div>
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: 30, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 24 }}>
                 <button onClick={() => setIsMuted(m => !m)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isMuted ? ICONS.mute : ICONS.unmute}</button>
                 <button onClick={onEnd} style={{width: 72, height: 72, borderRadius: 36, border:'none', background: PALETTE.error, color: PALETTE.light, display:'flex', alignItems:'center', justifyContent:'center', fontSize: 16, fontWeight: 700, cursor: 'pointer'}}>End</button>
                 <button onClick={() => setIsPlaying(p => !p)} style={{width: 56, height: 56, borderRadius: 28, border:'none', color: PALETTE.light, background: PALETTE.uiGrey, display:'flex', alignItems:'center', justifyContent:'center', cursor: 'pointer'}}>{isPlaying ? ICONS.pause : ICONS.play}</button>
            </div>
        </div>
    );
}

