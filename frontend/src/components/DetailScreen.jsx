import React from 'react';
import { ICONS, PALETTE } from '../constants.jsx';

export default function DetailScreen({ exercise, onBack, onPlay }) {
    if (!exercise) return null;
    return (
        <div style={{ padding: '16px 16px 0', height: '100%', display: 'flex', flexDirection: 'column', boxSizing: 'border-box' }}>
            <div style={{ height: 220, borderRadius: 12, background: '#000', position: 'relative', flexShrink: 0, overflow: 'hidden' }}>
                <button onClick={onBack} style={{ background: 'rgba(0,0,0,0.6)', color: PALETTE.light, border: 'none', width: 30, height: 30, borderRadius: 15, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', position: 'absolute', right: 10, top: 10, zIndex: 1 }}>{ICONS.close}</button>
                <video 
                    key={exercise.preview_video_url}
                    src={exercise.preview_video_url} 
                    autoPlay 
                    loop 
                    muted 
                    playsInline 
                    style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 12 }} 
                    onError={(e) => e.target.style.display='none'} 
                />
            </div>

            <div style={{ marginTop: 20, flexGrow: 1, overflowY: 'auto' }} className="hide-scrollbar">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: 22 }}>{exercise.name}</div>
                        <div style={{ color: PALETTE.secondaryText, fontSize: 14, marginTop: 4 }}>{exercise.duration} · {exercise.level}</div>
                    </div>
                    <div>
                        <button style={{ background: PALETTE.primary, color: '#fff', padding: '12px 16px', borderRadius: 10, border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onPlay}>
                           {ICONS.play}
                        </button>
                    </div>
                </div>
                <p style={{ color: PALETTE.secondaryText, marginTop: 16, lineHeight: 1.6, fontSize: 14 }}>
                    {exercise.description || "ไม่มีคำอธิบายสำหรับท่านี้"}
                </p>
            </div>
        </div>
    );
}

