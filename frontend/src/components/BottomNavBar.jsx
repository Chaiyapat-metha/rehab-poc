import React from 'react';
import { ICONS, PALETTE, hexToRgb } from '../constants.jsx';

export default function BottomNavBar({ onPlay }) {
    return (
        <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, padding: '12px 18px 30px', background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(15px)', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', fontSize: 12, color: PALETTE.primary, cursor: 'pointer' }}>
                {ICONS.home}
                <div style={{ fontSize: 10, marginTop: 4 }}>หน้าหลัก</div>
            </div>
            <div onClick={onPlay} style={{ width: 68, height: 68, borderRadius: 34, background: PALETTE.primary, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `0 8px 20px rgba(${hexToRgb(PALETTE.primary)},0.3)`, cursor: 'pointer', transform: 'translateY(-20px)' }}>
                {ICONS.play}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', fontSize: 12, color: PALETTE.light, cursor: 'pointer' }}>
                {ICONS.profile}
                <div style={{ fontSize: 10, marginTop: 4 }}>โปรไฟล์</div>
            </div>
        </div>
    );
}

