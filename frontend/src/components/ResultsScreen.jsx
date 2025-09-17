import React from 'react';
import { PALETTE } from '../constants.jsx';

export default function ResultsScreen({ onDone }) {
    return (
        <div style={{ padding: '20px 16px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', boxSizing: 'border-box' }}>
            <h1 style={{ fontSize: 24, fontWeight: 700 }}>สรุปผล</h1>
            <div style={{ background: PALETTE.uiGrey, padding: 20, borderRadius: 12, width: '100%', margin: '20px 0' }}>
                <div style={{ fontSize: 16, color: PALETTE.secondaryText }}>ความแม่นยำ</div>
                <div style={{ fontSize: 48, fontWeight: 700, color: PALETTE.primary, margin: '10px 0' }}>87%</div>
                <div style={{ fontSize: 14, color: PALETTE.secondaryText }}>ข้อผิดพลาดหลัก: ข้อศอกซ้าย</div>
            </div>
            
            <div style={{textAlign: 'left', width: '100%', marginBottom: 30}}>
                <label style={{display: 'flex', alignItems: 'center', gap: 10, padding: 15, background: PALETTE.uiGrey, borderRadius: 8, cursor: 'pointer'}}>
                    <input type="checkbox" style={{width: 20, height: 20}}/>
                    <span>ยินยอมให้ระบบปรับความยากอัตโนมัติ</span>
                </label>
            </div>

            <button onClick={onDone} style={{ background: PALETTE.primary, color: '#fff', padding: '15px 30px', borderRadius: 999, border: 'none', cursor: 'pointer', fontSize: 16, fontWeight: 600 }}>
                เสร็จสิ้น
            </button>
        </div>
    );
}

