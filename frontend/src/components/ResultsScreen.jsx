// File: C:\Users\chaiyapat metha\Desktop\AI Project\rehab-poc\frontend\src\components\ResultsScreen.jsx

import React, { useState, useEffect } from 'react';
import { PALETTE, API_BASE_URL, USER_ID } from '../constants.jsx';

export default function ResultsScreen({ onDone, currentExerciseId, onThresholdCommit }) {
    const [summary, setSummary] = useState("กำลังประมวลผลสรุปผล...");
    const [proposal, setProposal] = useState(null);
    const [isAccepted, setIsAccepted] = useState(false);
    
    // 💡 Fetch LLM Summary and Threshold Proposal
    useEffect(() => {
        const fetchResults = async () => {
            if (!currentExerciseId) return;

            // 1. Fetch LLM Summary (Goal: Summarize the session via RAG chain)
            try {
                const summaryResponse = await fetch(`${API_BASE_URL}/api/rag/summarize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: USER_ID, question: `สรุปผลการออกกำลังกาย ${currentExerciseId} ที่ผ่านมา พร้อมข้อแนะนำ` })
                });
                const summaryData = await summaryResponse.json();
                setSummary(summaryData.answer || "ไม่สามารถสร้างสรุปผลจาก AI ได้");
            } catch (err) {
                console.error("LLM Summary fetch failed:", err);
                setSummary("เกิดข้อผิดพลาดในการประมวลผลสรุปผลจาก AI");
            }

            // 2. Fetch Threshold Proposal
            try {
                // 💡 FIX: Corrected API endpoint structure for POST request with exercise_id in path
                const proposalResponse = await fetch(`${API_BASE_URL}/user/${USER_ID}/thresholds/propose/${currentExerciseId}`, { method: 'POST' });
                const proposalData = await proposalResponse.json();
                
                // ตรวจสอบว่ามีการเสนอการปรับความยากจริงหรือไม่
                if (proposalData.proposed_thresholds && Object.keys(proposalData.proposed_thresholds).length > 0) {
                    setProposal(proposalData.proposed_thresholds);
                } else {
                    setProposal(null);
                }
            } catch (err) {
                console.error("Threshold Proposal fetch failed:", err);
                setProposal(null); // เคลียร์ข้อเสนอถ้าเกิดข้อผิดพลาด
            }
        };

        fetchResults();
    }, [currentExerciseId]);


    const handleDone = () => {
        if (proposal && isAccepted) {
            // 💡 Commit Threshold if user accepts
            onThresholdCommit(proposal); 
        }
        onDone();
    };


    return (
        <div style={{ padding: '20px 16px 90px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', boxSizing: 'border-box', overflowY: 'auto' }} className="hide-scrollbar">
            <h1 style={{ fontSize: 24, fontWeight: 700 }}>สรุปผลการฝึก</h1>
            
            {/* AI Summary Section */}
            <div style={{ background: PALETTE.uiGrey, padding: 15, borderRadius: 12, width: '100%', margin: '20px 0', textAlign: 'left' }}>
                <div style={{ fontSize: 14, color: PALETTE.primary, fontWeight: 600, marginBottom: 10 }}>AI Coach Summary</div>
                <p style={{ color: PALETTE.light, lineHeight: 1.5, fontSize: 14 }}>
                    {summary}
                </p>
            </div>

            {/* Threshold Proposal Section (Req. 8) */}
            {proposal ? (
                <div style={{textAlign: 'left', width: '100%', marginBottom: 30}}>
                    <div style={{fontSize: 16, fontWeight: 600, marginBottom: 10}}>ระบบเสนอการปรับความยาก</div>
                    <label style={{display: 'flex', alignItems: 'center', gap: 10, padding: 15, background: PALETTE.uiGrey, borderRadius: 8, cursor: 'pointer'}}>
                        <input 
                            type="checkbox" 
                            style={{width: 20, height: 20}} 
                            checked={isAccepted} 
                            onChange={(e) => setIsAccepted(e.target.checked)}
                        />
                        <span>ยินยอมให้ระบบปรับความยากอัตโนมัติ (เช่น ปรับเกณฑ์ข้อผิดพลาดให้เข้มงวดขึ้น)</span>
                    </label>
                    <div style={{fontSize: 12, color: PALETTE.secondaryText, marginTop: 8}}>การปรับความยากที่เสนอ: {Object.keys(proposal).map(k => `${k} -> ${proposal[k].toFixed(2)}`).join(', ')}</div>
                </div>
            ) : (
                <div style={{textAlign: 'left', width: '100%', marginBottom: 30, color: PALETTE.secondaryText}}>คุณทำได้ดีมาก! ไม่มีข้อเสนอการปรับความยากในรอบนี้</div>
            )}

            <button onClick={handleDone} style={{ background: PALETTE.primary, color: '#fff', padding: '15px 30px', borderRadius: 999, border: 'none', cursor: 'pointer', fontSize: 16, fontWeight: 600 }}>
                เสร็จสิ้น
            </button>
        </div>
    );
}