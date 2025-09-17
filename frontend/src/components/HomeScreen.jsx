import React from 'react';
import { ICONS, MOCK_DATA, PALETTE } from '../constants.jsx';

// This component is now simpler: it receives all data as props from App.jsx
export default function HomeScreen({ exercisesByCat, isLoading, error, onSelectExercise }) {

    if (isLoading) {
        return <div style={{ padding: 20, textAlign: 'center', color: PALETTE.light }}>กำลังโหลดข้อมูลท่าออกกำลังกาย...</div>;
    }

    if (error) {
        return <div style={{ padding: 20, textAlign: 'center', color: PALETTE.error }}>เกิดข้อผิดพลาดในการโหลดข้อมูล: {error}</div>;
    }

    return (
        <div style={{ padding: '0 16px 90px', height: '100%', overflowY: 'auto', boxSizing: 'border-box' }} className="hide-scrollbar">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, paddingTop: 10 }}>
                <div style={{ fontSize: 24, fontWeight: 700 }}>ออกกำลังกาย</div>
                <div style={{ width: 36, height: 36, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', border: `1px solid ${PALETTE.uiGrey}` }}>
                    {ICONS.search}
                </div>
            </div>

            <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 24 }}>
                {MOCK_DATA.categories.map(cat => (
                    <div key={cat.id}>
                        <div style={{ color: PALETTE.light, marginBottom: 12, fontSize: 18, fontWeight: 600 }}>{cat.title}</div>
                        <div id={`category-scroll-${cat.id}`} className="hide-scrollbar" style={{ display: 'flex', gap: 12, overflowX: 'auto', paddingBottom: 10, minHeight: 148 }}>
                            {exercisesByCat && exercisesByCat[cat.id] && exercisesByCat[cat.id].length > 0 ? (
                                exercisesByCat[cat.id].map(ex => (
                                    <div key={ex.name} onClick={() => onSelectExercise(ex, cat.id)} role="button" tabIndex={0} style={{ minWidth: 144, cursor: 'pointer' }}>
                                        <img 
                                            src={ex.thumbnail_url} 
                                            alt={ex.name} 
                                            style={{ width: 144, height: 128, borderRadius: 12, objectFit: 'cover', background: PALETTE.uiGrey }} 
                                            onError={(e) => { e.target.onerror = null; e.target.src=`https://placehold.co/288x256/ff4d4f/dde1e6?text=Image+Error`; }}
                                        />
                                        <div style={{ color: PALETTE.secondaryText, fontSize: 12, marginTop: 6 }}>{ex.name}</div>
                                    </div>
                                ))
                            ) : (
                                <div style={{ color: PALETTE.muted, fontSize: 14, fontStyle: 'italic', display: 'flex', alignItems: 'center', height: 128 }}>
                                    ไม่มีท่าในหมวดหมู่นี้
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

