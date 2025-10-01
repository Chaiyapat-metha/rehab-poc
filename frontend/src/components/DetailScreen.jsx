import React, { useRef, useMemo, useState } from 'react';
import { ICONS, PALETTE } from '../constants.jsx';

export default function DetailScreen({ exercise, exercisesInCat, onBack, onPlay, onSelect }) {
    if (!exercise) return null;

    const previewRef = useRef(null);
    const touchStartX = useRef(0);
    const touchEndX = useRef(0);
    const isDragging = useRef(false);
    const swipeThreshold = 50; 

    // 1. จัดเรียงรายการทั้งหมดเพื่อสร้าง Carousel
    const orderedExercises = useMemo(() => {
        if (!exercisesInCat || exercisesInCat.length === 0) return [exercise];
        
        const currentIndex = exercisesInCat.findIndex(ex => ex.id === exercise.id);
        
        return [
            exercisesInCat[(currentIndex - 1 + exercisesInCat.length) % exercisesInCat.length], // Previous
            exercise, 
            exercisesInCat[(currentIndex + 1) % exercisesInCat.length] // Next
        ];
    }, [exercise, exercisesInCat]);

    const handleSelectNewExercise = (delta) => {
        if (Math.abs(delta) < swipeThreshold) return;

        let nextExercise;
        if (delta < -swipeThreshold) { // Swipe Left -> Next Exercise
            nextExercise = orderedExercises[2]; 
        } else { // Swipe Right -> Previous Exercise
            nextExercise = orderedExercises[0]; 
        }
        
        // Call parent handler to update selectedExercise in App.jsx
        if (nextExercise.id !== exercise.id) {
             onSelect(nextExercise); 
        }
    };
    
    // --- Touch Handlers (Mobile) ---
    const handleTouchStart = (e) => {
        touchStartX.current = e.nativeEvent.touches[0].clientX;
    };

    const handleTouchMove = (e) => {
        touchEndX.current = e.nativeEvent.touches[0].clientX;
        // ป้องกันการ Scroll หน้าจอหลักเมื่อกำลังลาก Preview
        e.preventDefault(); 
    };
    
    const handleTouchEnd = () => {
        const delta = touchEndX.current - touchStartX.current;
        handleSelectNewExercise(delta);
        touchEndX.current = touchStartX.current; // Reset
    };

    // --- Mouse Handlers (Desktop/Drag) ---
    const handleMouseDown = (e) => {
        isDragging.current = true;
        touchStartX.current = e.clientX;
        e.preventDefault(); 
    };

    const handleMouseMove = (e) => {
        if (!isDragging.current) return;
        touchEndX.current = e.clientX;
    };
    
    const handleMouseUp = () => {
        if (isDragging.current) {
            isDragging.current = false;
            const delta = touchEndX.current - touchStartX.current;
            handleSelectNewExercise(delta);
        }
    };
    
    const currentPreviewUrl = orderedExercises[1]?.preview_url || exercise.preview_url;

    return (
        <div style={{ padding: '16px 16px 0', height: '100%', display: 'flex', flexDirection: 'column', boxSizing: 'border-box' }}>
            
            {/* --- PREVIEW SLIDER AREA --- */}
            <div 
                ref={previewRef}
                style={{ height: 220, borderRadius: 12, background: PALETTE.uiGrey, position: 'relative', flexShrink: 0, overflow: 'hidden', cursor: 'grab' }}
                // Mobile Touch Handlers
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                // Desktop Mouse Handlers (Drag)
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => isDragging.current = false} // Stop dragging if mouse leaves bounds
            >
                <button onClick={onBack} style={{ background: 'rgba(0,0,0,0.6)', color: PALETTE.light, border: 'none', width: 30, height: 30, borderRadius: 15, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', position: 'absolute', right: 10, top: 10, zIndex: 1 }}>{ICONS.close}</button>
                
                {/* แสดง Preview ของท่าปัจจุบัน */}
                <img
                    key={exercise.id}
                    src={currentPreviewUrl} 
                    alt={`Preview of ${exercise.name}`}
                    style={{ 
                        width: '100%', 
                        height: '100%', 
                        objectFit: 'cover', 
                        borderRadius: 12,
                        transition: 'opacity 0.2s ease', // Add fade transition for aesthetic change
                    }} 
                    onError={(e) => { 
                        e.target.onerror = null; 
                        e.target.src = `https://placehold.co/400x220/505050/dde1e6?text=GIF+Not+Found`; 
                    }}
                />
            </div>

            {/* --- DETAIL INFORMATION AREA --- */}
            <div style={{ marginTop: 20, flexGrow: 1, overflowY: 'auto' }} className="hide-scrollbar">
                {/* ... (Information Box เดิม) ... */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: 22 }}>{exercise.name}</div>
                        <div style={{ color: PALETTE.secondaryText, fontSize: 14, marginTop: 4 }}>{exercise.duration} · {exercise.level}</div>
                    </div>
                    <div>
                        <button style={{ background: PALETTE.primary, color: '#fff', padding: '12px 16px', borderRadius: 10, border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onPlay}>
                           {ICONS.play} เริ่มเซสชัน
                        </button>
                    </div>
                </div>
                <h3 style={{ marginTop: 20, fontSize: 16, fontWeight: 600, color: PALETTE.light }}>คำอธิบายท่า</h3>
                <p style={{ color: PALETTE.secondaryText, marginTop: 6, lineHeight: 1.6, fontSize: 14 }}>
                    {exercise.description || "ไม่มีคำอธิบายสำหรับท่านี้"}
                </p>
                <h3 style={{ marginTop: 15, fontSize: 16, fontWeight: 600, color: PALETTE.light }}>การใช้กล้ามเนื้อ</h3>
                <p style={{ color: PALETTE.secondaryText, marginTop: 6, lineHeight: 1.6, fontSize: 14 }}>
                    หลัก: {exercise.categoryId} | รอง: แขนและไหล่
                </p>
            </div>
        </div>
    );
}