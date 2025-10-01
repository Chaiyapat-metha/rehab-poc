import React, { useState, useEffect, useRef, useCallback } from 'react';
import HomeScreen from './components/HomeScreen.jsx';
import DetailScreen from './components/DetailScreen.jsx';
import SessionScreen from './components/SessionScreen.jsx';
import ResultsScreen from './components/ResultsScreen.jsx';
import BottomNavBar from './components/BottomNavBar.jsx';
import { EXERCISES_MAP, MOCK_DATA, PALETTE, API_BASE_URL, USER_ID } from './constants.jsx';

const CATEGORY_MAP_TO_THAI_FOLDER = {
    'neck': 'คอ', 
    'arm': 'แขน', 
    'torso': 'ลำตัว', 
    'leg': 'ขา', 
    'full': 'ทั้งตัว'
};

// --- Utility: Map asset URLs ---
const getAssetUrls = (exerciseId, categoryId) => {
    // 💡 NOTE: Path ใช้ชื่อโฟลเดอร์ภาษาไทยที่ถูกต้อง (คอ, แขน, ฯลฯ)
    const thaiFolderName = CATEGORY_MAP_TO_THAI_FOLDER[categoryId]; 
    const basePath = `/assets`;
    
    return {
        clip_url: `${basePath}/clips/${thaiFolderName}/${exerciseId}.mp4`, 
        preview_url: `${basePath}/previews/${thaiFolderName}/${exerciseId}.gif`, 
        thumbnail_url: `${basePath}/thumbnails/${thaiFolderName}/${exerciseId}.jpg`,
    };
};

const processExercises = () => {
    const processed = {};
    for (const [id, data] of Object.entries(EXERCISES_MAP)) {
        const catId = data.category;
        if (!processed[catId]) {
            processed[catId] = [];
        }
        processed[catId].push({
            id: id, 
            name: data.name_th, 
            duration: data.duration,
            level: data.level,
            description: data.description,
            categoryId: catId, 
            ...getAssetUrls(id, catId)
        });
    }
    return processed;
};

export default function App() {
    const [screen, setScreen] = useState('home');
    const [selectedExercise, setSelectedExercise] = useState(null);
    const [sessionResult, setSessionResult] = useState(null); 
    const homeScrollPositions = useRef({});

    const [exercisesByCat, setExercisesByCat] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const appContainerRef = useRef(null);

    // Initial Data Load (Uses enriched local data)
    useEffect(() => {
        const enrichedExercises = processExercises();
        setExercisesByCat(enrichedExercises);
        setIsLoading(false);
    }, []);


    const navigateToDetail = (exercise, categoryId) => {
        const scrollContainer = document.getElementById(`category-scroll-${categoryId}`);
        if (scrollContainer) {
            homeScrollPositions.current[categoryId] = scrollContainer.scrollLeft;
        }
        
        setSelectedExercise({ ...exercise, categoryId });
        setScreen('detail');
    };

    const navigateToHome = () => {
        setSelectedExercise(null);
        setScreen('home');
    };

    // Handler for DetailScreen when user swipes/selects a new exercise
    const handleExerciseSelect = (newExercise) => {
        setSelectedExercise({ ...newExercise, categoryId: newExercise.categoryId });
        // The screen remains 'detail'
    };


    const startSession = () => {
        if (!selectedExercise && exercisesByCat) {
            const firstCategoryWithExercises = MOCK_DATA.categories.find(cat => exercisesByCat[cat.id] && exercisesByCat[cat.id].length > 0);
            if (firstCategoryWithExercises) {
                const firstExercise = exercisesByCat[firstCategoryWithExercises.id][0];
                setSelectedExercise(firstExercise);
            } else {
                console.error("No exercises available to start a session.");
                return;
            }
        }
        setScreen('session');
    };
    
    // Handler when session ends (User presses End button)
    const endSession = useCallback((metrics = {}) => {
        setSessionResult(metrics);
        setScreen('results');
    }, []);

    // Handler to commit proposed thresholds to the Backend (Req. 8)
    const commitThresholds = async (proposal) => {
        if (!selectedExercise) return;

        try {
            // 💡 FIX: Corrected API endpoint structure for commit
            const response = await fetch(`${API_BASE_URL}/user/${USER_ID}/thresholds/commit/${selectedExercise.id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ updates: proposal }) 
            });
            if (response.ok) {
                console.log("Thresholds committed successfully.");
            } else {
                console.error("Failed to commit thresholds. Status:", response.status);
            }
        } catch (error) {
            console.error("API error during threshold commit:", error);
        }
    };


    useEffect(() => {
        if (screen === 'home') {
            const restoreScroll = () => {
                for (const categoryId in homeScrollPositions.current) {
                    const scrollContainer = document.getElementById(`category-scroll-${categoryId}`);
                    if (scrollContainer) {
                        scrollContainer.scrollLeft = homeScrollPositions.current[categoryId];
                    }
                }
            };
            requestAnimationFrame(restoreScroll);
        }
    }, [screen]);

    const renderScreen = () => {
        switch (screen) {
            case 'detail': 
                return <DetailScreen 
                    exercise={selectedExercise} 
                    exercisesInCat={exercisesByCat[selectedExercise.categoryId]} 
                    onBack={navigateToHome} 
                    onPlay={startSession}
                    onSelect={handleExerciseSelect}
                />;
            case 'session': 
                return <SessionScreen 
                    exercise={selectedExercise} 
                    onEnd={endSession} 
                />;
            case 'results': 
                return <ResultsScreen 
                    onDone={navigateToHome} 
                    currentExerciseId={selectedExercise?.id}
                    sessionMetrics={sessionResult} 
                    onThresholdCommit={commitThresholds} 
                />;
            default: 
                return <HomeScreen 
                    exercisesByCat={exercisesByCat} 
                    isLoading={isLoading}
                    error={error}
                    onSelectExercise={navigateToDetail} 
                />;
        }
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
            <div ref={appContainerRef} style={{ width: 375, height: 812, margin: '0 auto', borderRadius: 40, overflow: 'hidden', boxShadow: '0 12px 30px rgba(0,0,0,0.4)', position: 'relative', background: PALETTE.bg, display: 'flex', flexDirection: 'column' }}>
                
                {/* Header/Status Bar */}
                <div style={{ padding: '12px 20px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, zIndex: 10, color: PALETTE.light }}>
                    <div style={{ fontSize: 14, fontWeight: 600 }}>9:41</div>
                </div>
                
                {/* Content Area */}
                <div style={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
                    {renderScreen()}
                </div>
                
                {/* Bottom Navigation */}
                {screen === 'home' && <BottomNavBar onPlay={startSession} />}
            </div>
        </div>
    );
}
