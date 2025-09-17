import React, { useState, useEffect, useRef } from 'react';
import HomeScreen from './components/HomeScreen.jsx';
import DetailScreen from './components/DetailScreen.jsx';
import SessionScreen from './components/SessionScreen.jsx';
import ResultsScreen from './components/ResultsScreen.jsx';
import BottomNavBar from './components/BottomNavBar.jsx';
import { MOCK_DATA, PALETTE, API_BASE_URL } from './constants.jsx';

export default function App() {
    const [screen, setScreen] = useState('home');
    const [selectedExercise, setSelectedExercise] = useState(null);
    const homeScrollPositions = useRef({});

    // --- State for real data, fetched once at the top level ---
    const [exercisesByCat, setExercisesByCat] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // Fetch all exercise data when the app first loads
    useEffect(() => {
        async function fetchAllExercises() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/exercises`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();

                if (data.status === 'success') {
                    setExercisesByCat(data.exercisesByCat);
                } else {
                    throw new Error(data.message || 'Failed to fetch exercises.');
                }
            } catch (err) {
                setError(err.message);
                console.error("Fetch error:", err);
            } finally {
                setIsLoading(false);
            }
        }
        fetchAllExercises();
    }, []);


    const navigateToDetail = (exercise, categoryId) => {
        const scrollContainer = document.getElementById(`category-scroll-${categoryId}`);
        if (scrollContainer) {
            homeScrollPositions.current[categoryId] = scrollContainer.scrollLeft;
        }
        // Use the title from the mock data for the category title
        const categoryTitle = MOCK_DATA.categories.find(c => c.id === categoryId)?.title || categoryId;
        setSelectedExercise({ ...exercise, categoryTitle });
        setScreen('detail');
    };

    const navigateToHome = () => {
        setSelectedExercise(null);
        setScreen('home');
    };

    const startSession = () => {
        if (!selectedExercise && exercisesByCat) {
            const firstCategoryWithExercises = MOCK_DATA.categories.find(cat => exercisesByCat[cat.id] && exercisesByCat[cat.id].length > 0);
            if (firstCategoryWithExercises) {
                const firstExercise = exercisesByCat[firstCategoryWithExercises.id][0];
                setSelectedExercise(firstExercise);
            } else {
                alert("No exercises available to start a session.");
                return;
            }
        }
        setScreen('session');
    };
    
    const endSession = () => {
        setScreen('results');
    };

    useEffect(() => {
        if (screen === 'home') {
            setTimeout(() => {
                for (const categoryId in homeScrollPositions.current) {
                    const scrollContainer = document.getElementById(`category-scroll-${categoryId}`);
                    if (scrollContainer) {
                        scrollContainer.scrollLeft = homeScrollPositions.current[categoryId];
                    }
                }
            }, 0);
        }
    }, [screen]);

    const renderScreen = () => {
        switch (screen) {
            case 'detail': 
                return <DetailScreen exercise={selectedExercise} onBack={navigateToHome} onPlay={startSession} />;
            case 'session': 
                return <SessionScreen exercise={selectedExercise} onEnd={endSession} />;
            case 'results': 
                return <ResultsScreen onDone={navigateToHome} />;
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
            <div style={{ width: 375, height: 812, margin: '0 auto', borderRadius: 40, overflow: 'hidden', boxShadow: '0 12px 30px rgba(0,0,0,0.4)', position: 'relative', background: PALETTE.bg }}>
                <div style={{ width: '100%', height: '100%', color: PALETTE.light, display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '12px 20px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, zIndex: 10, color: PALETTE.light }}>
                        <div style={{ fontSize: 14, fontWeight: 600 }}>9:41</div>
                    </div>
                    <div style={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
                        {renderScreen()}
                    </div>
                    {screen === 'home' && <BottomNavBar onPlay={startSession} />}
                </div>
            </div>
        </div>
    );
}

