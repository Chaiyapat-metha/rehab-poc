// --- Global DOM Elements ---
const videoElement = document.getElementById('webcam');
const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const avatarContainer = document.getElementById('avatar-container');
const uploadForm = document.getElementById('upload-form');
const uploadStatusElement = document.getElementById('upload-status');
const feedbackPanel = document.getElementById('feedback-panel');
const proposalContainer = document.getElementById('proposal-container');
const proposalList = document.getElementById('proposal-list');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatHistory = document.getElementById('chat-history');
const API_BASE_URL = 'http://localhost:8000'; 

let poseDetector;
let ws;
const SESSION_ID = "session_" + Math.random().toString(36).substr(2, 9);
const USER_ID = "ground_truth_trainer"; // Hardcoded for this PoC

// --- Three.js Setup (Placeholder) ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, avatarContainer.clientWidth / avatarContainer.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(avatarContainer.clientWidth, avatarContainer.clientHeight);
avatarContainer.appendChild(renderer.domElement);
const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshNormalMaterial();
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);
camera.position.z = 5;

// --- Main Application Logic ---

async function setupCamera() {
    // ... (This function remains the same) ...
    const stream = await navigator.mediaDevices.getUserMedia({ 'video': { width: 640, height: 480 } });
    videoElement.srcObject = stream;
    return new Promise((resolve) => {
        videoElement.onloadedmetadata = () => resolve(videoElement);
    });
}

async function createPoseDetector() {
    // ... (This function remains the same) ...
    const model = poseDetection.SupportedModels.BlazePose;
    const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose',
        modelType: 'full'
    };
    return await poseDetection.createDetector(model, detectorConfig);
}

function connectWebSocket() {
    // ... (This function has been slightly updated for better feedback) ...
    const backendUrl = `ws://${window.location.hostname}:8000/ws/live/${SESSION_ID}`;
    ws = new WebSocket(backendUrl);

    ws.onopen = () => {
        statusElement.textContent = "สถานะ: เชื่อมต่อแล้ว";
        console.log("WebSocket connection established.");
        feedbackPanel.style.borderColor = '#4CAF50';
    };

    ws.onmessage = (event) => {
        const feedback = JSON.parse(event.data);
        console.log("Feedback received:", feedback);
        
        messageElement.textContent = feedback.message;
        
        if (feedback.status === 'error') {
            feedbackPanel.style.background = 'rgba(217, 83, 79, 0.8)'; // Red
            feedbackPanel.style.borderColor = '#d9534f';
        } else if (feedback.status === 'correct') {
            feedbackPanel.style.background = 'rgba(76, 175, 80, 0.8)'; // Green
            feedbackPanel.style.borderColor = '#4CAF50';
        } else {
            feedbackPanel.style.background = 'rgba(0,0,0,0.7)';
            feedbackPanel.style.borderColor = 'transparent';
        }

        speak(feedback.message);
    };

    ws.onclose = () => {
        statusElement.textContent = "สถานะ: ตัดการเชื่อมต่อ";
        feedbackPanel.style.background = 'rgba(255, 159, 64, 0.8)'; // Orange for disconnected/retrying
        console.log("WebSocket connection closed. Retrying in 3 seconds...");
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        statusElement.textContent = "สถานะ: เกิดข้อผิดพลาด";
    };
}

function speak(text) { /* ... (This function remains the same) ... */ }

async function renderLoop() {
    // ... (This function remains the same with the corrected send format) ...
    if (!poseDetector) return;
    const poses = await poseDetector.estimatePoses(videoElement);

    if (poses && poses.length > 0 && ws && ws.readyState === WebSocket.OPEN) {
        const keypoints = poses[0].keypoints3D; 
        if(keypoints) {
            ws.send(JSON.stringify({ "keypoints": keypoints }));
        }
    }
    
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    renderer.render(scene, camera);

    requestAnimationFrame(renderLoop);
}

async function fetchProposals(userId) {
    console.log(`Fetching proposals for user: ${userId}`);
    try {
        const response = await fetch(`${API_BASE_URL}/proposals/${userId}`);
        const data = await response.json();

        if (data.status === 'success' && data.proposals.length > 0) {
            proposalList.innerHTML = ''; // Clear old proposals
            data.proposals.forEach(p => {
                const proposalCard = document.createElement('div');
                proposalCard.className = 'proposal-card';
                // Using textContent for security, but innerHTML is fine for this controlled case
                proposalCard.innerHTML = `
                    <p>
                        <strong>ท่า:</strong> ${p.exercise_name}<br>
                        นักกายภาพ AI แนะนำให้เพิ่มเป้าหมาย <strong>${p.metric_key.replace(/_/g, ' ')}</strong>
                        จาก ${p.current_value.toFixed(1)}° เป็น <strong>${p.proposed_value.toFixed(1)}°</strong>
                    </p>
                    <button onclick="respondToProposal(${p.proposal_id}, 'accepted')">ยอมรับ</button>
                    <button onclick="respondToProposal(${p.proposal_id}, 'rejected')">ปฏิเสธ</button>
                `;
                proposalList.appendChild(proposalCard);
            });
            proposalContainer.style.display = 'block'; // Show the container
        } else {
            console.log("No pending proposals found.");
            proposalContainer.style.display = 'none'; // Hide if no proposals
        }
    } catch (error) {
        console.error("Failed to fetch proposals:", error);
    }
}

// Make this function globally accessible for the onclick attribute
window.respondToProposal = async function(proposalId, response) {
    console.log(`Responding to proposal ${proposalId} with: ${response}`);
    try {
        // --- CORRECTED: Use the full API URL ---
        await fetch(`${API_BASE_URL}/proposals/${proposalId}/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ response: response })
        });
        fetchProposals(USER_ID); 
    } catch (error) {
        console.error("Failed to respond to proposal:", error);
        alert('เกิดข้อผิดพลาดในการตอบรับข้อเสนอ');
    }
}

// make this function for event listener in chatbot
if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = chatInput.value;
        if (!message) return;

        addMessageToHistory('You', message);
        chatInput.value = '';
        addMessageToHistory('AI', 'กำลังคิดสักครู่...');

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: USER_ID, message: message })
            });
            const data = await response.json();
            
            // Remove the "thinking" message and add the real one
            chatHistory.removeChild(chatHistory.lastChild);
            addMessageToHistory('AI', data.reply);

        } catch (error) {
            console.error("Chat error:", error);
            chatHistory.removeChild(chatHistory.lastChild);
            addMessageToHistory('AI', 'ขออภัยค่ะ, การเชื่อมต่อขัดข้อง');
        }
    });
}

function addMessageToHistory(sender, message) {
    const p = document.createElement('p');
    p.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatHistory.appendChild(p);
    // Auto-scroll to the bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// --- Main Initialization Function ---
async function main() {
    try {
        await setupCamera();
        videoElement.play();
        
        poseDetector = await createPoseDetector();
        
        connectWebSocket();
        fetchProposals(USER_ID); // Fetch proposals on page load
        renderLoop();
    } catch (error) {
        console.error("Initialization failed:", error);
        alert(`ไม่สามารถเริ่มต้นระบบได้: ${error.message}`);
    }
}

// --- Run the application ---
main();

