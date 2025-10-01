AI-Powered Adaptive Physical Therapy System (Rehab-PoC)
1. Abstract
This project introduces a robust, low-latency system designed to assist elderly patients with unsupervised physical therapy (PT) exercises. Utilizing a TCN+GRU Multi-task Network, the system performs real-time 3D pose correction, identifying joint deviations, quantifying errors in degrees/mm, and providing immediate TTS feedback. The core innovation is the integration of an Adaptive Threshold Controller and an LLM-based RAG Chain for personalized difficulty adjustment and long-term progress monitoring. The modular ONNX deployment ensures high performance and scalability.

2. Introduction: Problem Statement
Current geriatric physiotherapy models lack effective remote tracking and personalized guidance, leading to poor patient compliance and risk of performing exercises incorrectly. This system directly addresses these gaps:

Lack of Real-time Guidance: Elderly patients require immediate correction, which standard video observation cannot provide.

Accessibility: The system provides synchronized audio and visual feedback (TTS and joint highlighting) to assist users with reduced visual acuity.

Personalization: The platform allows for monitoring individual progress and adapting difficulty thresholds over time, mimicking a dedicated personal therapist.

3. Design: Model Architecture & Data Pipeline
The system utilizes a specialized architecture tailored for sequential motion analysis and deployment efficiency.

A. Data Collection and Storage
Feature Extraction: MediaPipe BlazePose (Heavy Model) extracts 33 Keypoints in World Coordinates (X, Y, Z) from training videos.

Database: Data is persisted in TimescaleDB (PostgreSQL) for robust storage and efficient time-series indexing of joint data (stored as Protobuf BYTEA).

Augmentation: A Config-Driven Synthetic Augmentation Pipeline creates diverse Wrong Examples (e.g., insufficient depth, shoulder offset) and automatically updates multiclass labels.

B. Model Architecture (TCN+GRU)
The model employs a single Shared Backbone for temporal feature extraction, feeding modular diagnostic heads:

Component

Function

Loss Function

Output

Backbone

TCN + GRU (T=16 Window)

N/A

Shared Feature Vector (B×256)

Class Head

Correctness Prediction (0/1 or 0/1/2)

BCEWithLogitsLoss or CrossEntropyLoss

Logit or Vector of Classes

Angle Head

Quantified Error Estimation

Gaussian NLL Loss (Predicts Mean and LogVar)

Vector of Predicted Angles

Positional Head

3D Joint Coordinate Prediction

L1Loss

99D Predicted Position Vector

4. Workflow: End-to-End System Flow
A. Training and Deployment Workflow (Offline)
Video Assets 
Ingestion/Augment
​
 TimescaleDB 
Stratified Load
​
 TCN+GRU Training 
Save Best
​
 ONNX Export 
FastAPI
​
 Production Weights
B. Real-time Inference and Feedback Loop (Online)
Data Streaming (Frontend): 3D Keypoints are windowed (T=16) and streamed via WebSocket to Backend.

Model Inference (Backend): ONNX Runtime runs input through Backbone and Heads concurrently.

Decision & Feedback: Predicted Angles/Class → Threshold Controller checks Pass/Fail → Feedback Engine generates TTS Text and Wrong Joint Indices.

Client Action: WebSocket sends Feedback → Frontend highlights joint in red on Canvas Overlay and plays audio (TTS) after 90 frames (3s) of sustained error.

5. Experiments and Results
Our primary goal was achieving high classification accuracy while maintaining quantifiable error metrics necessary for personalized thresholds.

Metric

Significance

Result

Classification AUC

Area Under Curve (Correctness)

∼0.85−0.94

Angle MAE

Mean Absolute Error in Degrees

∼15−20

Positional MPJPE (Loss)

Accuracy of 3D Joint Coordinates

∼4

The Multi-task model achieved high discriminative power (AUC) while simultaneously reconstructing the pose with low MPJPE loss.

Fig. 3(a): Visual Proof of Quantified Error (Angle Prediction)

Fig. 3(b): Demonstration of Adaptive Threshold Adjustment based on user success rate.

6. Inference and Adaptive Control
A. Modular ONNX Inference
We utilize ONNX Runtime to decouple the AI components:

Efficiency: The shared Backbone.onnx is loaded only once.

Scalability: New exercises require only minimal Head.onnx files to be deployed, significantly reducing resource consumption and update time.

B. Adaptive Threshold Control
The system implements the Threshold Controller (Req. 8) which proposes a change in difficulty (tighten or relax the angle error tolerance) after analyzing Success Rate over Window K reps. The user must confirm the change before it is committed to the user_thresholds DB.

7. Conclusion
The Rehab-PoC project successfully delivered a robust, full-stack AI solution for physiotherapy. By integrating quantified error diagnosis and personalized adaptation via a multi-task model, the system transcends simple monitoring to actively guide and motivate user progress. The final architecture is highly efficient and ready for large-scale deployment.

8. Reference
K. Gong, J. Zhang, and J. Feng, “PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation,” in Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2021.

S. Shin, J. Kim, E. Halilaj, and M. J. Black, “WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion,” GitHub repository, 2024.