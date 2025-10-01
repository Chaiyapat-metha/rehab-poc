<!-- README.md -->

<h1 align="center">🤸 Rehab Pose Correction System</h1>

<p align="center">
  ระบบตรวจจับท่าออกกำลังกาย + ให้ฟีดแบคแบบเรียลไทม์ ด้วย <b>TCN+GRU Backbone</b> และ Multi-task Heads  
  (Position / Angle / Classification)
</p>

<p align="center">
  <img src="https://github.com/Chaiyapat-metha/rehab-poc/blob/main/Application_photo.png?raw=true" width="600">
</p>
---

<h2>📌 1 — เป้าหมายระบบ (Goal)</h2>

<ul>
  <li>รับ input: sequence ของ MediaPipe joints <code>(B, T, V=33, C=3)</code></li>
  <li>ประมวลผลด้วย Backbone (TCN×GRU) เพียงครั้งเดียวต่อ window → ส่งผลให้ heads เล็ก ๆ แยกทำงาน</li>
  <li><b>Heads</b>:
    <ol>
      <li><b>pos_head</b>: per-joint regression (99 dims → reshape (V,3))</li>
      <li><b>angle_head</b>: regress angles เฉพาะที่เลือก</li>
      <li><b>class_head</b>: binary/ multi-class classification (correct/wrong)</li>
    </ol>
  </li>
  <li>Inference API คืนค่า JSON ให้ client ใช้:
    <ul>
      <li>ข้อความบนจอ</li>
      <li>TTS (text/SSML)</li>
      <li>Highlight joints ที่ผิดบนภาพ</li>
    </ul>
  </li>
  <li>ThresholdController: ปรับ threshold ต่อผู้ใช้จาก user_history พร้อม UX ให้ยืนยัน</li>
  <li>Training pipeline: config-driven ต่อ exercise, mapping labels + augmentation</li>
</ul>

---

<h2>📂 2 — Data / Config Contract</h2>

<p>ระบบใช้ config ไฟล์:</p>

<ul>
  <li><code>model_config.yaml</code> → นิยาม backbone, heads, training setup</li>
  <li><code>exercises.yaml</code> → mapping ของแต่ละ exercise → joints, angles, metrics</li>
  <li><code>augmentation.yaml</code> → global/per-exercise augmentation</li>
</ul>

<pre>
dataset.py:
(data: BxTxVx3,
 target_pos: Bx99,
 target_angles: BxN_angles,
 target_class: Bx1,
 exercise_id: list[str])

augmentation.py:
- rotation, jitter, occlusion
- per-exercise augmentation → อัพเดท ground-truth
</pre>

---

<h2>🧠 3 — Model Architecture</h2>

<ul>
  <li>Backbone = <b>TCNBlock + GRU</b> → shared feature (B,256)</li>
  <li><b>pos_head</b>: MLP → (V,3), Loss=MSE</li>
  <li><b>angle_head</b>: MLP → angles + logvar, Loss=NLL</li>
  <li><b>class_head</b>: MLP → 1 logit, Loss=BCEWithLogits</li>
  <li>Multi-task loss: <code>L_total = Σ wᵢ * Lᵢ</code></li>
</ul>

<pre>
          joints (x,y,z)
                 ↓
          ┌─────────────┐
          │   Backbone  │  (TCN+GRU)
          └──────┬──────┘
                 │
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
 pos_head    angle_head    class_head
</pre>

---

<h2>🏋️ 4 — Training & Evaluation</h2>

<ul>
  <li>ใช้ dataset masks → ignore missing labels</li>
  <li>log per-head metrics</li>
  <li>save checkpoints แยก backbone + heads</li>
  <li>export ONNX (backbone.onnx, head.onnx)</li>
  <li>รองรับ reproducibility (seed, deterministic)</li>
</ul>

---

<h2>🚀 5 — Inference & API</h2>

<h3>Request:</h3>

<pre>
{
  "user_id": "u123",
  "exercise_id": "jump_squat",
  "window_frames": [[[x,y,z], ...], ...],
  "request": { "metrics": ["pos","angles","class"], "tts": true }
}
</pre>

<h3>Response:</h3>

<pre>
{
  "is_wrong": true,
  "class_prob": 0.12,
  "angles": { "LEFT_KNEE": 82.0 },
  "wrong_joints": [14],
  "tts_text": "ข้อศอกงอไม่พอค่ะ",
  "display_text": "ข้อศอกงอไม่พอ — 30° เกิน threshold",
  "timestamp": "2025-09-28T12:00:00Z"
}
</pre>

---

<h2>🎤 6 — Feedback Engine</h2>

<ul>
  <li>Template-based messages (Thai, randomized pool)</li>
  <li>TTS + OpenCV highlight</li>
  <li>Debounce/Throttle per rep</li>
  <li>LLM ใช้เฉพาะ session summary (ไม่ใช้ realtime)</li>
</ul>

---

<h2>⚙️ 7 — ThresholdController</h2>

<ul>
  <li>เก็บ per-user thresholds</li>
  <li>API:
    <ul>
      <li><code>GET /user/{id}/thresholds</code></li>
      <li><code>POST /user/{id}/thresholds/propose</code></li>
      <li><code>POST /user/{id}/thresholds/commit</code></li>
    </ul>
  </li>
  <li>rules: smoothing, min reps, auto adjust</li>
</ul>

---

<h2>🐳 8 — วิธีรันระบบ (Docker)</h2>

<pre>
cd infra
docker-compose up -d
</pre>

จะเปิด TimescaleDB ที่ <code>localhost:5433</code>

---

<h2>📜 9 — Generate Proto</h2>

<pre>
cd proto
bash generate_protos.sh
</pre>

สคริปต์จะ compile <code>rehab.proto</code> → เก็บใน <code>backend/app/proto_generated/</code>

---

<h2>🖥️ 10 — Run Backend API</h2>

<pre>
cd backend
uvicorn app.main:app --reload --port 8011
</pre>

API หลักอยู่ที่ <code>http://localhost:8011</code>  
Swagger UI: <code>http://localhost:8011/docs</code>

---

<h2>📊 11 — Frontend Integration</h2>

- Client รับ JSON feedback  
- แสดงข้อความ + TTS  
- Highlight joints บนภาพด้วย OpenCV  
- แสดง summary ที่ <code>frontend/src/components/ResultsScreen.jsx</code>  

---

<h2>👩‍💻 Contributors</h2>

<ul>
  <li><b>Lead Dev:</b> chaiyapat metha</li>
  <li><b>Stack:</b> FastAPI, PyTorch, ONNX, TimescaleDB, Docker</li>
</ul>

---
<p align="center">💪 ทำให้การฟื้นฟูร่างกายเป็นเรื่องง่าย — AI Pose Correction</p>
