rehab-poc/
├─ backend/
│  ├─ app/                      # FastAPI app (inference, data ingest, admin)
│  │  ├─ main.py
│  │  ├─ api/
│  │  ├─ models/                 # model loader interfaces (pluggable)
│  │  │  ├─ base_model.py        # abstract interface
│  │  │  ├─ anomaly_autoencoder.py
│  │  │  ├─ supervised_tcn_gru.py
│  │  ├─ utils/
│  │  ├─ config.py               # loads YAML config (models.yaml)
│  ├─ training/                  # training scripts (PyTorch)
│  │  ├─ train_autoencoder.py
│  │  ├─ train_multitask.py
│  │  ├─ data_loader.py
│  │  ├─ models/                 # model architecture definitions (config-driven)
│  │  ├─ losses.py               # pluggable loss functions
│  ├─ scripts/
│  │  ├─ export_onnx.py
│  │  ├─ onnx_to_tflite.py
│  ├─ requirements.txt
├─ frontend/
│  ├─ web/                       # web demo (Three.js + TFJS BlazePose or camera input)
│  │  ├─ index.html
│  │  ├─ main.js
│  │  ├─ three_utils/
│  │  ├─ models/                  # placeholder trainer GLB (Mixamo)
│  ├─ mobile_test/                # optional react-native / expo quick prototype
├─ proto/                         # Protobuf schemas
│  ├─ rehab.proto
│  ├─ generate_protos.sh
├─ infra/
│  ├─ docker-compose.yml          # TimescaleDB / Postgres (for Windows use Docker)
│  ├─ timescaledb-init.sql
├─ docs/
│  ├─ design.md
│  ├─ api-spec.md
├─ examples/
│  ├─ sample_frames.pb
│  ├─ sample_features.npz
├─ .github/
│  ├─ workflows/                  # CI: lint + unit tests + basic training smoke test
├─ README.md                      # นี่คือไฟล์ (Thai)

Rehab-PoC: ผู้ช่วยนักกายภาพบำบัด AI
โปรเจกต์นี้คือระบบต้นแบบ (Proof-of-Concept) สำหรับผู้ช่วยนักกายภาพบำบัด AI ที่สามารถให้ Feedback การออกกำลังกายของผู้ใช้ได้แบบ Real-time โดยใช้ MediaPipe ในการตรวจจับท่าทาง, โมเดลที่ฝึกสอนขึ้นมาเองเพื่อวิเคราะห์ข้อผิดพลาด, และ React สำหรับสร้างหน้าจอผู้ใช้งาน

✨ คุณสมบัติหลัก (Core Features)
วิเคราะห์ท่าทางแบบ Real-time: รับ Feedback ได้ทันทีขณะกำลังออกกำลังกาย

ขับเคลื่อนด้วย AI: โมเดล Supervised ที่สร้างขึ้นเอง สามารถระบุข้อผิดพลาดที่เฉพาะเจาะจงได้ (เช่น "ข้อศอกซ้ายงอน้อยเกินไป")

ตารางฝึกซ้อมที่ปรับได้: ระบบสามารถเสนอเป้าหมายใหม่ที่ท้าทายขึ้นตามประสิทธิภาพของผู้ใช้ (Phase 4)

ผู้ช่วย AI อัจฉริยะ (RAG): สนทนากับ AI ที่เข้าถึงข้อมูลการออกกำลังกายของคุณได้ (Phase 5)

ครูฝึก Avatar: Avatar 3 มิติ แสดงท่าออกกำลังกายที่ถูกต้อง

🛠️ เทคโนโลยีที่ใช้ (Tech Stack)
Frontend: React (Vite), Three.js (สำหรับ Avatar)

Backend: Python, FastAPI (สำหรับ APIs & WebSocket)

AI / ML: PyTorch, MediaPipe, ONNX Runtime, LangChain

Database: PostgreSQL + TimescaleDB (รันผ่าน Docker)

🚀 เริ่มต้นใช้งาน (Local Setup)
ทำตามขั้นตอนเหล่านี้เพื่อรันโปรเจกต์บนเครื่องของคุณ

สิ่งที่ต้องมี
Git

Docker Desktop

Python 3.10+

Node.js 18+ (มาพร้อมกับ npm)

1. Clone Repository
git clone <your-repo-url>
cd rehab-poc

2. ตั้งค่าฝั่ง Backend
# ไปที่โฟลเดอร์ backend
cd backend

# สร้างและ Activate virtual environment
python -m venv venv
# บน Windows:
venv\Scripts\activate
# บน Mac/Linux:
# source venv/bin/activate

# ติดตั้ง Dependencies ของ Python
pip install -r requirements.txt

# สร้างไฟล์ .env สำหรับเก็บ API Key
# สร้างไฟล์ใหม่ชื่อ .env แล้วใส่ OpenRouter Key ของคุณ
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > .env

# ไปที่โฟลเดอร์ infra แล้วรันฐานข้อมูลผ่าน Docker
cd ../infra
docker-compose up -d
cd ../backend

# รัน Backend server
uvicorn app.main:app --reload --host 0.0.0.0

3. ตั้งค่าฝั่ง Frontend
# เปิด Terminal ใหม่ แล้วไปที่โฟลเดอร์ frontend
cd frontend

# ติดตั้ง Dependencies ของ Node.js
npm install

# (Optional) ติดตั้ง Three.js สำหรับ Avatar
npm install three

# รัน Development server
npm run dev

ตอนนี้คุณควรจะสามารถเข้าถึงแอปได้ที่ http://localhost:5173

4. ขั้นตอนการเตรียมข้อมูลและเทรนโมเดล
รันสคริปต์เหล่านี้จากโฟลเดอร์ backend/ (ที่ activate venv แล้ว):

# 1. นำเข้าวิดีโอ Ground Truth (นำวิดีโอไปไว้ใน Video/Raw ก่อน)
python -m scripts.ingest_videos_from_folder

# 2. สร้างป้ายกำกับ (Labels) สำหรับ Supervised learning
python -m scripts.auto_labeler

# 3. เทรนโมเดล Supervised
python -m training.train_multitask --config training_configs/supervised_config.yaml

# 4. Export โมเดลเพื่อให้ Backend ใช้งาน
python -m training.export_onnx --config training_configs/supervised_config.yaml

เอกสาร README นี้จะเป็นคู่มือที่สมบูรณ์ให้เพื่อนของคุณสามารถนำโปรเจกต์ไปรันบนเครื่องของตัวเองได้