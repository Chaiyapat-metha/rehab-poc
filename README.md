Rehab-PoC: ผู้ช่วยนักกายภาพบำบัด AI อัจฉริยะ 🤖<!-- Badges: ทำให้โปรเจกต์ดูน่าเชื่อถือ --><div align="center"><img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python Version"><img src="https://img.shields.io/badge/React-18-blue?logo=react" alt="React Version"><img src="https://img.shields.io/badge/FastAPI-black?logo=fastapi" alt="FastAPI"><img src="https://img.shields.io/badge/PostgreSQL-blue?logo=postgresql" alt="PostgreSQL"><img src="https://img.shields.io/badge/Docker-blue?logo=docker" alt="Docker"></div><br><!-- Hero Image: ส่วนที่สำคัญที่สุด! ให้อัดวิดีโอตอนใช้งานแอปแล้วแปลงเป็น GIF มาใส่ตรงนี้ --><!-- คุณสามารถอัปโหลดไฟล์ GIF ขึ้นไปบน GitHub issue ของคุณเองเพื่อเอา URL มาใช้ได้ --><p align="center"><img src="https://placehold.co/800x450/040404/dde1e6?text=ใส่+GIF+ตัวอย่างแอปของคุณที่นี่!" alt="App Demo GIF" width="80%"></p>โปรเจกต์นี้คือ Proof-of-Concept สำหรับแอปพลิเคชันผู้ช่วยนักกายภาพบำบัดที่ให้ Feedback การออกกำลังกายแบบ Real-time โดยใช้เทคโนโลยีการตรวจจับท่าทาง (Pose Estimation) และ AI Model ที่ฝึกสอนขึ้นมาโดยเฉพาะ เพื่อให้คำแนะนำที่แม่นยำและเป็นประโยชน์ต่อผู้ใช้✨ คุณสมบัติหลัก (Core Features)ตรวจจับและแก้ไขท่าทางแบบ Real-time: รับ Feedback ทันทีเมื่อทำท่าผิดพลาดวิเคราะห์ด้วย AI: โมเดล Supervised Multi-task สามารถระบุประเภทของข้อผิดพลาดได้อย่างเฉพาะเจาะจง (เช่น "ข้อศอกซ้ายงอน้อยเกินไป")โปรแกรมที่ปรับได้อัตโนมัติ (Adaptive Scheduling): ระบบจะวิเคราะห์ความก้าวหน้าและเสนอเป้าหมายใหม่ที่ท้าทายขึ้นให้โดยอัตโนมัติผู้ช่วย AI (RAG with LLM): สนทนากับ AI ที่เข้าถึงข้อมูลการออกกำลังกายของคุณ เพื่อถามคำถามและขอคำแนะนำครูฝึก Avatar 3D: ครูฝึก Avatar เสมือนจริงจะแสดงท่าออกกำลังกายที่ถูกต้องให้ดูเป็นตัวอย่าง🛠️ เทคโนโลยีที่ใช้ (Tech Stack)ส่วนเทคโนโลยีส่วนหน้า (Frontend)React (Vite), Three.js (สำหรับ Avatar)ส่วนหลัง (Backend)Python, FastAPI (สำหรับ APIs & WebSocket)AI / MLPyTorch, MediaPipe, ONNX Runtime, LangChainฐานข้อมูลPostgreSQL + TimescaleDB (ทำงานผ่าน Docker)🚀 วิธีการติดตั้งและรันโปรเจกต์ (Local Setup)ทำตามขั้นตอนเหล่านี้เพื่อรันโปรเจกต์ทั้งหมดบนเครื่องของคุณสิ่งที่ต้องมี (Prerequisites)GitDocker DesktopPython 3.10+Node.js 18+ (มาพร้อมกับ npm)1. คัดลอกโปรเจกต์ (Clone)git clone <YOUR_REPO_URL>
cd rehab-poc
2. ตั้งค่า Backend# 1. เข้าไปที่โฟลเดอร์ backend
cd backend

# 2. สร้างและเปิดใช้งาน Virtual Environment
python -m venv venv
# บน Windows:
venv\Scripts\activate
# บน Mac/Linux:
# source venv/bin/activate

# 3. ติดตั้ง Dependencies ของ Python
pip install -r requirements.txt

# 4. สร้างไฟล์ .env สำหรับเก็บ API Key
# (สร้างไฟล์ชื่อ .env แล้วใส่ OPENROUTER_API_KEY="sk-or-v1-...")
echo 'OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY_HERE"' > .env

# 5. รันฐานข้อมูลด้วย Docker
cd ../infra
docker-compose up -d
cd ../backend

# 6. รัน Backend Server
# (ต้องเปิด Terminal นี้ค้างไว้)
uvicorn app.main:app --reload --host 0.0.0.0
3. ตั้งค่า Frontend# 1. เปิด Terminal ใหม่ขึ้นมาอีกอัน
# 2. เข้าไปที่โฟลเดอร์ frontend
cd frontend

# 3. ติดตั้ง Dependencies ของ Node.js
npm install

# 4. รัน Frontend Development Server
# (ต้องเปิด Terminal นี้ค้างไว้)
npm run dev
