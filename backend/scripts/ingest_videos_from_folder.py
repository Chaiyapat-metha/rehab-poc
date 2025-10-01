# File: .\backend\scripts\ingest_videos_from_folder.py

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1])) 

from scripts.video_processor import VideoProcessor

TRAIN_ROOT = Path(r"C:\Users\chaiyapat metha\Desktop\AI Project\rehab-poc\Video\train")

def scan_and_ingest(root_path: Path):
    """
    Scans the training video directory structure and processes each video.
    Structure: {Category}/{Exercise_Name}/{correct/wrong}/{video_files}
    """
    processor = VideoProcessor(model_complexity=2) 
    
    # 1. วนลูปผ่าน Category (ขา, แขน, คอ, ลำตัว, ทั้งตัว)
    for category_folder in root_path.iterdir():
        if not category_folder.is_dir():
            continue
            
        print(f"--- Scanning Category: {category_folder.name} ---")

        # 2. วนลูปผ่าน Exercise (เช่น Jump squats)
        for exercise_folder in category_folder.iterdir():
            if not exercise_folder.is_dir():
                continue

            exercise_name = exercise_folder.name.replace(' ', '_') # ใช้ชื่อท่าเป็น exercise_id 
            print(f"  Found Exercise: {exercise_name}")
         
            # 3a. โฟลเดอร์ 'correct'
            correct_path = exercise_folder / 'correct'
            if correct_path.is_dir():
                print(f"    Processing {len(list(correct_path.glob('*.mp4')))} correct videos...")
                for video_file in correct_path.glob('*.mp4'):
                    print(f"      Ingesting correct: {video_file.name}")
                    # ส่งไฟล์ไปประมวลผล (Label class=0 สำหรับ Correct)
                    processor.process_video(video_file, exercise_name, label_class=0)

            # 3b. โฟลเดอร์ 'wrong'
            wrong_path = exercise_folder / 'wrong'
            if wrong_path.is_dir():
                print(f"    Processing {len(list(wrong_path.glob('*.mp4')))} wrong videos...")
                for video_file in wrong_path.glob('*.mp4'):
                    print(f"      Ingesting wrong: {video_file.name}")
                    # ส่งไฟล์ไปประมวลผล (Label class=1 สำหรับ Wrong)
                    processor.process_video(video_file, exercise_name, label_class=1)
                    
    processor.flush_batches() 
    print("\nIngestion process complete. Database batches flushed.")

if __name__ == '__main__':
    scan_and_ingest(TRAIN_ROOT)