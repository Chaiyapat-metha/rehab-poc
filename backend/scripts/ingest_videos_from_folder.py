# scripts/ingest_videos_from_folder.py

from pathlib import Path

# Import our custom modules
from scripts.video_processor import process_video_for_training
from app.utils import db

# --- CONFIGURATION ---
# The path to the folder containing your training video files.
VIDEO_SOURCE_DIR = Path("C:/Users/chaiyapat metha/Desktop/AI Project/rehab-poc/Video/train/ขา/Jump squats")

def main():
    """
    Finds all .mp4 and .mov files in the source directory, assigns a label,
    and processes them for training data ingestion.
    """
    if not VIDEO_SOURCE_DIR.is_dir():
        print(f"❌ Error: Source directory not found: {VIDEO_SOURCE_DIR}")
        return

    print(f"Starting batch ingestion from: {VIDEO_SOURCE_DIR}")
    print("-" * 40)

    # --- Step 1: Find all video files and their labels ---
    video_files_list = []
    # Loop through 'correct' and 'wrong' subfolders
    for sub_folder in ['correct', 'wrong']:
        folder_path = VIDEO_SOURCE_DIR / sub_folder
        if not folder_path.is_dir():
            print(f"Warning: Subfolder '{sub_folder}' not found. Skipping.")
            continue
            
        files_in_sub = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.mov"))
        for file in files_in_sub:
            video_files_list.append({
                'path': str(file),
                'label': sub_folder
            })
    
    if not video_files_list:
        print("No video files (.mp4, .mov) found in the training subfolders.")
        return
        
    print(f"Found {len(video_files_list)} video file(s) for ingestion.")

    # --- Step 2: Process all found videos ---
    for video_info in video_files_list:
        process_video_for_training(video_info['path'], video_info['label'])
        print("-" * 40)

    print("\n" + "=" * 40)
    print("✅ Batch ingestion complete!")
    print(f"   - Total videos processed: {len(video_files_list)}")
    print("=" * 40)


if __name__ == "__main__":
    main()