from pathlib import Path

# Import our custom modules
from scripts.video_processor import process_video
from app.utils import db

# --- CONFIGURATION ---
# The path to the folder containing your ground truth video files.
VIDEO_SOURCE_DIR = Path("C:/Users/chaiyapat metha/Desktop/AI Project/rehab-poc/Video/Raw")

def main():
    """
    Finds all .mp4 and .mov files in the source directory, checks if they
    have already been processed, and processes them if they are new.
    """
    if not VIDEO_SOURCE_DIR.is_dir():
        print(f"❌ Error: Source directory not found: {VIDEO_SOURCE_DIR}")
        return

    print(f"Starting batch ingestion from: {VIDEO_SOURCE_DIR}")
    print("-" * 40)

    # --- Step 1: Get a list of exercises already in the database ---
    print("🔎 Checking database for existing exercises...")
    existing_exercises = db.get_existing_exercise_names()
    if existing_exercises:
        print(f"Found {len(existing_exercises)} existing exercises. These will be skipped if found.")
    else:
        print("Database is empty. All found videos will be processed.")
    
    print("-" * 40)

    # --- Step 2: Find all video files in the source directory ---
    video_files = list(VIDEO_SOURCE_DIR.glob("*.mp4")) + list(VIDEO_SOURCE_DIR.glob("*.mov"))
    
    if not video_files:
        print("No video files (.mp4, .mov) found in the directory.")
        return
        
    print(f"Found {len(video_files)} video file(s) in the source folder.")

    # --- Step 3: Loop through files, check, and process ---
    new_videos_processed = 0
    for video_path in video_files:
        # The exercise name is the filename without the extension.
        exercise_name = video_path.stem

        # Check if this exercise name is in the set we fetched from the DB
        if exercise_name in existing_exercises:
            print(f"⏭️  Skipping '{exercise_name}' as it already exists in the database.")
            continue
        
        # If not in the set, process it
        process_video(str(video_path))
        new_videos_processed += 1
        print("-" * 40)

    print("\n" + "=" * 40)
    print("✅ Batch ingestion complete!")
    print(f"   - Total videos in folder: {len(video_files)}")
    print(f"   - Skipped existing videos: {len(video_files) - new_videos_processed}")
    print(f"   - Newly processed videos: {new_videos_processed}")
    print("=" * 40)


if __name__ == "__main__":
    # Ensure we run this as a module to handle imports correctly
    main()
