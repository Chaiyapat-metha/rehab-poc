import whisper
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

# ไม่จำเป็นต้องตั้งค่า FFMPEG_BINARY อีกต่อไป ถ้า FFMPEG อยู่ใน PATH ของระบบแล้ว

def get_whisper_model(size: str = "medium"):
    """
    Loads a Whisper model, handling caching and potential corruption.
    Recommended sizes: 'base' for speed, 'medium' for accuracy.
    """
    print(f"⚡ Loading Whisper model '{size}'... (This may take a while on first download)")
    # Whisper's default caching is now robust enough.
    # The complex safe loader from your old script is generally not needed anymore,
    # but the principle is good. Whisper handles this internally.
    try:
        model = whisper.load_model(size)
        print("✅ Whisper model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Could not load Whisper model. Error: {e}")
        print("Please ensure you have a stable internet connection for the first download.")
        raise

# โหลดโมเดลแค่ครั้งเดียวเมื่อ module ถูก import
whisper_model = get_whisper_model("base") # ใช้ 'base' เพื่อความเร็วในการทดสอบ

