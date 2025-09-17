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

def transcribe_video_audio(video_path: str) -> list:
    """
    Extracts audio from a video, transcribes it using Whisper,
    and returns a list of timed caption segments.
    """
    video_path_obj = Path(video_path)
    audio_path = video_path_obj.with_suffix(".wav")

    try:
        # 1. Extract audio if it doesn't exist
        if not audio_path.exists():
            print(f"🔊 Extracting audio from '{video_path_obj.name}'...")
            with VideoFileClip(video_path) as clip:
                clip.audio.write_audiofile(str(audio_path), fps=16000, logger=None)
        
        # 2. Transcribe using Whisper
        print(f"🎤 Transcribing audio with Whisper...")
        result = whisper_model.transcribe(str(audio_path), language="th", fp16=False)

        captions = []
        for seg in result.get("segments", []):
            captions.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip()
            })
        
        print(f"✅ Transcription complete. Found {len(captions)} segments.")
        return captions

    finally:
        # 3. Clean up the temporary audio file
        if audio_path.exists():
            os.remove(audio_path)
