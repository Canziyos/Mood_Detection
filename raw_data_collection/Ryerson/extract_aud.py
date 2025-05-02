import os
from moviepy.editor import VideoFileClip

# Set paths for the Ryerson dataset and output
VIDEO_ROOT = r"C:\Users\Dator\OneDrive - Mälardalens universitet\Documents\data\ryeson_emo_db"
AUDIO_OUTPUT = r"C:\Users\Dator\Documents\ryerson_audio"

# Emotion label map (same as your face extraction script)
label_map = {
    "an": "Angry",
    "di": "Disgust",
    "fe": "Fear",
    "ha": "Happy",
    "sa": "Sad",
    "su": "Surprise"
}

# Traverse video files
for root, _, files in os.walk(VIDEO_ROOT):
    for fname in files:
        if not fname.lower().endswith(".avi"):
            continue

        prefix = fname[:2].lower()
        if prefix not in label_map:
            continue  # Skip unknown labels

        emotion = label_map[prefix]
        parent_folder = os.path.basename(root)
        segment_id = f"{parent_folder}_{fname[:-4]}"

        # Prepare output folder and audio file path
        out_folder = os.path.join(AUDIO_OUTPUT, emotion)
        os.makedirs(out_folder, exist_ok=True)
        audio_path = os.path.join(out_folder, f"{segment_id}.wav")

        video_path = os.path.join(root, fname)
        try:
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio:
                    audio.write_audiofile(audio_path, verbose=False, logger=None)
                    print("Extracted:", audio_path)
                else:
                    print("No audio found in:", video_path)
        except Exception as e:
            print("Failed to extract from:", video_path, "| Error:", str(e))
