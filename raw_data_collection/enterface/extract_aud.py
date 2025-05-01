import os
from moviepy.editor import VideoFileClip

# Input and output.
VIDEO_ROOT = r"path_to_your_unzipped_folder_enterface_db"
AUDIO_OUTPUT = r"to your_output (enterface_audio)"

# Emotion mapping as in the image extraction.
label_map = {
    "an": "Angry",
    "di": "Disgust",
    "fe": "Fear",
    "ha": "Happy",
    "sa": "Sad",
    "su": "Surprise"
}

# Traverse all avi files.
for root, _, files in os.walk(VIDEO_ROOT):
    for fname in files:
        if not fname.lower().endswith(".avi"):
            continue

        parts = fname.lower().split("_")
        if len(parts) < 3:
            continue

        prefix = parts[1]  # e.g., 'an'
        if prefix not in label_map:
            continue

        emotion = label_map[prefix]
        relative_path = os.path.relpath(root, VIDEO_ROOT)
        flat_path = relative_path.replace(os.sep, "_").replace(" ", "")
        segment_id = f"{flat_path}_{fname[:-4]}"  # e.g., subject1_s1_an_1

        # Prepare output path
        out_folder = os.path.join(AUDIO_OUTPUT, emotion)
        os.makedirs(out_folder, exist_ok=True)
        audio_path = os.path.join(out_folder, f"{segment_id}.wav")

        # Extract audio from .avi (here i use moviepy).
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
