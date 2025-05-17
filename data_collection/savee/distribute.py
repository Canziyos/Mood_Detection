import os
import shutil

# Path to the SAVEE 'ALL' folder containing all .wav files.
INPUT_DIR = r"path_to_unzipped\savee\ALL"

# Output base directory where files will be copied into subfolders by emotion.
OUTPUT_DIR = r"storage_path_to\savee_audio"

# Mapping from SAVEE emotion codes to full labels.
emotion_map = {
    "a": "Angry",
    "d": "Disgust",
    "f": "Fear",
    "h": "Happy",
    "n": "Neutral",
    "sa": "Sad",
}

# emotion folders.
for emotion in set(emotion_map.values()):
    out_path = os.path.join(OUTPUT_DIR, emotion)
    os.makedirs(out_path, exist_ok=True)
    print(f"Created folder: {out_path}")

moved = 0
skipped = 0

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".wav"):
        continue

    name_parts = fname.split("_")
    if len(name_parts) != 2:
        print(f"Skipping invalid filename: {fname}")
        skipped += 1
        continue

    speaker, emo_code = name_parts
    emo_code = emo_code.lower().replace(".wav", "")  # e.g., 'a01', 'sa14'.
    
    # Special case: 2-letter codes like 'sa', 'su'.
    code = emo_code[:2] if emo_code[:2] in emotion_map else emo_code[0]
    emotion = emotion_map.get(code)

    if emotion is None:
        print(f"Unknown emotion in: {fname}")
        skipped += 1
        continue

    src = os.path.join(INPUT_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, emotion, fname)
    shutil.copy2(src, dst)
    moved += 1
    print(f"Copied: {fname} → {emotion}/")

print(f"\nDone. {moved} files copied. {skipped} skipped.")
