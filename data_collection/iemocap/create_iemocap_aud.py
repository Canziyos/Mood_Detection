import os
import shutil
import re

# According to the README, in the current folder, set the paths below.

# Define the root directory containing all emotion evaluation TXT files.
EVAL_DIR = r"path_to_your\emoevaluation"

# Define the root directory that contains all utterance-level WAV subfolders.
WAV_DIR = r"path_to_your\wav"

# Output directory where the filtered and relabeled WAV files will be stored.
OUTPUT_DIR = r"path_to_your\iemocap_audio"

# Mapping from raw IEMOCAP emotion labels to six target categories.
# Since we are using 6 categories for the discrete part, we map:
# frustration -> angry. excited -> happy. surprise -> happy.
emotion_map = {
    "ang": "Angry",
    "dis": "Disgust",
    "fea": "Fear",
    "hap": "Happy",
    "sur": "Happy",
    "neu": "Neutral",
    "sad": "Sad"
}

# Create destination folders based on mapped emotions if they do not exist.
for emotion in set(emotion_map.values()):
    out_dir = os.path.join(OUTPUT_DIR, emotion)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created or found output directory: {out_dir}")

# Regex pattern to match the summary line of each utterance entry.
line_pattern = re.compile(r"\[(.*?) - (.*?)\]\s+(\S+)\s+(\w+)\s+\[(.*?)\]")

moved = 0
skipped = 0  # Counter for skipped files.

# Iterate over all evaluation files.
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".txt"):
        continue  # Skip non-TXT files.

    base_folder = os.path.splitext(fname)[0]  # Extract base name to locate WAV folder.
    print(f"\nProcessing file: {fname} → Base folder: {base_folder}")

    with open(os.path.join(EVAL_DIR, fname), "r", encoding="utf-8") as f:
        for line in f:
            match = line_pattern.match(line.strip())
            if not match:
                continue  # Skip lines that do not match the utterance pattern.

            _, _, segment_id, raw_emotion, _ = match.groups()
            label = emotion_map.get(raw_emotion.lower())

            if label is None:
                print(f"[Skipping] Unknown or unmapped emotion: {raw_emotion} → {segment_id}")
                skipped += 1
                continue

            # Build the full WAV file path.
            wav_path = os.path.join(WAV_DIR, base_folder, f"{segment_id}.wav")
            if not os.path.exists(wav_path):
                print(f"[Missing] File not found: {wav_path}")
                skipped += 1
                continue

            # Copy the WAV file to the appropriate emotion folder.
            dest_path = os.path.join(OUTPUT_DIR, label, f"{segment_id}.wav")
            shutil.copy2(wav_path, dest_path)
            moved += 1
            print(f"[Copied] {segment_id}.wav → {label}/")

# summary.
print(f"\nFinished.\nCopied: {moved} files.\nSkipped: {skipped} files (missing or unknown emotion).")
