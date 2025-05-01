import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Init RetinaFace.
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))  # 0 = GPU, -1 = CPU

# Input and output paths.
DATA_ROOT = r"to your unzipped_enterface_db_file"
OUTPUT_DIR = r"output_dir\enterface_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion code mapping.
label_map = {
    "an": "Angry",
    "di": "Disgust",
    "fe": "Fear",
    "ha": "Happy",
    "sa": "Sad",
    "su": "Surprise"
}

# extract and save 5 face crops.
def extract_5_faces(video_path, emotion, out_root, filename_prefix):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        print(f"Skipping {filename_prefix}: too short ({total_frames} frames)")
        return 0

    frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    out_folder = os.path.join(out_root, emotion)
    os.makedirs(out_folder, exist_ok=True)

    saved = 0
    for target_frame in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {target_frame} in {filename_prefix}")
            continue

        faces = face_app.get(frame)
        print(f"{filename_prefix}: {len(faces)} face(s) detected in frame {target_frame}")

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_path = os.path.join(out_folder, f"{filename_prefix}_frame{target_frame:04d}_face{i}.jpg")
            cv2.imwrite(out_path, crop)
            saved += 1

    cap.release()
    return saved

# walk through all .avi files.
total_files = 0
processed_files = 0

for root, dirs, files in os.walk(DATA_ROOT):
    print("Scanning folder:", root)
    for fname in files:
        if not fname.lower().endswith(".avi"):
            continue

        total_files += 1
        video_path = os.path.join(root, fname)

        # Extract emotion prefix from filename.
        parts = fname.lower().split("_")
        if len(parts) < 2:
            print(f"Skipping {fname}: unexpected filename format")
            continue

        prefix = parts[1]  # s14_an_2.avi → 'an'
        if prefix not in label_map:
            print(f"Skipping {fname}: unknown emotion prefix '{prefix}'")
            continue

        emotion = label_map[prefix]

        # Segment ID from folder structure and filename.
        relative_path = os.path.relpath(root, DATA_ROOT)
        flat_path = relative_path.replace(os.sep, "_").replace(" ", "")
        segment_id = f"{flat_path}_{fname[:-4]}"

        count = extract_5_faces(video_path, emotion, OUTPUT_DIR, segment_id)
        print(f"{segment_id}: saved {count} face(s)")
        processed_files += 1

print("Done. Total .avi files found:", total_files)
print("Total processed:", processed_files)
