import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Initialize RetinaFace
print("Initializing RetinaFace...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("RetinaFace is ready.\n")

# Parameters
INPUT_ROOT = r"main_path_to_AFEW_VA"
OUTPUT_DIR = r"where_to_store\afew_selected_img_flat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FRAMES = 5
metadata = {}
total_faces = 0
total_images = 0

for folder_name in sorted(os.listdir(INPUT_ROOT)):
    folder_path = os.path.join(INPUT_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder_name}")

    json_path = os.path.join(folder_path, f"{folder_name}.json")
    if not os.path.exists(json_path):
        print(f"Missing JSON file: {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_data = data.get("frames", {})
    frame_names = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    if len(frame_names) < N_FRAMES:
        print(f"Not enough frames in {folder_name} (found {len(frame_names)}, need {N_FRAMES})")
        continue

    selected_indices = np.linspace(0, len(frame_names) - 1, N_FRAMES, dtype=int)

    for idx in selected_indices:
        fname = frame_names[idx]
        frame_id = fname[:-4]

        if frame_id not in frame_data:
            print(f"Frame ID {frame_id} not found in JSON.")
            continue

        va = {
            "valence": frame_data[frame_id]["valence"],
            "arousal": frame_data[frame_id]["arousal"]
        }

        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        print(f"Running face detection on: {img_path}")
        faces = face_app.get(img)

        if not faces:
            print(f"No faces detected in {fname}")
            continue

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"Skipped empty crop in {fname}")
                continue

            out_name = f"{folder_name}_{frame_id}_face{i}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, crop)

            metadata[out_name] = va
            total_faces += 1

        total_images += 1

# Save final combined JSON.
json_out_path = os.path.join(OUTPUT_DIR, "afew_va_faces.json")
with open(json_out_path, "w", encoding="utf-8") as jf:
    json.dump(metadata, jf, indent=2)

print("\nProcessing complete.")
print(f"Total frames processed: {total_images}")
print(f"Total faces saved: {total_faces}")
print(f"Metadata saved to: {json_out_path}")
