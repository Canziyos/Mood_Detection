import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Initialize RetinaFace face detector
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))  # 0 = GPU, -1 = CPU

DATA_ROOT = r"C:\Users\Dator\OneDrive - Mälardalens universitet\Documents\data\ryeson_emo_db"
OUTPUT_DIR = r"C:\Users\Dator\Documents\ryerson_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

label_map = {
    "an": "Angry",
    "di": "Disgust",
    "fe": "Fear",
    "ha": "Happy",
    "sa": "Sad",
    "su": "Surprise"
}

def extract_5_faces(video_path, emotion, out_root, filename_prefix):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get number of frames in video.

    if total_frames < 5:
        return 0  # Skip very short videos.

    # Pick 5 frames spaced evenly across the video.
    frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)

    out_folder = os.path.join(out_root, emotion)
    os.makedirs(out_folder, exist_ok=True)

    saved = 0
    for target_frame in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
        ret, frame = cap.read()
        if not ret:
            continue  # Skip unreadable frames.

        faces = face_app.get(frame)  # Detect.

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # Skip empty crops.

            out_path = os.path.join(out_folder, f"{filename_prefix}_frame{target_frame:04d}_face{i}.jpg")
            cv2.imwrite(out_path, crop)  # Save cropped face image
            saved += 1

    cap.release()
    return saved

total_files = 0
processed_files = 0

for root, dirs, files in os.walk(DATA_ROOT):
    print("Scanning folder:", root) 
    for fname in files:
        print("Found file:", fname)
        if not fname.lower().endswith(".avi"):
            continue

        total_files += 1
        video_path = os.path.join(root, fname)
        prefix = fname[:2].lower()

        if prefix not in label_map:
            continue  # Skip unrecognized emotion prefixes.

        emotion = label_map[prefix]
        parent_folder = os.path.basename(root)
        segment_id = f"{parent_folder}_{fname[:-4]}"  # Build unique segment ID from folder + filename

        # Run face extraction and log how many faces were saved.
        count = extract_5_faces(video_path, emotion, OUTPUT_DIR, segment_id)
        processed_files += 1

# Summary.
print("Done. Total avi files found:", total_files)
print("Total processed:", processed_files)
