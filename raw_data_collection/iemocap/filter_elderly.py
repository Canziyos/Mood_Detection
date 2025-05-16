import os
import shutil
import cv2
from insightface.app import FaceAnalysis

# Setup
INPUT_FOLDER = r"C:\Users\Dator\Documents\ryerson_Images"
OUTPUT_FOLDER = r"C:\Users\Dator\Documents\ryerson_elderly_faces"
AGE_THRESHOLD = 65

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Traverse input folders (by emotion)
for emotion in os.listdir(INPUT_FOLDER):
    emo_path = os.path.join(INPUT_FOLDER, emotion)
    if not os.path.isdir(emo_path):
        continue

    out_emo_path = os.path.join(OUTPUT_FOLDER, emotion)
    os.makedirs(out_emo_path, exist_ok=True)

    for fname in os.listdir(emo_path):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        img_path = os.path.join(emo_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        elderly_found = any(f.age >= AGE_THRESHOLD for f in faces)

        if elderly_found:
            shutil.copy(img_path, os.path.join(out_emo_path, fname))
            print("Kept:", fname)
        else:
            print("Skipped (not elderly):", fname)
