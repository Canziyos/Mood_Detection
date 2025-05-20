import sys, os, csv
sys.path.append(os.path.abspath('../src/fusion'))
sys.path.append(os.path.abspath('.'))

import cv2
from moviepy import VideoFileClip
import numpy as np
from PIL import Image
from ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from ex_image.image_model_interface import load_image_model, extract_image_features
from AV_Fusion import FusionAV
import torch

import os
import sys
from contextlib import contextmanager

# this part has nothing to do with fusion pipeline.
#-------------------------------------------------#
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
# -------------------------------------------------#

# Face detection util.
def detect_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        h, w, _ = img.shape
        sz = min(h, w)
        startx = w//2 - sz//2
        starty = h//2 - sz//2
        face_img = img[starty:starty+sz, startx:startx+sz]
    else:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (224, 224))
    return Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

video_root = "../dataset/video_test"
fusion_mode = "gate"
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

audio_model, device = load_audio_model(model_path="../models/mobilenetv2_aud.pth")
load_image_model(model_path="../models/mobilenetv2_img.pth", class_names=class_names)

results = []

for class_name in class_names:
    video_folder = os.path.join(video_root, class_name)
    video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith('.avi')])

    for vid_file in video_files:
        video_path = os.path.join(video_folder, vid_file)
        print(f"\nProcessing class: {class_name} | video: {vid_file}")

        # Extract audio (suppressed output)
        with suppress_stdout_stderr():
            clip = VideoFileClip(video_path)
            audio_array = clip.audio.to_soundarray(fps=16000)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        sr = 16000

        # Extract faces.
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = total_frames // 2
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vidcap.read()
        vidcap.release()
        if not ret:
            print("Could not read frame from video, skipping.")
            continue
        face_pil = detect_and_crop_face(frame)

        # Audio inference.
        aud_tensor = audio_to_tensor(waveform, sr)
        logits_a, softmax_a, latent_a, pred_a = audio_predict(audio_model, aud_tensor, device)

        
        # Image inference.
        face_pil = detect_and_crop_face(frame)
        face_np = np.array(face_pil)  # Convert PIL Image to NumPy array. (img model modified to accept path & np array)
        label_img, softmax_img, latent_img = extract_image_features(face_np)
        softmax_img = torch.tensor(softmax_img, dtype=torch.float32)
        latent_img = torch.tensor(latent_img, dtype=torch.float32).reshape(1, -1)


        # Fusion.
        fusion_model = FusionAV(
            num_classes=6,
            fusion_mode=fusion_mode,
            latent_dim_audio=latent_a.shape[1],
            latent_dim_image=latent_img.shape[1]
        )
        if fusion_mode == "latent":
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_img,
                latent_audio=latent_a, latent_image=latent_img
            )
        else:
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_img
            )

        fused_label = class_names[torch.argmax(fused_probs).item()]
        print(f"Fusion mode: {fusion_mode} | Fused pred: {fused_label}")

        results.append({
            "class": class_name,
            "video_file": vid_file,
            "audio_pred": pred_a,
            "image_pred": label_img,
            "fusion_pred": fused_label,
            "fusion_mode": fusion_mode,
            "audio_probs": softmax_a.tolist(),
            "image_probs": softmax_img.tolist(),
            "fusion_probs": fused_probs.tolist()
        })

# Save to CSV
with open(f"sync_fusion_results_{fusion_mode}.csv", "w", newline="") as f:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Done. Saved results for {fusion_mode}.")
