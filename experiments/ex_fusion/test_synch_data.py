import os, sys, cv2, torch, warnings, subprocess
import numpy as np, pandas as pd
from moviepy import VideoFileClip
from scipy.io.wavfile import write as wav_write
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.utils import load_config
from audio import load_audio_model, audio_to_tensor, audio_predict
from image_model_interface import load_image_model, extract_image_features
from utils.AudioImageFusion_bi import AudioImageFusion

# === Load config ===
config = load_config("config.yaml")

video_path = config["demo"]["video_clip"]
csv_out = config["demo"]["csv_out"]

audio_model_path = config["models"]["audio_model"]
image_model_path = config["models"]["image_model"]
gate_ckpt = config["models"]["gate"]

classes = config["classes"]
fusion_type = config["demo"]["fusion_type"]  # "avg" or "gate"
alpha = config["demo"].get("alpha", 0.3)
audio_win_s = config["demo"].get("audio_window", 1.0)

# === Use ffmpeg to extract clean audio ===
def extract_audio_ffmpeg(video_path, output_path, sr=16000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === Frame/audio helpers ===
def central_square(img):
    h, w, _ = img.shape
    sz = min(h, w)
    return cv2.resize(img[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2], (224, 224))

def audio_slice(wav, sr, center, win):
    half = int(win * sr / 2)
    c = int(center * sr)
    s, e = max(c - half, 0), min(c + half, wav.shape[-1])
    chunk = wav[..., s:e]
    if chunk.shape[-1] < win * sr:
        pad = int(win * sr - chunk.shape[-1])
        chunk = torch.nn.functional.pad(chunk, (0, pad))
    return chunk

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model, _ = load_audio_model(audio_model_path)
image_model = load_image_model(image_model_path)

if fusion_type == "avg":
    fusion_head = AudioImageFusion(len(classes), "avg", alpha).to(device)
    print(f"Fusion head loaded in avg mode (alpha={alpha})")
elif fusion_type == "gate":
    ckpt = torch.load(gate_ckpt, map_location=device)
    fusion_head = AudioImageFusion(len(classes), "gate").to(device)
    fusion_head.load_state_dict(ckpt["state_dict"])
    fusion_head.eval()
    print("Gate fusion head loaded.")
else:
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")

video_name = os.path.basename(video_path)
base_name = os.path.splitext(video_name)[0]

# === Extract audio ===
audio_out_path = f"extracted/audio/{base_name}_ffmpeg.wav"
extract_audio_ffmpeg(video_path, audio_out_path)

import torchaudio
wav, sr = torchaudio.load(audio_out_path)

# === Extract every 5th frame ===
os.makedirs("extracted/frames", exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Extracting every 5th frame from {total_frames} frames")

for idx in range(0, total_frames, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    out_path = os.path.join("extracted/frames", f"{base_name}_frame{idx}.jpg")
    cv2.imwrite(out_path, frame)

# === Step through video in 0.5s intervals ===
clip = VideoFileClip(video_path)
duration = clip.duration
records = []

for t_sec in np.arange(0.25, duration, 0.5):
    chunk = audio_slice(wav, sr, t_sec, win=0.5).float()
    aud_t = audio_to_tensor(chunk, sr)
    logits_a, probs_a, _, _ = audio_predict(audio_model, aud_t, device)
    logits_a, probs_a = logits_a.to(device), probs_a.to(device)

    frame_start = int((t_sec - 0.25) * fps)
    frame_end = int((t_sec + 0.25) * fps)
    frame_indices = list(range(frame_start, frame_end + 1))  # Use all frames in the 0.5s window

    image_probs_list = []
    logits_i_list = []

    for idx in frame_indices:
        if idx < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = central_square(frame)
        _, probs_i, logits_i, _ = extract_image_features(crop)
        probs_i = torch.tensor(probs_i, dtype=torch.float32, device=device)
        logits_i = torch.tensor(logits_i, dtype=torch.float32, device=device)
        image_probs_list.append(probs_i)
        logits_i_list.append(logits_i)

    if not image_probs_list:
        print(f"Warning: No valid frames around {t_sec:.2f}s.")
        continue

    probs_i = torch.stack(image_probs_list).mean(dim=0)
    logits_i = torch.stack(logits_i_list).mean(dim=0)

    if fusion_type == "avg":
        fused_probs = fusion_head.fuse_probs(probs_a, probs_i)
    elif fusion_type == "gate":
        fused_probs, _ = fusion_head.fuse_probs(
            probs_audio=probs_a,
            probs_image=probs_i,
            pre_softmax_audio=logits_a,
            pre_softmax_image=logits_i,
            return_gate=True
        )

    records.append(dict(
        video_file=video_name,
        timestamp=round(t_sec, 2),
        audio_pred=classes[int(torch.argmax(probs_a))],
        image_pred=classes[int(torch.argmax(probs_i))],
        fusion_pred=classes[int(torch.argmax(fused_probs))],
        audio_probs=probs_a.detach().cpu().numpy().tolist(),
        image_probs=probs_i.detach().cpu().numpy().tolist(),
        fusion_probs=fused_probs.detach().cpu().numpy().tolist()
    ))

    print(f"\n--- t = {t_sec:.2f}s ---")
    print(f"- Audio   : {classes[int(torch.argmax(probs_a))]}")
    print(f"- Image   : {classes[int(torch.argmax(probs_i))]}")
    print(f"- Fusion  : {classes[int(torch.argmax(fused_probs))]}")
    print(f"- Audio probs : {probs_a.squeeze().detach().cpu().numpy().round(3)}")
    print(f"- Image probs : {probs_i.squeeze().detach().cpu().numpy().round(3)}")
    print(f"- Fusion probs: {fused_probs.squeeze().detach().cpu().numpy().round(3)}")

cap.release()
pd.DataFrame(records).to_csv(csv_out, index=False)
print(f"\nSaved all predictions to {csv_out}")
