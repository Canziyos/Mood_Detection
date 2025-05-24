import os, warnings, cv2, torch, numpy as np, pandas as pd
from moviepy import VideoFileClip
from PIL import Image
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from audio import load_audio_model, audio_to_tensor, audio_predict
from image_model_interface import load_image_model, extract_image_features
from AudioImageFusion import AudioImageFusion

video_path   = "../../experiments/test_samples/4.mp4"
audio_model_path  = "../../models/mobilenetv2_aud.pth"
image_model_path  = "../../models/mobilenetv2_img.pth"
gate_ckpt    = "../../models/best_gate_head_logits.pth"
csv_out      = "../../results/clip_fusion.csv"

# Hyper-params
frames_num     = 10         # frames per clip
audio_win_s    = 1.0        # seconds of audio centered on each frame
classes        = ["Angry","Disgust","Fear","Happy","Neutral","Sad"]
fusion_type    = "avg"      # "avg" or "gate"
alfa           = 0.3        # used in "avg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models.
audio_model, _ = load_audio_model(audio_model_path)
image_model = load_image_model(image_model_path)

# Load fusion head.
if fusion_type == "avg":
    fusion_head = AudioImageFusion(
        num_classes=len(classes),
        fusion_mode="avg",
        alpha=alfa
    ).to(device)
    print(f"AudioImageFusion loaded for avg fusion (alpha={alfa}).")
elif fusion_type == "gate":
    ckpt = torch.load(gate_ckpt, map_location=device)
    fusion_head = AudioImageFusion(
        num_classes=len(classes),
        fusion_mode="gate"
    ).to(device)
    fusion_head.load_state_dict(ckpt["state_dict"])
    fusion_head.eval()
    print("Gate head loaded (logits-only).")
else:
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

# Helpers
def pick_frames(cap, n):
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, tot-1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok: frames.append((i, fr.copy()))
    return frames

def central_square(img):
    h, w, _ = img.shape
    sz = min(h, w)
    return cv2.resize(img[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2], (224, 224))

def audio_slice(wav, sr, center, win):
    half = int(win*sr/2)
    c    = int(center*sr)
    s, e = max(c-half, 0), min(c+half, wav.shape[-1])
    chunk = wav[..., s:e]
    if chunk.shape[-1] < win*sr:
        pad = win*sr - chunk.shape[-1]
        chunk = torch.nn.functional.pad(chunk, (0, int(pad)))
    return chunk

# Main loop
records = []

video_name = os.path.basename(video_path)
print(f"Processing {video_name}")

clip = VideoFileClip(video_path, audio=True)
sr = 16000
aud_np = clip.audio.to_soundarray(fps=sr).mean(axis=1)   # mono
wav = torch.from_numpy(aud_np).unsqueeze(0)              # (1, T)

cap = cv2.VideoCapture(video_path)
samples = pick_frames(cap, frames_num)
cap.release()
if not samples:
    warnings.warn(f"No frames in {video_name}, skipping.")

pa, pi, pf = [], [], []

for idx, frame in samples:
    t_sec = clip.duration * idx / clip.reader.n_frames   # timestamp

    # Audio branch
    chunk = audio_slice(wav, sr, t_sec, audio_win_s)
    chunk = chunk.float()
    aud_t = audio_to_tensor(chunk, sr)
    logits_a, probs_a, _, _ = audio_predict(audio_model, aud_t, device)
    logits_a = logits_a.to(device)
    probs_a  = probs_a.to(device)

    # Image branch
    crop = central_square(frame)
    lab_i, probs_i, logits_i, _ = extract_image_features(crop)
    logits_i = torch.tensor(logits_i, dtype=torch.float32, device=device)
    probs_i  = torch.tensor(probs_i,  dtype=torch.float32, device=device)

    # Fusion
    if fusion_type == "avg":
        fused_probs = fusion_head.fuse_probs(
            probs_audio=probs_a,
            probs_image=probs_i,
            pre_softmax_audio=logits_a,
            pre_softmax_image=logits_i
        )
    elif fusion_type == "gate":
        fused_probs, _ = fusion_head.fuse_probs(
            probs_audio=probs_a,
            probs_image=probs_i,
            pre_softmax_audio=logits_a,
            pre_softmax_image=logits_i,
            return_gate=True
        )
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    pa.append(probs_a.detach().cpu().numpy())
    pi.append(probs_i.detach().cpu().numpy())
    pf.append(fused_probs.detach().cpu().numpy())

pa = np.mean(np.vstack(pa), axis=0)
pi = np.mean(np.vstack(pi), axis=0)
pf = np.mean(np.vstack(pf), axis=0)

records.append(dict(
    video_file  = video_name,
    audio_pred  = classes[int(np.argmax(pa))],
    image_pred  = classes[int(np.argmax(pi))],
    fusion_pred = classes[int(np.argmax(pf))],
    audio_probs = pa.tolist(),
    image_probs = pi.tolist(),
    fusion_probs= pf.tolist()
))

# Save results
os.makedirs(os.path.dirname(csv_out), exist_ok=True)
pd.DataFrame(records).to_csv(csv_out, index=False)
print(f"\nSaved results to {csv_out}")

print(f"\nPredictions for {video_name}:")
print(f"- Audio    : {classes[int(np.argmax(pa))]}")
print(f"- Image    : {classes[int(np.argmax(pi))]}")
print(f"- Fusion   : {classes[int(np.argmax(pf))]}")
print(f"- Audio Prob: {pa.round(3)}")
print(f"- Image Prob: {pi.round(3)}")
print(f"- Fusion Prb: {pf.round(3)}")
