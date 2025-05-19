import sys, os

sys.path.append(os.path.abspath('../src/fusion'))
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import imageio
from ex_image.image_model_interface import load_image_model, extract_image_features
from ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from AV_Fusion import FusionAV
from moviepy import VideoFileClip

# --- Config ---
vid_path = "./test_samples/4.mp4"
fusion_mode = "avg"  # options: "avg", "mlp", "gate", "prod", "latent"
num_classes = 6
fps = 25
chunk_len_sec = 0.5
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
subsample_rate = 5         # Process every 5th frame.
window_size = 10           # Aggregate over 10 processed frames (2 seconds at 5 fps).

img_softmax_list = []
audio_softmax_list = []
fused_softmax_list = []
fusion_model = None

# Load audio from video directly.
video = VideoFileClip(vid_path)
audio_array = video.audio.to_soundarray(fps=16000)
if audio_array.ndim > 1:
    audio_array = audio_array.mean(axis=1)
full_waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
sr = 16000

# Load models.
load_image_model(model_path="../models/mobilenetv2_emotion.pth", class_names=class_names)
audio_model, device = load_audio_model(model_path="../models/mobilenetv2_aud_68.35.pth")

def get_audio_chunk(frame_idx):
    center_sec = frame_idx / fps
    start = int(max((center_sec - chunk_len_sec/2) * sr, 0))
    end = int(min((center_sec + chunk_len_sec/2) * sr, full_waveform.shape[1]))
    chunk = full_waveform[:, start:end]
    min_len = int(chunk_len_sec * sr)
    if chunk.shape[1] < min_len:
        pad = torch.zeros(chunk.shape[0], min_len - chunk.shape[1])
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk

reader = imageio.get_reader(vid_path)

for frame_idx, frame in enumerate(reader):
    if frame_idx % subsample_rate != 0:
        continue  # Skip for subsampling

    # IMAGE.
    try:
        label_img, img_softmax, img_emb = extract_image_features(frame)
        img_softmax = torch.tensor(img_softmax, dtype=torch.float32)
        img_emb = torch.tensor(img_emb, dtype=torch.float32).reshape(1, -1)
        print(f"Frame {frame_idx} | image softmax sum: {img_softmax.sum().item():.5f}")
    except Exception as e:
        print(f"Frame {frame_idx}: ERROR in image model ({e}), skipping.")
        continue

    # AUDIO.
    audio_chunk = get_audio_chunk(frame_idx)
    aud_tensor = audio_to_tensor(audio_chunk, sr)
    logits, audio_softmax, aud_features, predicted_audio = audio_predict(audio_model, aud_tensor, device)


    #print(f"Frame {frame_idx} | audio softmax sum: {audio_softmax.sum().item():.5f}")

    # FUSION.
    if fusion_model is None:
        fusion_model = FusionAV(
            num_classes=num_classes,
            fusion_mode=fusion_mode,
            latent_dim_audio=aud_features.shape[1],
            latent_dim_image=img_emb.shape[1]
        )
    if fusion_mode == "latent":
        fused_probs = fusion_model.fuse_probs(
            probs_audio=audio_softmax, probs_image=img_softmax,
            latent_audio=aud_features, latent_image=img_emb
        )
    else:
        fused_probs = fusion_model.fuse_probs(
            probs_audio=audio_softmax, probs_image=img_softmax
        )
    # print(f"Frame {frame_idx} | fused softmax sum: {fused_probs.sum().item():.5f}")

    # Store for window aggregation.
    img_softmax_list.append(img_softmax)
    audio_softmax_list.append(audio_softmax)
    fused_softmax_list.append(fused_probs)

    # Window aggregation.
    if len(img_softmax_list) == window_size:
        img_softmax_avg = torch.stack(img_softmax_list).mean(dim=0)
        audio_softmax_avg = torch.stack(audio_softmax_list).mean(dim=0)
        fused_softmax_avg = torch.stack(fused_softmax_list).mean(dim=0)

        img_label = class_names[torch.argmax(img_softmax_avg).item()]
        audio_label = class_names[torch.argmax(audio_softmax_avg).item()]
        fused_label = class_names[torch.argmax(fused_softmax_avg).item()]

        print(f"Frames {frame_idx-subsample_rate*(window_size-1)}-{frame_idx} (window): "
              f"image={img_label}, audio={audio_label}, fused={fused_label}")

        img_softmax_list.clear()
        audio_softmax_list.clear()
        fused_softmax_list.clear()

reader.close()
print("Done.")
