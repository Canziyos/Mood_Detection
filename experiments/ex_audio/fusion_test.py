import sys, os
sys.path.append(os.path.abspath('../src/fusion'))

import torch
import numpy as np
from audio import load_audio_model, audio_to_tensor, audio_predict
from AV_Fusion import FusionAV
import imageio
from moviepy import VideoFileClip
import torchaudio

# --- Config ---
vid_path = r"./4.mp4"
audio_path = r"./4.wav"
fusion_mode = "avg" # options: "avg", "mlp", "gate", "prod" and "latent" (if not, avg by defult)
num_classes = 6
fps = 25
img_emb_dim = 1280
chunk_len_sec = 0.5
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Extract audio if needed.
if not os.path.exists(audio_path):
    video = VideoFileClip(vid_path)
    video.audio.write_audiofile(audio_path)

# Load audio and model (once).
full_waveform, sr = torchaudio.load(audio_path)
model, device = load_audio_model()

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

fusion_model = None
reader = imageio.get_reader(vid_path)
for frame_idx, frame in enumerate(reader):

    # Fake image branch for now.
    fake_img_softmax = torch.tensor(np.random.dirichlet(np.ones(num_classes)), dtype=torch.float32) #.unsqueeze(0)
    fake_img_emb = torch.tensor(np.random.randn(1, img_emb_dim), dtype=torch.float32)

    # Real audio branch (using refactored audio.py).
    audio_chunk = get_audio_chunk(frame_idx)
    img_tensor = audio_to_tensor(audio_chunk, sr)
    logits, audio_softmax, aud_features, predicted = audio_predict(model, img_tensor, device)
    
    # Print logits and softmax for inspection. (i made a huge mistake before, when reviewing image brach)
    print(f"Frame {frame_idx}:")
    print("- Logits:", logits.cpu().numpy())
    print("- Softmax:", audio_softmax.cpu().numpy())
    print("- oftmax sum:", audio_softmax.sum().item())
    print(f"- Softmax sums to 1? {'YES' if np.allclose(audio_softmax.sum().item(), 1.0, atol=1e-5) else 'NO'}")

    if fusion_model is None:
        fusion_model = FusionAV(
            num_classes=num_classes,
            fusion_mode=fusion_mode,
            latent_dim_audio=aud_features.shape[1],
            latent_dim_image=img_emb_dim
        )
    if fusion_mode == "latent":
        fused_probs = fusion_model.fuse_probs(
            probs_audio=audio_softmax, probs_image=fake_img_softmax,
            latent_audio=aud_features, latent_image=fake_img_emb,
        )
    else:
        fused_probs = fusion_model.fuse_probs(
            probs_audio=audio_softmax, probs_image=fake_img_softmax
        )
    fused_label = torch.argmax(fused_probs, dim=1).item()
    print(
        f"Frame {frame_idx}: fused_label={class_names[fused_label]}, "
        f"audio_label={predicted}"
    )

reader.close()
print("Done.")
