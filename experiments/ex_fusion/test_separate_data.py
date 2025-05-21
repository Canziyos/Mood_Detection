import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torchaudio
import numpy as np
from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_image.image_model_interface import load_image_model, extract_image_features
from src.fusion.AV_Fusion import FusionAV
import csv

audio_root = "./Dataset/audio/test"
image_root = "./Dataset/images/test"
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Set fusion mode here.
fusion_mode = "gate"  # "avg", "gate", or "latent"

audio_model, device = load_audio_model(model_path="./models/mobilenetv2_aud.pth")
load_image_model(model_path="./models/mobilenetv2_img.pth", class_names=class_names)
print("models loaded.")
fusion_model = None
if fusion_mode == "gate":
    print("Loading gate fusion head...")
    fusion_model = FusionAV(
        num_classes=6,
        fusion_mode="gate",
        use_pre_softmax=True
    )
    fusion_model.load_state_dict(torch.load("./models/best_logits_head3225.pth", map_location=device))
    fusion_model.to(device)
    fusion_model.eval()
    print("Gate head loaded!")
    
elif fusion_mode == "latent":
    fusion_model = FusionAV(
        num_classes=6,
        fusion_mode="latent",
        latent_dim_audio=1280,
        latent_dim_image=1280
    )
    fusion_model.load_state_dict(torch.load("./models/best_latent_head.pth", map_location=device))
    fusion_model.to(device)
    fusion_model.eval()
    print("latent head loaded!")

results = []

for class_name in class_names:
    audio_folder = os.path.join(audio_root, class_name)
    image_folder = os.path.join(image_root, class_name)
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    n_pairs = min(len(audio_files), len(image_files))
    print(f"  Audio files: {len(audio_files)} | Image files: {len(image_files)} | Pairing up: {n_pairs} samples.")

    for i in range(n_pairs):
        audio_path = os.path.join(audio_folder, audio_files[i])
        image_path = os.path.join(image_folder, image_files[i])

        waveform, sr = torchaudio.load(audio_path)
        aud_tensor = audio_to_tensor(waveform, sr)
        logits_a, softmax_a, latent_a, pred_a = audio_predict(audio_model, aud_tensor, device)

        label_i, softmax_i, logits_i, latent_i = extract_image_features(image_path)
        softmax_i = torch.tensor(softmax_i, dtype=torch.float32).to(device)
        logits_i = torch.tensor(logits_i, dtype=torch.float32).to(device)
        latent_i = torch.tensor(latent_i, dtype=torch.float32).reshape(1, -1).to(device)
        logits_a = logits_a.to(device)
        softmax_a = softmax_a.to(device)
        latent_a = latent_a.to(device)
        
        # === Fusion ===#
        if fusion_mode == "avg":
            alpha = 0.5
            fused_probs = alpha * softmax_a + (1 - alpha) * softmax_i
        elif fusion_mode == "gate":
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_i,
                pre_softmax_audio=logits_a, pre_softmax_image=logits_i
            )
        elif fusion_mode == "latent":
            fused_probs = fusion_model.fuse_probs(
                probs_audio=None, probs_image=None,
                latent_audio=latent_a, latent_image=latent_i
            )
        else:
            raise ValueError("Unsupported fusion_mode! Use 'avg', 'gate', or 'latent'.")
        
        fused_label = class_names[torch.argmax(fused_probs).item()]
        print(f"Fusion: {fused_label} (Audio: {pred_a}, Image: {label_i})")

        results.append({
            "class": class_name,
            "audio_file": audio_files[i],
            "image_file": image_files[i],
            "audio_pred": pred_a,
            "image_pred": label_i,
            "fusion_pred": fused_label,
            "fusion_mode": fusion_mode,
            "audio_probs": softmax_a.tolist(),
            "image_probs": softmax_i.tolist(),
            "fusion_probs": fused_probs.tolist()
        })

out_dir = "./results"
os.makedirs(out_dir, exist_ok=True)

save_path = f"{out_dir}/unsync_fusion_results_{fusion_mode}.csv"
with open(save_path, "w", newline="") as f:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Done. Saved results for {fusion_mode} at: {os.path.abspath(save_path)}")
