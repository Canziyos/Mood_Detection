# This script evaluates audio and image models on test sets non synchronised inputs
# and fuses their predictions uing different late fusion strategies.

# Set fusion_mode at the top to control which strategy to test.

# Results are saved as a CSV file for later analysis.
#
# Audio and image files should be in class subfolders under /dataset/audio/test and /dataset/images/test.

# For "gate" and "mlp", the fusion uses model logits (pre-softmax); for "latent", it uses latent features.
# "avg" and "prod" use softmax probabilities.

# For -avg- and -prod- modes, you can override the default AV_Fusion logic to apply custom weights
# (e.g., change 'alpha' to adjust audio/image importance).

import sys, os

sys.path.append(os.path.abspath('../src/fusion'))
sys.path.append(os.path.abspath('.'))


import csv
from ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from ex_image.image_model_interface import load_image_model, extract_image_features
from AV_Fusion import FusionAV
import torch
import torchaudio

audio_root = "../dataset/audio/test"
image_root = "../dataset/images/test"
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Set fusion mode here.
fusion_mode = "gate"  # Change to "prod", "gate", "mlp", "latent" in other runs


audio_model, device = load_audio_model(model_path="../models/mobilenetv2_aud_68.35.pth")
load_image_model(model_path="../models/mobilenetv2_emotion.pth", class_names=class_names)

results = []

for class_name in class_names:
    audio_folder = os.path.join(audio_root, class_name)
    image_folder = os.path.join(image_root, class_name)
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    n_pairs = min(len(audio_files), len(image_files))

    for i in range(n_pairs):
        audio_path = os.path.join(audio_folder, audio_files[i])
        image_path = os.path.join(image_folder, image_files[i])

        print(f"\nProcessing class: {class_name} | audio: {audio_files[i]} | image: {image_files[i]}")

        # Audio.
        waveform, sr = torchaudio.load(audio_path)
        print(f"Loaded audio: {audio_path} | waveform shape: {waveform.shape} | sr: {sr}")
        aud_tensor = audio_to_tensor(waveform, sr)
        logits_a, softmax_a, latent_a, pred_a = audio_predict(audio_model, aud_tensor, device)
        print(f"Audio pred: {pred_a} | Audio softmax: {softmax_a.tolist()} | Latent shape: {latent_a.shape}")

        # Image.
        label_i, softmax_i, logits_i, latent_i = extract_image_features(image_path)
        softmax_i = torch.tensor(softmax_i, dtype=torch.float32)
        logits_i = torch.tensor(logits_i, dtype=torch.float32)  # added for logits.
        latent_i = torch.tensor(latent_i, dtype=torch.float32).reshape(1, -1)
        print(f"Image pred: {label_i} | Image softmax: {softmax_i.tolist()} | Latent shape: {latent_i.shape}")

        # Fusion model, with logits enabled
        fusion_model = FusionAV(
            num_classes=6,
            fusion_mode=fusion_mode,
            latent_dim_audio=latent_a.shape[1],
            latent_dim_image=latent_i.shape[1],
            use_pre_softmax=True
        )

        if fusion_mode == "latent":
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_i,
                latent_audio=latent_a, latent_image=latent_i
            )
        elif fusion_mode in ["mlp", "gate"]:
            # pass the logits for both branches here!
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_i,  # Still needed for fallback, shape checks, warnings.
                pre_softmax_audio=logits_a, pre_softmax_image=logits_i 
            )
        elif fusion_mode == "avg":
            alpha = 0.2
            fused_probs = alpha * softmax_a + (1 - alpha) * softmax_i
        elif fusion_mode == "prod":
            fused_probs = (softmax_a * softmax_i) / (softmax_a * softmax_i).sum()
        else:
            fused_probs = fusion_model.fuse_probs(
                probs_audio=softmax_a, probs_image=softmax_i
            )


        fused_label = class_names[torch.argmax(fused_probs).item()]
        print(f"Fusion mode: {fusion_mode} | Fused pred: {fused_label} | Fused softmax: {fused_probs.tolist()}")


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

# Save to CSV.
with open(f"unsync_fusion_results_{fusion_mode}.csv", "w", newline="") as f:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Done. Saved results for {fusion_mode}.")
