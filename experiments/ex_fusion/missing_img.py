import sys, os, torch, torchaudio, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path for local imports.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.ex_fusion.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_fusion.image_model_interface import load_image_model
from src.fusion.AudioImageFusion import FusionAV

# Config.
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
audio_root = "./EmoDB_For_AudioTest"
image_down = True  # Simulate image missing/down.
fusion_type = "gate"  # "gate" or "avg".
alpha = 0.3  # Only used if fusion_type="avg".
ckpt_path = "./models/best_gate_head_logits.pth"
out_dir = "./results"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/unsync_compare_{fusion_type}_imgdown.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models.
audio_model, _ = load_audio_model(model_path="./models/mobilenetv2_aud.pth")
print("Audio backbone loaded.")

# Load fusion model.
if fusion_type == "avg":
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="avg",
        alpha=alpha
    ).to(device)
    print(f"FusionAV loaded for avg fusion (alpha={alpha}).")
elif fusion_type == "gate":
    ckpt = torch.load(ckpt_path, map_location=device)
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="gate"
    ).to(device)
    fusion_head.load_state_dict(ckpt["state_dict"])
    fusion_head.eval()
    print("Gate head loaded.")
else:
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

# Inference loop.
records, y_true, y_a, y_i, y_f = [], [], [], [], []
alpha_a_list, alpha_i_list = [], []

with torch.no_grad():
    for cname in class_names:
        a_dir = os.path.join(audio_root, cname)
        if not os.path.exists(a_dir):
            print(f"Skipping class '{cname}' (missing audio folder).")
            continue

        a_files = sorted([f for f in os.listdir(a_dir) if f.endswith(".wav")])
        N = len(a_files)
        if N == 0:
            print(f"Skipping class '{cname}' (no audio files).")
            continue

        for idx in range(N):
            # Audio forward.
            wav, sr = torchaudio.load(os.path.join(a_dir, a_files[idx]))
            aud_t = audio_to_tensor(wav, sr)
            logits_a, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)

            # Simulate image down: Set image logits and probs to zeros.
            logits_i = torch.zeros_like(logits_a)
            probs_i = torch.zeros_like(probs_a)
            lab_i = "IMG_DOWN"

            logits_a = logits_a.to(device)
            probs_a = probs_a.to(device)
            logits_i = logits_i.to(device)
            probs_i = probs_i.to(device)

            # Fusion.
            if fusion_type == "avg":
                fused_probs = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i
                )
                alpha_weights = torch.tensor([[fusion_head.alpha, 1-fusion_head.alpha]])
            elif fusion_type == "gate":
                fused_probs, alpha = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i,
                    return_gate=True
                )
                alpha_a, alpha_i = alpha[:, 0], alpha[:, 1]
                alpha_weights = torch.stack([alpha_a, alpha_i], dim=1)
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            pred_f = class_names[torch.argmax(fused_probs).item()]

            # Bookkeeping.
            y_true.append(class_names.index(cname))
            y_a.append(class_names.index(pred_a))
            y_i.append(-1)  # Not a real class, since image is missing.
            y_f.append(class_names.index(pred_f))
            alpha_a_list.append(alpha_weights[0, 0].item())
            alpha_i_list.append(alpha_weights[0, 1].item())

            records.append({
                "class": cname,
                "audio_file": a_files[idx],
                "image_pred": lab_i,
                "fusion_pred": pred_f,
                "alpha_audio": alpha_weights[0, 0].item(),
                "alpha_image": alpha_weights[0, 1].item(),
            })

        print(f"[{cname}] audio-only {N} samples.")

# Metrics & Save.
print("\n=== AUDIO-ONLY vs FUSION (Image Down) ===")
print("\nAudio report:")
print(classification_report(y_true, y_a, target_names=class_names, digits=3))
print("\nFusion report:")
print(classification_report(y_true, y_f, target_names=class_names, digits=3))

mean_a = np.mean(alpha_a_list) if alpha_a_list else 0
mean_i = np.mean(alpha_i_list) if alpha_i_list else 0
print(f"\nAlpha audio (mean): {mean_a:.3f}   |   Alpha image (mean): {mean_i:.3f}")

pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nResults written to {csv_path}")
