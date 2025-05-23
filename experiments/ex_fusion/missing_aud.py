# ------------------------------------------------------------#
# Unsynchronized image-only fusion test (LOGITS-ONLY, no audio).
# ------------------------------------------------------------#
import sys, os, torch, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Project path hack.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.ex_image.image_model_interface import load_image_model, extract_image_features
from src.fusion.AV_Fusion import FusionAV

# Config.
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
image_root = "./EmoDB_For_AudioTest"

fusion_type = "gate"  # "avg" or "gate".
alpha = 0.3           # Used for "avg" only.
ckpt_path = "./models/best_gate_head_logits.pth"
out_dir = "./results"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/unsync_compare_{fusion_type}_noaudio.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models.
image_model = load_image_model(model_path="./models/mobilenetv2_img.pth")
if fusion_type == "avg":
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="avg",
        alpha=alpha
    ).to(device)
elif fusion_type == "gate":
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="gate",
        latent_dim_audio=None,
        latent_dim_image=None,
        use_latents=False
    ).to(device)
    fusion_head.load_state_dict(ckpt["state_dict"])
    fusion_head.eval()
else:
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

# Inference.
records = []
y_true, y_i, y_f = [], [], []
alpha_a_list, alpha_i_list = [], []

with torch.no_grad():
    for cname in class_names:
        i_dir = os.path.join(image_root, cname)
        if not os.path.exists(i_dir):
            print(f"Skipping class '{cname}' (missing folder).")
            continue
        i_files = sorted([f for f in os.listdir(i_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if not i_files:
            print(f"Skipping class '{cname}' (no images).")
            continue

        for img_name in i_files:
            img_path = os.path.join(i_dir, img_name)
            # Simulate no audio (all zeros).
            logits_a = torch.zeros(1, len(class_names)).to(device)
            probs_a  = torch.zeros(1, len(class_names)).to(device)
            pred_a   = 'NO_AUDIO'

            # Image forward.
            lab_i, probs_i, logits_i, _ = extract_image_features(img_path)
            logits_i = torch.tensor(logits_i, dtype=torch.float32).unsqueeze(0).to(device)
            probs_i  = torch.tensor(probs_i,  dtype=torch.float32).unsqueeze(0).to(device)

            # Fusion.
            if fusion_type == "avg":
                fused_probs = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i
                )
                alpha_weights = torch.tensor([[fusion_head.alpha, 1 - fusion_head.alpha]])
            elif fusion_type == "gate":
                fused_probs, alpha = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i,
                    latent_audio=None,
                    latent_image=None,
                    return_gate=True
                )
                alpha_a, alpha_i = alpha[:, 0], alpha[:, 1]
                alpha_weights = torch.stack([alpha_a, alpha_i], dim=1)
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            pred_f = class_names[torch.argmax(fused_probs).item()]

            y_true.append(class_names.index(cname))
            y_i.append(class_names.index(lab_i))
            y_f.append(class_names.index(pred_f))
            alpha_a_list.append(alpha_weights[0, 0].item())
            alpha_i_list.append(alpha_weights[0, 1].item())

            records.append({
                "class": cname,
                "image_file": img_name,
                "audio_pred": pred_a,
                "image_pred": lab_i,
                "fusion_pred": pred_f,
                "alpha_audio": alpha_weights[0, 0].item(),
                "alpha_image": alpha_weights[0, 1].item(),
            })
        print(f"[{cname}] {len(i_files)} image samples processed.")

# Metrics & Save.
print("\n=== IMAGE-ONLY vs FUSION ===")
for name, y_hat in [("Image", y_i), ("Fusion", y_f)]:
    print(f"\n{name} report:")
    print(classification_report(y_true, y_hat, target_names=class_names, digits=3))

mean_a = np.mean(alpha_a_list) if alpha_a_list else 0
mean_i = np.mean(alpha_i_list) if alpha_i_list else 0
print(f"\nalfa audio (mean): {mean_a:.3f}   |   alfa image (mean): {mean_i:.3f}")

pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nResults written to {csv_path}")
