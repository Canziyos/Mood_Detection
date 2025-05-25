import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import load_config

config = load_config("config.yaml")
results_folders = config["current_results"]

# Select the folder you want to analyze
results_root = results_folders["test_subset_combined"]
# results_root = results_folders["emodb_ravdess_combined"]
# results_root = results_folders["test_subset_cremad"]
# results_root = results_folders["emodb_ravdess_cremad"]

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fusion_mode = "gate"  # "avg" or "gate"

# Set the CSV path using the selected results_root.
csv_path = os.path.join(results_root, f"_fusion_results_{fusion_mode}.csv")
df = pd.read_csv(csv_path)

y_true = df['class']
y_pred_fusion = df['fusion_pred']
y_pred_audio = df['audio_pred']
y_pred_image = df['image_pred']

fusion_acc = accuracy_score(y_true, y_pred_fusion)
audio_acc = accuracy_score(y_true, y_pred_audio)
image_acc = accuracy_score(y_true, y_pred_image)

fusion_r = classification_report(y_true, y_pred_fusion)
audio_r = classification_report(y_true, y_pred_audio)
image_r = classification_report(y_true, y_pred_image)

# Save metrics to a TXT file
metrics_txt = f"""
Fusion acc: {fusion_acc:.4f}
Audio-only acc: {audio_acc:.4f}
Image-only acc: {image_acc:.4f}

--- Fusion ---
{fusion_r}

--- Audio-only ---
{audio_r}

--- Image-only ---
{image_r}
"""
metrics_path = os.path.join(results_root, f"metrics_report_{fusion_mode}.txt")
with open(metrics_path, 'w') as f:
    f.write(metrics_txt)

print(metrics_txt)

# Plot and save confusion matrices
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
cm = confusion_matrix(y_true, y_pred_fusion, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"CM (Fusion: {fusion_mode})")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(results_root, f"cm_fusion_{fusion_mode}.png"))
plt.show()

# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.title(f"Normalized CM (Fusion: {fusion_mode})")
# plt.ylabel('True')
# plt.xlabel('Predicted')
# plt.tight_layout()
# plt.savefig(os.path.join(results_root, f"cm_fusion_normalized_{fusion_mode}.png"))
# plt.show()
