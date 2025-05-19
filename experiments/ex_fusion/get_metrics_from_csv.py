# Analyze fusion, audio-only, and image-only results.
# Prints accuracy and classification reports, and plots/exports confusion matrices.
# NOTE: Uncomment the correct line for the results file you want to analyze.


# Analyze and save fusion, audio-only, and image-only results.
# NOTE: Uncomment the correct results file you want to analyze.

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Uncomment one of the following:
df = pd.read_csv('unsync_fusion_results_avg.csv')
# df = pd.read_csv('unsync_fusion_results_prod.csv')
#df = pd.read_csv('unsync_fusion_results_gate.csv')

y_true = df['class']
y_pred_fusion = df['fusion_pred']
y_pred_audio = df['audio_pred']
y_pred_image = df['image_pred']

# Compute metrics
fusion_acc = accuracy_score(y_true, y_pred_fusion)
audio_acc = accuracy_score(y_true, y_pred_audio)
image_acc = accuracy_score(y_true, y_pred_image)

fusion_r = classification_report(y_true, y_pred_fusion)
audio_r = classification_report(y_true, y_pred_audio)
image_r = classification_report(y_true, y_pred_image)

# Save to a TXT file
metrics_txt = f"""
Fusion accuracy: {fusion_acc:.4f}
Audio-only accuracy: {audio_acc:.4f}
Image-only accuracy: {image_acc:.4f}

--- Fusion classification report ---
{fusion_r}

--- Audio-only classification report ---
{audio_r}

--- Image-only classification report ---
{image_r}
"""
with open('metrics_report.txt', 'w') as f:
    f.write(metrics_txt)

print(metrics_txt)  # Also print to console.

# Plot and save confusion matrices as before
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
cm = confusion_matrix(y_true, y_pred_fusion, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Fusion)")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_fusion.png')
plt.show()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix (Fusion)")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_fusion_normalized.png')
plt.show()
