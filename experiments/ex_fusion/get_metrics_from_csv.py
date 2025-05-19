import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#df = pd.read_csv('fusion_results_avg.csv')
#df = pd.read_csv('fusion_results_prod.csv')
df = pd.read_csv('fusion_results_gate.csv')
y_true = df['class']
y_pred_fusion = df['fusion_pred']
y_pred_audio = df['audio_pred']
y_pred_image = df['image_pred']

print("Fusion Acc:", accuracy_score(y_true, y_pred_fusion))
print("Audio-only Acc:", accuracy_score(y_true, y_pred_audio))
print("Image-only Acc:", accuracy_score(y_true, y_pred_image))

print("\nFusion Report:\n", classification_report(y_true, y_pred_fusion))
print("\nAudio Report:\n", classification_report(y_true, y_pred_audio))
print("\nImage Report:\n", classification_report(y_true, y_pred_image))

print("\nConfusion Matrix (Fusion):\n", confusion_matrix(y_true, y_pred_fusion))

# Confusion Matrix Plotting.
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

# Normalize confusion matrix ----
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
