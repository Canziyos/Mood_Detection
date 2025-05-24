import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from utils import compute_mean_std, load_config

# Load config.
config = load_config("config.yaml")

# Directory paths from config.
aud_logits_train_dir = config["logits"]["train_aud_logits_dir"]
img_logits_train_dir = config["logits"]["train_img_logits_dir"]
results_dir = config["models"]["root"]

# Class names from config.
class_names = config["classes"]
n_classes = len(class_names)

# Compute normalization statistics for training audio and image logits.
aud_logits_mean, aud_logits_std = compute_mean_std(aud_logits_train_dir)
img_logits_mean, img_logits_std = compute_mean_std(img_logits_train_dir)

def load_and_normalize_logits(logits_dir, mean, std):
    all_logits = []
    all_labels = []
    for idx, cname in enumerate(class_names):
        npy_path = os.path.join(logits_dir, f"{cname}.npy")
        arr = np.load(npy_path)
        arr = (arr - mean) / std
        all_logits.append(arr)
        all_labels.append(np.full(len(arr), idx))
    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    return all_logits, all_labels

# Load and normalize.
audio_logits, y = load_and_normalize_logits(aud_logits_train_dir, aud_logits_mean, aud_logits_std)
image_logits, _ = load_and_normalize_logits(img_logits_train_dir, img_logits_mean, img_logits_std)

# Concatenate audio and image logits to create the feature matrix.
X = np.concatenate([audio_logits, image_logits], axis=1)
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Split the data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Train a linear SVM as a late fusion classifier.
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the SVM on the validation set.
y_pred = svm.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Save the trained SVM model.
joblib.dump(svm, os.path.join(results_dir, "late_fusion_svm.pkl"))
