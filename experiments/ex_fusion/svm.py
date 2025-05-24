import os
import numpy as np
from sklearn.model_selection import train_test_split

# Logit normalization values based on training statistics.
aud_logits_mean = -2.953
aud_logits_std  = 5.11
img_logits_mean = -0.592
img_logits_std  = 1.58

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
n_classes = len(class_names)

def load_and_normalize_logits(logits_dir, mean, std):
    all_logits = []
    all_labels = []
    for idx, cname in enumerate(class_names):
        npy_path = os.path.join(logits_dir, f"{cname}.npy")
        arr = np.load(npy_path)  # shape: [N, 6]
        arr = (arr - mean) / std
        all_logits.append(arr)
        all_labels.append(np.full(len(arr), idx))
    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    return all_logits, all_labels

audio_logits_dir = "../../logits/audio/train"
image_logits_dir = "../../logits/images/train"

audio_logits, y = load_and_normalize_logits(audio_logits_dir, aud_logits_mean, aud_logits_std)
image_logits, _ = load_and_normalize_logits(image_logits_dir, img_logits_mean, img_logits_std)

# Concatenate audio and image logits to form late-fusion features.
X = np.concatenate([audio_logits, image_logits], axis=1)
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Split into training and validation sets (stratified by class).
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Train a linear SVM as a late fusion classifier.
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate on the validation set.
y_pred = svm.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Save the SVM model for later use.
import joblib
joblib.dump(svm, "../../models/late_fusion_svm.pkl")
