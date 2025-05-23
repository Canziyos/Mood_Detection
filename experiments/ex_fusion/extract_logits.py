import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torchaudio
import numpy as np
from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_image.image_model_interface import load_image_model, extract_image_features

audio_model, device = load_audio_model(model_path="./models/mobilenetv2_aud.pth")
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Audio logits extraction.
audio_input_set = "test"  # Change to "train", "val", or "test"
audio_root = f"./Dataset/audio/{audio_input_set}"
audio_out_dir = f"./logits/audio/{audio_input_set}"
os.makedirs(audio_out_dir, exist_ok=True)
logits_aud = {c: [] for c in class_names}

for class_name in class_names:
    folder = os.path.join(audio_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    for f in files:
        path = os.path.join(folder, f)
        waveform, sr = torchaudio.load(path)
        tensor = audio_to_tensor(waveform, sr)
        logits, _, _, _ = audio_predict(audio_model, tensor, device)
        logits_aud[class_name].append(np.squeeze(logits.cpu().numpy()))
        print(f"Extracted {class_name}: {f}, logits shape {logits.shape}")

for class_name in class_names:
    arr = np.stack(logits_aud[class_name])
    np.save(f"{audio_out_dir}/{class_name}.npy", arr)
print(f"Done saving audio logits for {audio_input_set} set.")


# Image logits extraction.
image_input_set = "test"  # Change to "train", "val", or "test".
load_image_model(model_path="./models/mobilenetv2_img.pth")
image_root = f"./dataset/images/{image_input_set}"
image_out_dir = f"./logits/images/{image_input_set}"
os.makedirs(image_out_dir, exist_ok=True)
logits_img = {c: [] for c in class_names}

for class_name in class_names:
    folder = os.path.join(image_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for f in files:
        path = os.path.join(folder, f)
        _, _, logits, _ = extract_image_features(path)
        logits = np.squeeze(logits)
        logits_img[class_name].append(logits)
        print(f"Extracted {class_name}: {f}, logits shape {logits.shape}")

for class_name in class_names:
    arr = np.stack(logits_img[class_name])
    np.save(f"{image_out_dir}/{class_name}.npy", arr)
print(f"Done saving image logits for {image_input_set} set.")

