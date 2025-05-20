import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import torch
import torchaudio
import numpy as np
from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_image.image_model_interface import load_image_model, extract_image_features


audio_model, device = load_audio_model(model_path="./models/mobilenetv2_aud.pth")

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
load_image_model(model_path="./models/mobilenetv2_img.pth", class_names=class_names)

#Set the correct output dir depending on the input set.
input_set = "val"  # switch then to "train" or "test"
audio_root = f"./dataset/audio/{input_set}"
out_dir = f"./latents/audio/{input_set}"
os.makedirs(out_dir, exist_ok=True)

latent_aud = {c: [] for c in class_names}

for class_name in class_names:
    folder = os.path.join(audio_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    for f in files:
        path = os.path.join(folder, f)
        waveform, sr = torchaudio.load(path)
        tensor = audio_to_tensor(waveform, sr)
        logits, softmax, latent, pred = audio_predict(audio_model, tensor, device)
        latent_aud[class_name].append(np.squeeze(latent.cpu().numpy()))
        print(f"Extracted {class_name}: {f}, latent shape {latent.shape}")

# Save (as npy, per class).
for class_name in class_names:
    arr = np.stack(latent_aud[class_name])
    np.save(f"{out_dir}/{class_name}.npy", arr)
print(f"Done saving aud-latents for {input_set} set.")


input_set = "val"  # switch to "train" or "val".
image_root = f"../dataset/images/{input_set}"
out_dir = f"../latents/images/{input_set}" 
os.makedirs(out_dir, exist_ok=True)
latent_img = {c: [] for c in class_names}

for class_name in class_names:
    folder = os.path.join(image_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for f in files:
        path = os.path.join(folder, f)
        _, _, _, latent = extract_image_features(path)
        # Squeeze batch dimension if present (this fucked up things before).
        latent = np.squeeze(latent)
        latent_img[class_name].append(latent)
        print(f"Extracted {class_name}: {f}, latent shape {latent.shape}")

# Save as npy, per class.
for class_name in class_names:
    arr = np.stack(latent_img[class_name])
    np.save(f"{out_dir}/{class_name}.npy", arr)
print(f"Done saving img-latents for {input_set} set.")
