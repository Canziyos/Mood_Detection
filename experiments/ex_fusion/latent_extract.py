import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torchaudio
import numpy as np
from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_image.image_model_interface import load_image_model, extract_image_features

<<<<<<< HEAD
audio_model, device = load_audio_model(model_path="./models/mobilenetv2_aud.pth")
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# === Audio latent extraction ===
audio_input_set = "val"  # "train", "val", or "test"
audio_root = f"./dataset/audio/{audio_input_set}"
audio_out_dir = f"./latent/audio/{audio_input_set}"
os.makedirs(audio_out_dir, exist_ok=True)
latents_aud = {c: [] for c in class_names}
=======

audio_model, device = load_audio_model(model_path="../models/mobilenetv2_aud.pth")

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
load_image_model(model_path="../models/mobilenetv2_img.pth", class_names=class_names)

#Set the correct output dir depending on the input set.
input_set = "val"  # switch then to "train" or "test"
audio_root = f"../Dataset/Audio/{input_set}"
out_dir = f"../logits/Audio/{input_set}"
os.makedirs(out_dir, exist_ok=True)

logits_aud = {c: [] for c in class_names}
>>>>>>> b579a8f9e6a0bc94dc49978d0f53b0f4dad61b0d

for class_name in class_names:
    folder = os.path.join(audio_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    for f in files:
        path = os.path.join(folder, f)
        waveform, sr = torchaudio.load(path)
        tensor = audio_to_tensor(waveform, sr)
<<<<<<< HEAD
        _, _, latent, _ = audio_predict(audio_model, tensor, device)
        latents_aud[class_name].append(np.squeeze(latent.cpu().numpy()))
        print(f"Extracted {class_name}: {f}, latent shape {latent.shape}")
=======
        logits, _, _, _ = audio_predict(audio_model, tensor, device)
        logits_aud[class_name].append(np.squeeze(logits.cpu().numpy()))
        print(f"Extracted {class_name}: {f}, logits shape {logits.shape}")
>>>>>>> b579a8f9e6a0bc94dc49978d0f53b0f4dad61b0d

for class_name in class_names:
<<<<<<< HEAD
    arr = np.stack(latents_aud[class_name])
    np.save(f"{audio_out_dir}/{class_name}.npy", arr)
print(f"Done saving audio latents for {audio_input_set} set.")

# === Image latent extraction ===
image_input_set = "val"  # "train", "val", or "test"
load_image_model(model_path="./models/mobilenetv2_img.pth", class_names=class_names)
image_root = f"./dataset/images/{image_input_set}"
image_out_dir = f"./latent/images/{image_input_set}"
os.makedirs(image_out_dir, exist_ok=True)
latents_img = {c: [] for c in class_names}
=======
    if not logits_aud[class_name]:
        print(f"WARNING: No logits collected for class '{class_name}', skipping.")
        continue
    arr = np.stack(logits_aud[class_name])
    np.save(f"{out_dir}/{class_name}.npy", arr)
print(f"Done saving aud-logits for {input_set} set.")


input_set = "val"  # switch to "train" or "val".
image_root = f"../dataset/Images/{input_set}"
out_dir = f"../logits/Images/{input_set}" 
os.makedirs(out_dir, exist_ok=True)
logits_img = {c: [] for c in class_names}
>>>>>>> b579a8f9e6a0bc94dc49978d0f53b0f4dad61b0d

for class_name in class_names:
    folder = os.path.join(image_root, class_name)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for f in files:
        path = os.path.join(folder, f)
<<<<<<< HEAD
        _, _, _, latent = extract_image_features(path)
        latent = np.squeeze(latent)
        latents_img[class_name].append(latent)
        print(f"Extracted {class_name}: {f}, latent shape {latent.shape}")
=======
        _, _, _, logits = extract_image_features(path)
        # Squeeze batch dimension if present (this fucked up things before).
        logits = np.squeeze(l)
        logits_img[class_name].append(logits)
        print(f"Extracted {class_name}: {f}, logits shape {logits.shape}")
>>>>>>> b579a8f9e6a0bc94dc49978d0f53b0f4dad61b0d

for class_name in class_names:
<<<<<<< HEAD
    arr = np.stack(latents_img[class_name])
    np.save(f"{image_out_dir}/{class_name}.npy", arr)
print(f"Done saving image latents for {image_input_set} set.")
=======
    if not logits_aud[class_name]:

        print(f"WARNING: No logits collected for class '{class_name}', skipping.")
        continue
    arr = np.stack(logits_aud[class_name])
    np.save(f"{out_dir}/{class_name}.npy", arr)
print(f"Done saving aud-logits for {input_set} set.")
>>>>>>> b579a8f9e6a0bc94dc49978d0f53b0f4dad61b0d
