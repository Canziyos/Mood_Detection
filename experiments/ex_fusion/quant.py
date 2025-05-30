import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torchvision.models.quantization import mobilenet_v2
from utils.utils import load_config
from audio import audio_to_tensor
import torchaudio
from torchvision import transforms
from PIL import Image

config = load_config("config.yaml")
class_names = config["classes"]
num_classes = len(class_names)
device = torch.device("cpu")

# === Audio model quantization ===
audio_model_path = config["models"]["audio_model"]
audio_root = config["data"]["aud_train_dir"]

audio_model = mobilenet_v2(pretrained=False)
audio_model.classifier[1] = torch.nn.Linear(audio_model.last_channel, num_classes)
audio_model.load_state_dict(torch.load(audio_model_path, map_location="cpu"))
audio_model.eval()
audio_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
audio_model.fuse_model()
torch.quantization.prepare(audio_model, inplace=True)

with torch.no_grad():
    for cname in class_names:
        class_dir = os.path.join(audio_root, cname)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist, skipping.")
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.wav', '.flac', '.mp3'))]
        for fname in files:
            print(f"Calibrating on: {os.path.join(class_dir, fname)}")
            audio_path = os.path.join(class_dir, fname)
            waveform, sample_rate = torchaudio.load(audio_path)
            input_tensor = audio_to_tensor(waveform, sample_rate)
            audio_model(input_tensor)

torch.quantization.convert(audio_model, inplace=True)

# Always save using torch.save(model, ...) not state_dict for quantized
torch.save(audio_model, "models/mobilenetv2_audio_quantized.pt")
print("Static quantization complete for audio model.")

# === Image model quantization ===
image_model_path = config["models"]["image_model"]
image_root = config["data"]["img_train_dir"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_model = mobilenet_v2(pretrained=False)
image_model.classifier[1] = torch.nn.Linear(image_model.last_channel, num_classes)
image_model.load_state_dict(torch.load(image_model_path, map_location="cpu"))
image_model.eval()
image_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
image_model.fuse_model()
torch.quantization.prepare(image_model, inplace=True)

with torch.no_grad():
    for cname in class_names:
        class_dir = os.path.join(image_root, cname)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist, skipping.")
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in files:
            print(f"Calibrating on: {os.path.join(class_dir, fname)}")
            image_path = os.path.join(class_dir, fname)
            img = Image.open(image_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0)
            image_model(input_tensor)

torch.quantization.convert(image_model, inplace=True)
torch.save(image_model, "models/mobilenetv2_image_quantized.pt")
print("Static quantization complete for image model.")
