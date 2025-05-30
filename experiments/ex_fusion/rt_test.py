import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import numpy as np
from PIL import Image
import sounddevice as sd
from torchvision import transforms
import torch
from utils.utils import load_config
from audio import load_audio_model, audio_to_tensor, audio_predict
from image_model_interface import load_image_model, extract_image_features
from AudioImageFusion import AudioImageFusion

config = load_config("config.yaml")
class_names = config["classes"]

audio_model_path = config["models"]["audio_model"]
image_model_path = config["models"]["image_model"]

print(os.path.getsize(audio_model_path) / (1024*1024), "MB")
print(os.path.getsize(image_model_path) / (1024*1024), "MB")

alpha = 0.4


#print(sd.query_devices())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model, _ = load_audio_model(audio_model_path)
image_model = load_image_model(image_model_path)
fusion_head = AudioImageFusion(num_classes=len(class_names), alpha=alpha).to(device)

cap = cv2.VideoCapture(0)
sample_rate = 16000
aud_win_s = 0.5  # seconds
aud_win_samples = int(sample_rate * aud_win_s)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

frame_skip = 2
frame_count = 0

latest_pred_f = ""
latest_pred_a = ""
latest_lab_i = ""

print("Press 'q' in the camera window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not available.")
        break

    frame_count += 1

    # Only do prediction every 'frame_skip' frames for speed
    if frame_count % frame_skip == 0:
        # Central crop and resize
        h, w, _ = frame.shape
        sz = min(h, w)
        crop = frame[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2]
        img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 224))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_np = np.array(img_pil)
        lab_i, probs_i, _, _ = extract_image_features(img_np)
        probs_i = torch.tensor(probs_i, dtype=torch.float32).to(device)

        #sd.default.device = (2, None)
        audio_chunk = sd.rec(aud_win_samples, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_chunk_torch = torch.from_numpy(audio_chunk.T)
        aud_t = audio_to_tensor(audio_chunk_torch, sample_rate)
        _, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)
        probs_a = probs_a.to(device)

        fused_probs = fusion_head.fuse_probs(probs_audio=probs_a, probs_image=probs_i)
        pred_f = class_names[torch.argmax(fused_probs).item()]

        latest_pred_f = pred_f
        latest_pred_a = pred_a
        latest_lab_i = lab_i

        print(f"\nAudio: {pred_a}, Image: {lab_i}, Fusion: {pred_f}")
        print("Audio probs:", probs_a.cpu().numpy().round(3))
        print("Image probs:", probs_i.cpu().numpy().round(3))
        print("Fusion probs:", fused_probs.cpu().numpy().round(3))

    # Always overlay the latest predictions
    cv2.putText(frame, f"Fusion: {latest_pred_f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,225,255), 2)
    cv2.putText(frame, f"Audio: {latest_pred_a}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Image: {latest_lab_i}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Webcam (press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
