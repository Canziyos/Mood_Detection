import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import cv2, torch, numpy as np
import sounddevice as sd
from PIL import Image
from torchvision import transforms
from utils import load_config
from audio import load_audio_model, audio_to_tensor, audio_predict
from image_model_interface import load_image_model, extract_image_features
from AudioImageFusion import AudioImageFusion

config = load_config("config.yaml")
class_names = config["classes"]
audio_model_path = config["models"]["audio_model"]
image_model_path = config["models"]["image_model"]
alpha = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model, _ = load_audio_model(audio_model_path)
image_model = load_image_model(image_model_path)
fusion_head = AudioImageFusion(num_classes=len(class_names), alpha=alpha).to(device)

cap = cv2.VideoCapture(0)
SAMPLE_RATE = 16000
AUDIO_WIN_S = 0.3  # Reduced window for less latency
AUDIO_WIN_SAMPLES = int(SAMPLE_RATE * AUDIO_WIN_S)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
frame_counter = 0
frame_interval = 3

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1
    if frame_counter % frame_interval != 0:
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    h, w, _ = frame.shape
    sz = min(h, w)
    crop = frame[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2]
    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_np = np.array(img_pil)
    lab_i, probs_i, _, _ = extract_image_features(img_np)
    probs_i = torch.tensor(probs_i, dtype=torch.float32).to(device)

    audio_chunk = sd.rec(AUDIO_WIN_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio_chunk_torch = torch.from_numpy(audio_chunk.T)
    aud_t = audio_to_tensor(audio_chunk_torch, SAMPLE_RATE)
    _, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)
    probs_a = probs_a.to(device)

    fused_probs = fusion_head.fuse_probs(probs_audio=probs_a, probs_image=probs_i)
    pred_f = class_names[torch.argmax(fused_probs).item()]
    print(f"Audio: {pred_a}, Image: {lab_i}, Fusion: {pred_f}")
    print("Audio probs:", probs_a.cpu().numpy().round(3))
    print("Image probs:", probs_i.cpu().numpy().round(3))
    print("Fusion probs:", fused_probs.cpu().numpy().round(3))

    cv2.putText(crop, f"Fusion: {pred_f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
