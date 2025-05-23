# import os, sys, cv2, time, threading, queue, numpy as np, torch, sounddevice as sd
# from collections import deque

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# if ROOT not in sys.path: sys.path.insert(0, ROOT)

# from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
# from experiments.ex_image.image_model_interface import load_image_model, extract_image_features
# from src.fusion.AV_Fusion import FusionAV


# fusion_mode = "avg"
# alfa = 0.3
# cam_id = 0
# sample_rate = 16000
# aud_win_s = 1.0
# frame_rate = 10
# num_win = 10
# classes = ["Angry","Disgust","Fear","Happy","Neutral","Sad"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# aud_model, _ = load_audio_model("./models/mobilenetv2_aud.pth")
# image_model = load_image_model("./models/mobilenetv2_img.pth")
# if fusion_mode == "gate":
#     fusion_head = FusionAV(
#         num_classes=len(classes), fusion_mode="gate",
#         latent_dim_audio=None, latent_dim_image=None, use_latents=False
#     ).to(device)
#     fusion_head.load_state_dict(torch.load("./models/best_gate_head_logits.pth", map_location=device)["state_dict"])
#     fusion_head.eval()
#     print("Gate fusion head loaded.")
# elif fusion_mode == "avg":
#     fusion_head = FusionAV(
#         num_classes=len(classes), fusion_mode="avg", alpha=alfa
#     ).to(device)
#     print(f"Avg fusion head loaded (alpha={alfa}).")
# else:
#     raise ValueError("FUSION_TYPE must be 'gate' or 'avg'.")

# frame_buf = deque(maxlen=num_win)
# audio_chunk_buf = deque(maxlen=num_win)
# audio_buf = deque(maxlen=int(sample_rate * aud_win_s * 2))

# def audio_callback(indata, frames, time_info, status):
#     audio_buf.extend(indata[:, 0])

# audio_stream = sd.InputStream(callback=audio_callback, samplerate=sample_rate, channels=1, blocksize=2048)
# audio_stream.start()

# def cam_loop():
#     cap = cv2.VideoCapture(cam_id)
#     cap.set(cv2.CAP_PROP_FPS, frame_rate)
#     while True:
#         ok, frame = cap.read()
#         if not ok: continue
#         if len(audio_buf) >= sample_rate * aud_win_s:
#             center = len(audio_buf) // 2
#             half = int(sample_rate * aud_win_s / 2)
#             wav_win = np.array(audio_buf)[center-half:center+half]
#             if len(wav_win) == sample_rate * aud_win_s:
#                 frame_buf.append(frame.copy())
#                 audio_chunk_buf.append(wav_win.astype(np.float32))
#         time.sleep(1.0 / frame_rate)

# cam_thread = threading.Thread(target=cam_loop, daemon=True)
# cam_thread.start()

# def run_fusion():
#     while True:
#         if len(frame_buf) == num_win and len(audio_chunk_buf) == num_win:
#             all_logits_a = []
#             all_probs_a = []
#             all_logits_i = []
#             all_probs_i = []
#             for k in range(num_win):
#                 wav_t = torch.from_numpy(audio_chunk_buf[k]).unsqueeze(0)
#                 aud_t = audio_to_tensor(wav_t, sample_rate)
#                 logits_a, probs_a, _, _ = audio_predict(aud_model, aud_t, device)
#                 all_logits_a.append(logits_a.to(device))
#                 all_probs_a.append(probs_a.to(device))
#                 frame = frame_buf[k]
#                 h, w, _ = frame.shape
#                 sz = min(h, w)
#                 crop = cv2.resize(frame[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2], (224,224))
#                 _, probs_i, logits_i, _ = extract_image_features(crop)
#                 all_logits_i.append(torch.tensor(logits_i, dtype=torch.float32, device=device))
#                 all_probs_i.append(torch.tensor(probs_i, dtype=torch.float32, device=device))
#             mean_logits_a = torch.mean(torch.stack(all_logits_a), dim=0)
#             mean_probs_a = torch.mean(torch.stack(all_probs_a), dim=0)
#             mean_logits_i = torch.mean(torch.stack(all_logits_i), dim=0)
#             mean_probs_i = torch.mean(torch.stack(all_probs_i), dim=0)
#             if fusion_mode == "gate":
#                 fused_probs, alpha = fusion_head.fuse_probs(
#                     probs_audio=mean_probs_a,
#                     probs_image=mean_probs_i,
#                     pre_softmax_audio=mean_logits_a,
#                     pre_softmax_image=mean_logits_i,
#                     latent_audio=None,
#                     latent_image=None,
#                     return_gate=True
#                 )
#                 fused_label = classes[int(torch.argmax(fused_probs))]
#                 alpha_np = alpha.squeeze().detach().cpu().numpy()
#                 print(f"[Windowed] Fused:{fused_label} | α: {alpha_np}")
#             else:
#                 fused_probs = fusion_head.fuse_probs(
#                     probs_audio=mean_probs_a,
#                     probs_image=mean_probs_i,
#                     pre_softmax_audio=mean_logits_a,
#                     pre_softmax_image=mean_logits_i
#                 )
#                 fused_label = classes[int(torch.argmax(fused_probs))]
#                 print(f"Windowed- Fused:{fused_label} [avg α={alfa}]")
#             frame_buf.clear()
#             audio_chunk_buf.clear()
#         else:
#             time.sleep(0.05)

# fusion_thread = threading.Thread(target=run_fusion, daemon=True)
# fusion_thread.start()

# try:
#     while True: time.sleep(1)
# except KeyboardInterrupt:
#     audio_stream.stop()
#     print("Exiting.")
import os, sys, cv2, time, threading, queue, numpy as np, torch, sounddevice as sd
from collections import deque

# --- RetinaFace import ---
from retinaface import RetinaFace  

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from experiments.ex_audio.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_image.image_model_interface import load_image_model, extract_image_features
from src.fusion.AV_Fusion import FusionAV

fusion_mode = "avg"
alfa = 0.3
cam_id = 0
sample_rate = 16000
aud_win_s = 1.0
frame_rate = 10
num_win = 10
classes = ["Angry","Disgust","Fear","Happy","Neutral","Sad"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aud_model, _ = load_audio_model("./models/mobilenetv2_aud.pth")
image_model = load_image_model("./models/mobilenetv2_img.pth")
if fusion_mode == "gate":
    fusion_head = FusionAV(
        num_classes=len(classes), fusion_mode="gate",
        latent_dim_audio=None, latent_dim_image=None, use_latents=False
    ).to(device)
    fusion_head.load_state_dict(torch.load("./models/best_gate_head_logits.pth", map_location=device)["state_dict"])
    fusion_head.eval()
    print("Gate fusion head loaded.")
elif fusion_mode == "avg":
    fusion_head = FusionAV(
        num_classes=len(classes), fusion_mode="avg", alpha=alfa
    ).to(device)
    print(f"Avg fusion head loaded (alpha={alfa}).")
else:
    raise ValueError("FUSION_TYPE must be 'gate' or 'avg'.")

frame_buf = deque(maxlen=num_win)
audio_chunk_buf = deque(maxlen=num_win)
audio_buf = deque(maxlen=int(sample_rate * aud_win_s * 2))

def audio_callback(indata, frames, time_info, status):
    audio_buf.extend(indata[:, 0])

audio_stream = sd.InputStream(callback=audio_callback, samplerate=sample_rate, channels=1, blocksize=2048)
audio_stream.start()

# RetinaFace helper function.
def detect_and_crop_face(frame, target_size=(224, 224)):
    faces = RetinaFace.detect_faces(frame)
    if not faces:
        # Center crop if no face detected.
        h, w, _ = frame.shape
        sz = min(h, w)
        return cv2.resize(frame[h//2-sz//2:h//2+sz//2, w//2-sz//2:w//2+sz//2], target_size)
    # Pick the largest face (by area).
    best_key, best_face = max(faces.items(), key=lambda item: (
        (item[1]['facial_area'][2] - item[1]['facial_area'][0]) *
        (item[1]['facial_area'][3] - item[1]['facial_area'][1])
    ))
    x1, y1, x2, y2 = best_face['facial_area']
    face_crop = frame[y1:y2, x1:x2]
    return cv2.resize(face_crop, target_size)

def cam_loop():
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    while True:
        ok, frame = cap.read()
        if not ok: continue
        if len(audio_buf) >= sample_rate * aud_win_s:
            center = len(audio_buf) // 2
            half = int(sample_rate * aud_win_s / 2)
            wav_win = np.array(audio_buf)[center-half:center+half]
            if len(wav_win) == sample_rate * aud_win_s:
                # RetinaFace
                face = detect_and_crop_face(frame)  # Replace center crop.
                frame_buf.append(face)
                audio_chunk_buf.append(wav_win.astype(np.float32))
        time.sleep(1.0 / frame_rate)

cam_thread = threading.Thread(target=cam_loop, daemon=True)
cam_thread.start()

def run_fusion():
    while True:
        if len(frame_buf) == num_win and len(audio_chunk_buf) == num_win:
            all_logits_a = []
            all_probs_a = []
            all_logits_i = []
            all_probs_i = []
            for k in range(num_win):
                wav_t = torch.from_numpy(audio_chunk_buf[k]).unsqueeze(0)
                aud_t = audio_to_tensor(wav_t, sample_rate)
                logits_a, probs_a, _, _ = audio_predict(aud_model, aud_t, device)
                all_logits_a.append(logits_a.to(device))
                all_probs_a.append(probs_a.to(device))
                face = frame_buf[k]
                _, probs_i, logits_i, _ = extract_image_features(face)
                all_logits_i.append(torch.tensor(logits_i, dtype=torch.float32, device=device))
                all_probs_i.append(torch.tensor(probs_i, dtype=torch.float32, device=device))
            mean_logits_a = torch.mean(torch.stack(all_logits_a), dim=0)
            mean_probs_a = torch.mean(torch.stack(all_probs_a), dim=0)
            mean_logits_i = torch.mean(torch.stack(all_logits_i), dim=0)
            mean_probs_i = torch.mean(torch.stack(all_probs_i), dim=0)
            if fusion_mode == "gate":
                fused_probs, alpha = fusion_head.fuse_probs(
                    probs_audio=mean_probs_a,
                    probs_image=mean_probs_i,
                    pre_softmax_audio=mean_logits_a,
                    pre_softmax_image=mean_logits_i,
                    latent_audio=None,
                    latent_image=None,
                    return_gate=True
                )
                fused_label = classes[int(torch.argmax(fused_probs))]
                alpha_np = alpha.squeeze().detach().cpu().numpy()
                print(f"[Windowed] Fused:{fused_label} | α: {alpha_np}")
            else:
                fused_probs = fusion_head.fuse_probs(
                    probs_audio=mean_probs_a,
                    probs_image=mean_probs_i,
                    pre_softmax_audio=mean_logits_a,
                    pre_softmax_image=mean_logits_i
                )
                fused_label = classes[int(torch.argmax(fused_probs))]
                print(f"Windowed- Fused:{fused_label} [avg α={alfa}]")
            frame_buf.clear()
            audio_chunk_buf.clear()
        else:
            time.sleep(0.05)

fusion_thread = threading.Thread(target=run_fusion, daemon=True)
fusion_thread.start()

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    audio_stream.stop()
    print("Exiting.")
