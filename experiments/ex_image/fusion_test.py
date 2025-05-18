import sys, os
sys.path.append(os.path.abspath('../src/fusion'))

import numpy as np
import torch
import imageio
from ex_image.image_model_interface import load_image_model, extract_image_features
from AV_Fusion import FusionAV

# Initialize model-
model_path = r"../models/mobilenetv2_emotion.pth"
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
load_image_model(model_path=model_path, class_names=class_names)


vid_path = r"./4.mp4"
fusion_mode = "prod" # options: avg, prod, gate, mlp, latent
num_classes = 6

reader = imageio.get_reader(vid_path)
frame_idx = 0
img_emb_dim = None
fusion_model = None

for frame in reader:
    temp_path = f"temp_frame_{frame_idx}.png"
    imageio.imwrite(temp_path, frame)

    try:
        label, img_softmax, img_emb = extract_image_features(temp_path)
        img_emb = torch.tensor(img_emb, dtype=torch.float32)
        img_softmax = torch.tensor(img_softmax, dtype=torch.float32)  # .unsqueeze(0)
        img_emb = img_emb.reshape(1, -1)
    except Exception as e:
        print(f"Frame {frame_idx}: ERROR in image model ({e}), skipping.")
        os.remove(temp_path)
        frame_idx += 1
        continue

    if img_emb_dim is None:
        img_emb_dim = img_emb.shape[1]
        fusion_model = FusionAV(
            num_classes=num_classes,
            fusion_mode=fusion_mode,
            latent_dim_audio=img_emb_dim,
            latent_dim_image=img_emb_dim,
        )

    fake_audio_softmax = torch.tensor(
        np.random.dirichlet(np.ones(num_classes)), dtype=torch.float32
    ).unsqueeze(0)
    fake_audio_emb = torch.tensor(
        np.random.randn(1, img_emb_dim), dtype=torch.float32
    )

    if fusion_mode == "latent":
        fused_probs = fusion_model.fuse_probs(
            probs_audio=fake_audio_softmax, probs_image=img_softmax,
            latent_audio=fake_audio_emb, latent_image=img_emb,
        )
    else:
        fused_probs = fusion_model.fuse_probs(
            probs_audio=fake_audio_softmax, probs_image=img_softmax
        )

    fused_label = torch.argmax(fused_probs, dim=1).item()
    print(f"Frame {frame_idx}: image={label}, fused_label={fused_label}, fused_probs={fused_probs.detach().numpy()}")


    os.remove(temp_path)
    frame_idx += 1

reader.close()
print("Done.")
