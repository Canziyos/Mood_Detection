import os
import random
import shutil

# Source folders
audio_root = "dataset/audio/emo_db_test"
image_root = "dataset/images/raf_db_test"
dst_root = "dataset/matched_images_for_audio"

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

os.makedirs(dst_root, exist_ok=True)

for cname in class_names:
    audio_dir = os.path.join(audio_root, cname)
    image_dir = os.path.join(image_root, cname)
    dst_dir = os.path.join(dst_root, cname)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(audio_dir) or not os.path.isdir(image_dir):
        print(f"Missing folder for {cname}, skipping.")
        continue

    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not audio_files or not image_files:
        print(f"No files for {cname}, skipping.")
        continue

    # Pick one image per audio, random selection.
    for i, aud_fn in enumerate(audio_files):
        img_fn = random.choice(image_files)
        src_img = os.path.join(image_dir, img_fn)
        dst_img = os.path.join(dst_dir, f"{i:04d}_{img_fn}")
        shutil.copy(src_img, dst_img)

    print(f"{cname}: {len(audio_files)} audios, {len(image_files)} images, {len(audio_files)} images copied to {dst_dir}")

print("Done. Each audio now has a randomly picked image in the destination folder.")
