import os

# Root folder containing emotion subfolders.
IMAGE_ROOT = r"to_what you extracted\enterface_images"

for emotion in os.listdir(IMAGE_ROOT):
    emotion_dir = os.path.join(IMAGE_ROOT, emotion)
    if not os.path.isdir(emotion_dir):
        continue

    for fname in os.listdir(emotion_dir):
        if not fname.lower().endswith(".jpg"):
            continue

        old_path = os.path.join(emotion_dir, fname)

        # Currently: subjectx_emotion_sentencex_sx_di(abbr)_x_framex_facex.jpg
        parts = fname[:-4].split("_")  # remove .jpg and split

        try:
            # Find indices of useful parts
            subject_part = parts[0]            # 'subject1'
            video_parts = parts[3:6]           # ['s1', 'di', '1']
            frame_part = parts[-2]             # 'frame0049'
            face_part = parts[-1]              # 'face0'

            # Build new name
            new_name = f"{subject_part}_{'_'.join(video_parts)}_{frame_part}_{face_part}.jpg"
            new_path = os.path.join(emotion_dir, new_name)

            os.rename(old_path, new_path)
            print("Renamed:", fname, "→", new_name)

        except Exception as e:
            print("Failed to rename:", fname, "| Error:", str(e))
