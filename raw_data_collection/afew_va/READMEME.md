
# AFEW-VA Dataset Preprocessing

This folder contains preprocessing utilities for working with the AFEW-VA dataset — a valence-arousal annotated video database of facial expressions in the wild.

The dataset I downloaded is already processed in the sense that each video has been converted into image frames. However, the goal of this preprocessing pipeline is to: pipeline is to:
- Remove unnecessary cropped folders.
- Clean JSON annotations.
- Extract 5 representative images per subject (actor) with valence-arousal values.

---

## Dataset Overview

Each video is represented as a folder (`001`, `002`, ..., `600`) containing:
- A sequence of PNG frames from the video (named `00000.png` -> `00NNN.png`).
- A JSON file named like `001.json` containing per-frame annotations:
  - `"valence"` and `"arousal"` scores between **-10 and 10**.
  - `"landmarks"`: facial keypoints (removed in preprocessing).

Example JSON format:
```json
{
  "actor": "Actor Name",
  "frames": {
    "00000": {
      "valence": -2,
      "arousal": 3,
      "landmarks": [...]
    },
    ...
  },
  "video_id": "001"
}
```

---

## What's Done in This Folder

### `delete_cropped_folders.py`
- Removes all subfolders ending with `_cropped`, which are often duplicated face-only versions from Kaggle.
- Handles read-only file attributes.

### `remove_landmarks.py`
- Traverses all JSON files and deletes the `"landmarks"` entry from each frame (not needed for our project).
- Keeps `"valence"` and `"arousal"`.

### `select_img.py`
- For each video folder, selects **5 evenly spaced frames**.
- Detects faces using **RetinaFace (buffalo_l)** and crops them.
- Saves each cropped image with a name like:  
  `001_00024_face0.jpg`.

### Final Output
- 5 faces crops for each actor, are saved into a flat folder `afew_selected_img_flat/`.
- Metadata with `"valence"` and `"arousal"` for each saved image is saved to:  
  `afew_va_faces.json`.

---

### Output Folder Structure

```
afew_selected_img_flat/
├── 001_00000_face0.jpg
├── 001_00045_face0.jpg
├── ...
└── afew_va_faces.json
```

Example entry in `afew_va_faces.json`:
```json
{
  "001_00045_face0.jpg": {
    "valence": 2,
    "arousal": 3
  },
  ...
}
```

---

### Tools Used

- **Face Detection**: `InsightFace` with `RetinaFace` `buffalo_l`.
- **JSON Cleaning**: `json`.
- **Image Handling**: OpenCV (`cv2`), `os`, `shutil`, and `tqdm`.

---

### Dataset Source

- [Kaggle - AFEW-VA Cropped](https://www.kaggle.com/datasets/susmitdas1053/afew-va-cropped)
