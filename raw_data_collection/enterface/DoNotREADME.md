# eNTERFACE05 Dataset Processing

This directory contains the complete scripts and setup I used to process the eNTERFACE_DB, Audio-Visual Emotion Database.    
The dataset includes recordings of subjects expressing six basic emotions.

---

### Dataset Structure

The original videos are named like:  
`sXX_yy_Z.avi`, where:
- `sXX` = subject ID.
- `yy` = emotion code (`an`, `di`, `fe`, `ha`, `sa`, `su`).
- `Z` = repetition number.

---

## What’s going on in this folder

- **Face Extraction** (`extract_enterface_faces.py`)  
  For each video, I extract 5 representative face frames using RetinaFace (buffalo_l) and save them under emotion folders.

- **Audio Extraction** (`extract_enterface_audio.py`)  
  The audio from each video is saved as `.wav` in a folder corresponding to the emotion (also based on filename).

- **Filename Cleanup** (`rename_enterface_faces.py`)  
  After extraction, I rename all face image files to a shorter format like:  
  `subject1_s1_di_1_frame0049_face0.jpg`.

---

### Output Folder Structure

```
enterface_images/Happy/subject1_s1_ha_2_frame0049_face0.jpg
enterface_audio/Happy/subject1_s1_ha_2.wav
and so on
```

---

### Tools Used

- Face detection: InsightFace with RetinaFace (`buffalo_l`).
- Audio extraction: MoviePy.

- Renaming: Python with `os` and string parsing.

---

### Links

- Dataset page: [eNTERFACE'05 Emotion Database](https://enterface.net/enterface05/main.php?frame=emotion)
- Processed outputs (audio + faces): [OneDrive Folder](https://studentmdh-my.sharepoint.com/:f:/g/personal/bmd24001_student_mdu_se/EpB5r3ZSkIJKr22GcjwHaqwBOQZU8l0eo0CghWL4qzzYhA)