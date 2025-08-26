# Ryerson Emotion Dataset Processing

In this folder, you’ll find everything related to preparing the **Ryerson Multimedia Lab Emotion Database (RML)**.  
The dataset contains short audio-visual clips (around 5 seconds each) of subjects expressing six basic emotions across multiple languages and accents.  

All selected samples are validated by listeners who didn’t speak the languages, ensuring that **perceived emotion** matched the intended label. Neat :)
---

### Dataset Overview

- Total: ~500 clips  
- Duration: ~5 seconds each  
- Frame rate: 30 FPS  
- Audio: 22050 Hz, mono, 16-bit  
- Emotions: Angry (`an`), Disgust (`di`), Fear (`fe`), Happy (`ha`), Sad (`sa`), Surprise (`su`)  
- Format: `.avi` videos with embedded audio and visual

---

### What’s done in this folder

- **Face Extraction** (`extract_ryerson_faces.py`)  
  From each video, five evenly spaced frames are sampled. We detect and crop all faces using RetinaFace (buffalo_l) and store them in folders by emotion.

- **Audio Extraction** (`extract_ryerson_audio.py`)  
  The embedded audio is extracted using MoviePy and saved as `.wav` files under the appropriate emotion folder.

---

### Output Folder Structure

```
ryerson_images/Happy/folderX_ha_1_frame0012_face0.jpg
ryerson_audio/Happy/folderX_ha_1.wav
and so on
```

---

### Tools Used

- **Face Detection**: InsightFace with RetinaFace (`buffalo_l`)
- **Audio Extraction**: MoviePy
- **Parsing & Management**: Python (`os`, `cv2`, `tqdm`, `re`)

---

### Links

- Dataset: [Ryerson Emotion Database on Kaggle](https://www.kaggle.com/datasets/ryersonmultimedialab/ryerson-emotion-database)
- Sample Processed Output: [OneDrive Folder](https://studentmdh-my.sharepoint.com/:f:/g/personal/bmd24001_student_mdu_se/EpB5r3ZSkIJKr22GcjwHaqwBOQZU8l0eo0CghWL4qzzYhA)
