
# SAVEE Dataset Preprocessing

This folder contains a short script for organizing the **SAVEE** dataset into a format that matches the six target emotion categories.

The original SAVEE dataset contains `.wav` audio files named like:

```
DC_a01.wav
JE_d04.wav
KL_h12.wav
```

The filenames use actor initials followed by emotion codes. For example:
- `a` → Angry
- `d` → Disgust
- `f` → Fear
- `h` → Happy
- `n` → Neutral
- `sa` → Sad
- `su` → Surprise *(not used in this project)*

---

## What's Included

- **Audio file sorting script**  
  Moves `.wav` files into subfolders based on emotion categories.

- **Surprise (su)** is ignored, as it's not used in our project.

---

After running the script, you'll get something like:

```
savee_audio/
  Angry/
    DC_a01.wav
    JE_a04.wav
    ...
  Disgust/
    DC_d03.wav
    ...
  Fear/
    ...
  Happy/
    ...
  Neutral/
    ...
  Sad/
    ...
```

So update the paths in the script, `distribute.py`, and run it, then the audio files will be organized by emotion.

- No renaming is done — filenames are preserved as-is.


### Dataset Source

- [Kaggle: Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
