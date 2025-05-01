
## IEMOCAP Audio Preparation

From the IEMOCAP dataset, audio extracted and grouped by six emotion categories.

Download preprocessed IEMOCAP audio segments, by [clicking Me](https://studentmdh-my.sharepoint.com/:f:/g/personal/bmd24001_student_mdu_se/EpB5r3ZSkIJKr22GcjwHaqwBOQZU8l0eo0CghWL4qzzYhA).

If you want to play around, here are some instructions.

Steps:

- All audio files from each session in the IEMOCAP dataset are originally located inside:

```
Session*/sentences/wav/<dialog_folder>/<segment_id>.wav
```

Example:

```
Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
```

All of them are moved into a unified structure:

```
IEMOCAP/wav/<dialog_folder>/<segment_id>.wav
```

- Emotion label files are originally found under:

```
Session*/dialog/EmoEvaluation/*.txt
```

These are copied into a folder called:

```
IEMOCAP/emoevaluation/
```

Each text file includes segment timing, emotion label, and valence-arousal-dominance scores etc.

- The extraction script then parses each label file and copies the matching `.wav` file into:

```
IEMOCAP/iemocap_audio/<Emotion>/<segment_id>.wav
```

Only the following emotion categories are kept (some labels are merged):

- Angry  (includes: ang, fru)
- Happy  (includes: hap, exc, sur)
- Sad
- Neutral
- Fear
- Disgust
---
- Any segment labeled with unknown or non-informative tags like `xxx` is skipped.

- `iemocap_audio/` contains all usable audio clips categorized by emotion, i.e., Happy folder, sad folder etc.
- After organizing files and folders as above, run the script `create_iemocap_aud.py` to extract and relabel the WAV files based on their discrete emotion labels.
---