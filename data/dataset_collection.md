# Blueprint: Steps to Create the Dataset

## 1. Dataset Selection

- Primary datasets:
  - RAVDESS
  - CREMA-D
  - IEMOCAP
  - AFEW
  - IIIT-H AVE
  - eNTERFACE'05
  - RML
- Criteria:
  - Must include both audio and video.
  - Must cover a mix of acted (clean) and naturalistic (noisy, spontaneous) samples.

## 2. Dataset Download and Organization

- Download datasets from official sources or university archives.
- Create a metadata CSV file for each dataset:
  - Columns: Filepath | Emotion label | Speaker ID | Gender (if available) | Ethnicity | Valence (to be added later)

## 3. Subset Selection (if needed)

- If a dataset is small (like RAVDESS or CREMA-D): we use the full dataset.
- If a dataset is very large (like IEMOCAP):  we may select a representative subset.
  - Criteria for subset:
    - Preserve balance across emotions.
    - Preserve speaker diversity (gender, ethnicity if metadata exists).
    - Preserve a range of recording conditions (studio-quality, noisy samples if possible).

## 4. Label Harmonization

- Map discrete emotions across datasets to a common set (e.g., anger, sadness, happiness, fear, neutral, surprise).
- Create a label mapping table per dataset.
- Create valence labels (positive, neutral, negative):
  - Positive: happiness, calm, excitement
  - Neutral: neutral, no emotion
  - Negative: anger, sadness, fear, disgust, frustration
- Add valence column into **the metadata CSV**.

## 5. Preprocessing and Validation

- Audio:
  - Audio clips are extracted and standardized (e.g., 16kHz mono WAV files).
- Video:
  - Available?.
  - If only videos are provided, extract frames at a consistent rate (e.g., 5 fps).
- Face Detection:
  - Apply face detector (MTCNN or RetinaFace) to crop faces from frames.
  - Save cropped face images aligned to center.
- Metadata Update:
  - Link each audio sample and video frame (or face crop) to metadata.

## 6. Diversity Analysis

- Analyze gender and ethnicity distributions across combined datasets.
- If there is a large imbalance, we can note it (for interpretation during results, or re-balancing if needed).

## 7. Final Assembly

- Merge metadata files from all datasets into a single metadata table.
- Assign split for:
  - Training set (~80 percent)
  - Validation set (~20 percent)
- Speaker independence: no same speaker in training and validation.

## 8. Dataset Versions for Experiments

- Version 1: Full discrete emotion labels (fine-grained).
- Version 2: Simplified valence-based labels (positive/neutral/negative).
- Both versions will be used for different experimental settings (RQ2)?.

## 9. Cross-corpus Test Preparation

- Reserve, at least one full dataset, (e.g., AFEW or IIIT-H AVE) as a held-out test set.
- This dataset will not be used in training at all to measure true generalization (RQ1).

## 10. Elderly Representation Strategy (Later Step)

- After initial dataset is finalized:
  - If elderly faces are still lacking, we can apply face-aging augmentation to a portion of the dataset (but only later after baselines are trained).
