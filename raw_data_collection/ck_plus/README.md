
# CK+ Emotion Dataset (Kaggle Version)

This folder contains the CK+ (Cohn-Kanade) facial expression dataset downloaded from the following Kaggle link:

[Kaggle Dataset – CK+](https://www.kaggle.com/datasets/shuvoalok/ck-dataset)

the modified zipped folder is availnble in project_datasets folder on [OneDrive](https://studentmdh-my.sharepoint.com/:f:/g/personal/bmd24001_student_mdu_se/EpB5r3ZSkIJKr22GcjwHaqwBOQZU8l0eo0CghWL4qzzYhA)

---

## Dataset Description

This version of the dataset includes facial expression images labeled with seven basic emotions:
- **Anger**
- **Contempt**
- **Disgust**
- **Fear**
- **Happiness**
- **Sadness**
- **Surprise**

> Note: There is no "Neutral" folder in this version of the dataset, although it's often included in other CK+ variants. I haven’t spent time digging deeper into this, since we’ll eventually shuffle everything anyway. Besides, we already have plenty of neutral samples from other datasets.

---

## Alternative CSV Version

Alongside the image folders, this dataset is presented as a **CSV version** in which each row corresponds to a 48x48 grayscale face image stored as 2304 pixel values.

- **pixels**: 2304 values representing the flattened 48x48 image.
- **emotion**: Corresponding label.
- **Usage**:
  - `Training` → 80% of the data.
  - `PublicTest` → 10%.
  - `PrivateTest` → 10%.

---

I’ll test converting the pixel values back into images to extract neutral samples if needed.


