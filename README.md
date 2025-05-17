## How to Use for Fusion

To extract features for fusion from the trained image model, use the `image_model_interface.py`.

Note: Trained model is available on [\[OneDrive link\] ](https://studentmdh-my.sharepoint.com/:f:/r/personal/lpc24001_student_mdu_se/Documents/MoodDetection_Image_Model?csf=1&web=1&e=X2xPBQ) — not included in the repo.
 
### Example Usage

```python
from src.image_model_interface import extract_image_features

label, softmax, embedding = extract_image_features("path/to/image.jpg")

print("Label:", label)
print("Softmax:", softmax)
print("Embedding shape:", embedding.shape)

