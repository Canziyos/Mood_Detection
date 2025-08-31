## How to Use for Fusion

To extract features for fusion from the trained image model, use the `image_model_interface.py`.

Note: Trained model is available soon, on [\[OneDrive link\] ]() — not included in the repo.
 
### Example Usage

```python
from src.image_model_interface import extract_image_features

label, softmax, embedding = extract_image_features("path/to/image.jpg")

print("Label:", label)
print("Softmax:", softmax)
print("Embedding shape:", embedding.shape)

