import torch
from torchvision import models, transforms
import torchaudio
import numpy as np
from PIL import Image
import matplotlib.cm as cm


class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)
model_path = "mobilenetv2_TrainedOnSpectrogramDataset_71.09.pth" # Path to the model

# -------------------- LOAD MODEL AND MAKE PREDICTION ----------------------------------------------

# In this form, the function loads and prepares the model only once from disk.
# The model is loaded just once into memory (RAM or GPU), 
# not from disk every time, and can be reused for all predictions.

def load_audio_model(model_path=model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# The audio_predict function takes a preprocessed input tensor from the first part of the original code
# and runs it through the loaded model via the second function. Then, as before, it produces
#whatever it was producing.

def audio_predict(model, img_path, device):

    # --------- IMAGE TRANSFORM ---------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    # ----------------------------------

    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    # Here is the completing step, we send the tensor
    # to the same place (device) as the model.
    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        # Forward pass image through the model
        output = model(img_tensor)
        logits = output  # Output is already logits

        # Apply global average pooling manually to get the presoftMax-features
        features = model.features(img_tensor)
        pooled_features = features.mean([2, 3])  # Average over the dimensions height and width, corresponding to dimensions 2 and 3
        
        # Extract predicted class
        _, predictedIdx = torch.max(output, 1)
        predicted_class = class_names[predictedIdx.item()] # e.g., "Angry"

        # Apply softmax to get probabilities    
        softmax = torch.nn.Softmax(dim=1)
        outSoftmax = softmax(logits)
    return logits, outSoftmax, pooled_features, predicted_class

