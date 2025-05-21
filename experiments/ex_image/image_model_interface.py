# src/image_model_interface.py
import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from PIL import Image
from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer


# === Global cache for the model instance ===
_model = None

def load_image_model(model_path="models/mobilenetv2_emotion.pth", data_dir=None):
    """
    Loads and caches the MobileNetV2 model for inference.

    Arguments:
    - model_path: path to the .pth file containing model weights
    - data_dir: not used during inference, but required to initialize the class

    Returns:
    - loaded model instance, ready for prediction
    """
    global _model
    if _model is None:
        _model = MobileNetV2EmotionRecognizer(data_dir=data_dir or "dummy", model_path=model_path)
        _model.load_model()
    return _model

def extract_image_features(image_path):
    """
    Runs emotion recognition on a single RGB image.

    Arguments:
    - image_path: path to the image file (must be RGB-compatible)

    Returns:
    - label: predicted class label (string)
    - softmax: softmax probabilities (numpy array)
    - last_hidden_layer: latent feature vector (numpy array)
    """
    model = load_image_model()
    result = model.predict(image_path)
    return result["label"], result["softmax"],result["logits"], result["last_hidden_layer"]


