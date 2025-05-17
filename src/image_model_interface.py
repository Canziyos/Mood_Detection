# src/image_model_interface.py

import torch
from PIL import Image
from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer

# Load once globally
_model = None

def load_image_model(model_path="models/mobilenetv2_emotion.pth", data_dir=None):
    global _model
    if _model is None:
        class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
        _model = MobileNetV2EmotionRecognizer(data_dir=data_dir or "dummy", model_path=model_path)
        _model.load_model()
    return _model

def extract_image_features(image_path):
    """
    Given a path to an image, returns:
    - predicted label
    - softmax probabilities
    - last hidden layer embedding
    """
    model = load_image_model()
    result = model.predict(image_path)
    return result["label"], result["softmax"], result["last_hidden_layer"]
