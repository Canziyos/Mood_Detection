# src/image_model_interface.py
import sys, os
sys.path.append(os.path.dirname(__file__))

from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer

# NOTE: Modified to support real inference mode!
# load_image_model now accepts class_names as an argument (optional).
# If class_names is given: skips dataset loading -just loads the model weights for inference, no folder structure needed.
# If class_names is not given: works like before (for training-expects a ata_dir, loads data, gets classes).
# This makes it possible to use this file for both pure inference and for training, depending on args.

# also: added a safety check in extract_image_features to make sure model is loaded before use.
#
# Main changes are in load_image_model(), and the interface is now more flexible-
# just pass your own class names if you want to skip dataset stuff.

_model = None

def load_image_model(model_path="models/mobilenetv2_emotion.pth", class_names=None):
    """
    Loads the image emotion recognition model ONCE with given weights and class names.
    For inference, pass class_names as a list.
    """
    global _model
    if _model is None:
        if class_names is None:
            class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
        _model = MobileNetV2EmotionRecognizer(model_path=model_path, class_names=class_names)
        _model.load_model()
        # debug.
        #print(f"Loaded emotion model with classes: {class_names}")
    return _model

 
def extract_image_features(image_path):
    """
    Given a path to an image, returns:
    - predicted label
    - softmax probabilities
    - last hidden layer embedding
    """
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded! Call load_image_model() first.")
    result = _model.predict(image_path)
    return result["label"], result["softmax"], result["last_hidden_layer"]
