import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from PIL import Image
from experiments.ex_fusion.mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer
from utils.utils import load_config

config = load_config("config.yaml")
model_path_float = config["models"].get("img_model", "models/mobilenetv2_emotion.pth")
model_path_quant = config["models"].get("img_quant_model", "mobilenetv2_image_quantized.pt")

# === Global cache for model instance ===
_model_float = None
_model_quant = None
def load_image_model(model_path, data_dir=None, quantized=False):
    global _model_float, _model_quant
    if quantized:
        if _model_quant is None:
            from torchvision.models.quantization.mobilenetv2 import QuantizableMobileNetV2
            torch.serialization.add_safe_globals([QuantizableMobileNetV2])
            _model_quant = torch.load(model_path, map_location="cpu", weights_only=False)
            _model_quant.eval()
        return _model_quant
    else:
        if _model_float is None:
            _model_float = MobileNetV2EmotionRecognizer(data_dir=data_dir or "dummy", model_path=model_path)
            _model_float.load_model()
        return _model_float

def extract_image_features(image_path, quantized=False):
    """
    Runs emotion recognition on a single RGB image.

    Arguments:
    - image_path: path to the image file (must be RGB-compatible)
    - quantized: bool, use quantized model

    Returns:
    - label: predicted class label (string)
    - softmax: softmax probabilities (numpy array)
    - logits: logits vector (numpy array)
    - last_hidden_layer: latent feature vector (numpy array, only for float model)
    """
    if quantized:
        model = load_image_model(model_path=model_path_quant, quantized=True)
        # Simple MobileNetV2 forward, no class wrapper or extra features
        transform = ImageTransform()
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            softmax = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1))
        return str(pred_idx), softmax, logits.cpu().numpy(), None
    else:
        model = load_image_model(model_path=model_path_float, quantized=False)
        result = model.predict(image_path)
        return result["label"], result["softmax"], result["logits"], result["last_hidden_layer"]

def ImageTransform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
