import sys
import os

import torch
from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer

if __name__ == "__main__":
    # === Paths ===
    data_path = "../../Dataset/Images"  # Only required if training or using ImageFolder
    model_path = "../../models/mobilenetv2_emotion.pth"
    test_image_path = "../../Full Dataset/Images/Angry/01-01-05-01-01-01-18_1.png"

    # === (Optional) Class names for readability in report/metrics ===
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

    # === Initialize recognizer in inference mode (doesn't need valid data_dir) ===
    recognizer = MobileNetV2EmotionRecognizer(data_dir="dummy", model_path=model_path)
    recognizer.load_model()

    # === Uncomment this to (re)train the model if needed ===
    #recognizer = MobileNetV2EmotionRecognizer(data_dir=data_path, model_path=model_path)
    #recognizer.train(epochs=20)

    # === Run a prediction ===
    result = recognizer.predict(test_image_path)

    # === Output Results ===
    print("\n================ Prediction ================")
    print("Predicted Emotion:", result["label"])
    print("Softmax Scores:", result["softmax"])
    print("Last Hidden Layer Output Shape:", result["last_hidden_layer"].shape)

