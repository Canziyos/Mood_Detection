import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer

if __name__ == "__main__":
    # Paths 
    data_path = "DatasetTrainSpec" # Path to training data, with Angry, Disgust, Fear, Happy, Neutral, Sad subfolders 
    val_path = "DatasetValSpec"  # Path to validation data, with Angry, Disgust, Fear, Happy, Neutral, Sad subfolders 
    model_path = "model.pth"

    # Class names for readability in report/metrics 
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

    #  Initialize and train the model 
    recognizer = MobileNetV2EmotionRecognizer(data_dir=data_path, val_dir=val_path, model_path=model_path, device="cuda")
    recognizer.train(epochs=20)