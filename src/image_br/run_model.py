import sys
import os
sys.path.append(os.path.dirname(__file__))

from mobilenetv2_emotion_recognizer import MobileNetV2EmotionRecognizer

if __name__ == "__main__":
    data_path = "../Full Dataset/Images"
    model_path = "../models/mobilenetv2_emotion.pth"
    test_image_path = ".png" 
    recognizer = MobileNetV2EmotionRecognizer(data_dir=data_path, model_path=model_path)

    # Train the model
    recognizer.train(epochs=20)

    # Predict on a sample image
    result = recognizer.predict(test_image_path)

    # Print prediction results
    print("\n================ Prediction ================")
    print("Predicted Emotion:", result["label"])
    print("Softmax Scores:", result["softmax"])
    print("Last Hidden Layer Output:", result["last_hidden_layer"])

