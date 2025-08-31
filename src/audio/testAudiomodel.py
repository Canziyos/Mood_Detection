import torchaudio
from audiomodel import audio_predict, load_audio_model
from glob import glob
import os


# Load the model and device (once).
model, device = load_audio_model()

imgFolderPath = "testSpec/newDatasetTestSpec" # Path to the folder containing Angry, Disgust, Fear, Happy, Neutral, Sad folders
test_files = glob(os.path.join(imgFolderPath, "*", "*.png"))

total = 0
correct = 0

# Loop and run inference.
for i, test_aud in enumerate(test_files, 1):

    print(f"\n====Testing file {i}: {test_aud} =====")

    logits, probs, features, predicted = audio_predict(model, test_aud, device) # Use logits, probs, or features for fusion
    true_class = os.path.basename(os.path.dirname(test_aud))
    print(f"True class: {true_class}")
    total += 1
    if predicted == true_class:
        correct += 1


# --------- FINAL STATS ---------
accuracy = 100 * correct / total if total else 0
print(f"\nTotal images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")