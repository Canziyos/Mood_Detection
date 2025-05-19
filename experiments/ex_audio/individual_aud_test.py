import torchaudio
from audio import audio_to_tensor, audio_predict, load_audio_model

# Load the model and device (once).
model, device = load_audio_model() # from function 2 in audio.py

# List of test files
test_files = [
    r"test_samples\val_ha.wav",
    r"test_samples\train_ang.wav",
    r"test_samples\test_fear.wav"
]

# Loop and run inference. (we do not need to reload everything)
for i, test_aud in enumerate(test_files, 1):

    print(f"\n====Testing file {i}: {test_aud} =====")
    waveform, sample_rate = torchaudio.load(test_aud)
    
    img_tensor = audio_to_tensor(waveform, sample_rate) # (from function 1)

    logits, probs, features, predicted = audio_predict(model, img_tensor, device) # from func 3.
    print("Predicted class:", predicted)
    print("Softmax probabilities:", probs)
    print("Logits:", logits)
    print("Pre-softmax features shape:", features.shape)
