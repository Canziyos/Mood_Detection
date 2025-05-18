import torchaudio
from audio import audio_model

# Load audio file (mono waveform, sample rate)

test_aud = r"../Dataset/Audio/val/val_ha.wav"
#test_aud = r"../Dataset/Audio/train/train_ang.wav"
#test_aud = r"../Dataset/Audio/test/test_fear.wav"

waveform, sample_rate = torchaudio.load(test_aud)

# Forward pass
logits, probs, features, predicted = audio_model(waveform, sample_rate)

print("Predicted class:", predicted)
print("Softmax probabilities:", probs)
print("Logits:", logits)
print("Pre-softmax features shape:", features.shape)
