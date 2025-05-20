import torch
from torchvision import models, transforms
import torchaudio
import numpy as np
from PIL import Image
import matplotlib.cm as cm
# A small NOTE:
# class_names, num_classes, and model_path are not actually used in this amazing preprocessing function
# and do not impact any of its steps.
# For clarity and easier maintenance, I think, it is a good idea to define them outside the function
# and only include them where they are really needed.
# the function name could reflect what it does, imean, it is not about model, but preprocessing
# and preparing data for the model. so i suggest the renaming below.


class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)
model_path = r"../models/mobilenetv2_aud.pth" # Path to the model- (I renamed the model i got from you)

def audio_to_tensor(waveform, sample_rate):
    
    target_sample_rate = 16000
    # -------------------- RESAMPLE AND EXTRACT LOG MEL SPECTROGRAM --------------------------------------------

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Compute log-Mel spectrogram (Big overlap for better time resolution, however, at expense of computation time)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=64,
        win_length=512,
        window_fn=torch.hamming_window,
        n_mels=128
    )(waveform)

    db_transform = torchaudio.transforms.AmplitudeToDB()
    log_mel_spec = db_transform(mel_spectrogram)

    # Normalize and apply "magma" colormap (To match training image orientation)
    spec_np = log_mel_spec.squeeze().numpy()
    spec_norm = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-9)
    colormapped = cm.magma(spec_norm)[:, :, :3]  # Drop alpha channel
    colormapped = (colormapped * 255).astype(np.uint8)

    # Convert to PIL image
    spec_img = Image.fromarray(colormapped)
    spec_img = spec_img.resize((224, 224))
    spec_img = spec_img.transpose(Image.FLIP_TOP_BOTTOM)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Tensor
    img_tensor = transform(spec_img).unsqueeze(0)
    return img_tensor

# --------------------------------------------------------------------------------------------------


# -------------------- LOAD MODEL AND MAKE PREDICTION ----------------------------------------------

# In this form, the function loads and prepares the model only once from disk.
# Before, when it was part of the original version, the model, including its architecture and weights,
# was reconstructed and reloaded at every run or call.
# This way, the model is loaded just once into memory (RAM or GPU), 
# not from disk every time, and can be reused for all predictions.

def load_audio_model(model_path="../models/mobilenetv2_aud.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# The audio_predict function takes a preprocessed input tensor from the first part of the original code
# and runs it through the loaded model via the second function. Then, as before, it produces
#whatever it was producing.

def audio_predict(model, img_tensor, device):
    # Here is the completing step, we send the tensor
    # to the same place (device) as the model.
    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        # Forward pass image through the model
        output = model(img_tensor)
        logits = output  # Output is already logits

        # Apply global average pooling manually to get the presoftMax-features
        features = model.features(img_tensor)
        presoftMax = features.mean([2, 3])  # Average over the dimensions height and width, corresponding to dimensions 2 and 3
        
        # Extract predicted class
        # By the way, this (_) instaed of "maxLogit" is a style thing.
        # In Python it means: I don not care about this value. :(
        _, predictedIdx = torch.max(output, 1)
        predicted_class = class_names[predictedIdx.item()] # e.g., "Angry"

        # Apply softmax to get probabilities    
        softmax = torch.nn.Softmax(dim=1)
        outSoftmax = softmax(logits)
    return logits, outSoftmax, presoftMax, predicted_class

