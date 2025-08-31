import torch
from torchvision import models, transforms
import torchaudio
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from utils.utils import load_config

config = load_config("config.yaml")
class_names = config["classes"]
num_classes = len(class_names)

def spectrogram_image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def audio_to_tensor(waveform, sample_rate):
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
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
    spec_np = log_mel_spec.squeeze().numpy()
    spec_norm = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-9)
    colormapped = cm.magma(spec_norm)[:, :, :3]
    colormapped = (colormapped * 255).astype(np.uint8)
    spec_img = Image.fromarray(colormapped)
    spec_img = spec_img.resize((224, 224))
    spec_img = spec_img.transpose(Image.FLIP_TOP_BOTTOM)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(spec_img).unsqueeze(0)
    return img_tensor

def load_audio_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def load_quant_aud_model(model_path):
    import torch.serialization
    from torchvision.models.quantization.mobilenetv2 import (
        QuantizableMobileNetV2,
        QuantizableInvertedResidual,
    )
    from torch.nn import Sequential, Identity
    from torchvision.ops.misc import Conv2dNormActivation
    from torch.ao.nn.intrinsic.quantized.modules.conv_relu import ConvReLU2d

    torch.serialization.add_safe_globals([
        QuantizableMobileNetV2,
        QuantizableInvertedResidual,
        Sequential,
        Identity,
        Conv2dNormActivation,
        ConvReLU2d,
    ])

    model = torch.load(model_path, map_location="cpu", weights_only=True)
    return model, torch.device("cpu")


def audio_predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        logits = output
        softmax = torch.nn.Softmax(dim=1)
        outSoftmax = softmax(logits)
        _, predictedIdx = torch.max(output, 1)
        predicted_class = class_names[predictedIdx.item()]
    return logits, outSoftmax, None, predicted_class
