import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from shutil import move
from sklearn.model_selection import train_test_split
import random
import librosa

# -------- CONFIG --------
input_audio_dir = "input_audio_directory"   # Use folder structure such as: input_audio_dir/class_x/xxx.wav
output_image_dir = "output_spectrogram_directory"
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 64
WIN_LENGTH = 512                # Adjust hop, win length as needed to get desired resolution
N_MELS = 128
IMG_SIZE = (224, 224)           # Expected input size for models like MobileNetV2, ResNet, etc.
# ------------------------

# -------- TRANSFORMS --------
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window_fn=torch.hamming_window,
    n_mels=N_MELS
)
db_transform = torchaudio.transforms.AmplitudeToDB()
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------- PROCESS SINGLE AUDIO --------
def process_audio_file(file_path):
    waveform, sr = torchaudio.load(file_path)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)


    # # AUGMENTATION ---------------------------------

    # Pitch shifting (+/- 6 semitones)
    y = waveform.squeeze().numpy()  # Convert to NumPy

    # Pitch shift using librosa (faster than torchaudio)
    n_steps = np.random.randint(-6, 6)  # shift by -6 to +6 semitones
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Convert back to PyTorch tensor
    waveform_pitchShift = torch.tensor(y_shifted).unsqueeze(0)



    # Add white noise
    noise = torch.randn_like(waveform) * 0.006
    waveform_Noise = waveform + noise


    mel = mel_spectrogram(waveform)
    mel_Shift = mel_spectrogram(waveform_pitchShift)
    mel_Noise = mel_spectrogram(waveform_Noise)

    mel_db = db_transform(mel)
    mel_dbShift = db_transform(mel_Shift)
    mel_dbNoise = db_transform(mel_Noise)

    # Apply SpecAugment
    freq_mask = T.FrequencyMasking(freq_mask_param=15)
    time_mask = T.TimeMasking(time_mask_param=50)

    mel_dbaug = freq_mask(mel_db)
    mel_dbaug = time_mask(mel_dbaug)

    mel_db = torch.nn.functional.interpolate(mel_db.unsqueeze(0), size=IMG_SIZE)
    mel_dbaug = torch.nn.functional.interpolate(mel_dbaug.unsqueeze(0), size=IMG_SIZE)
    mel_dbShift = torch.nn.functional.interpolate(mel_dbShift.unsqueeze(0), size=IMG_SIZE)
    mel_dbNoise = torch.nn.functional.interpolate(mel_dbNoise.unsqueeze(0), size=IMG_SIZE)

    mel_db = mel_db.squeeze(0)
    mel_dbaug = mel_dbaug.squeeze(0)
    mel_dbShift = mel_dbShift.squeeze(0)
    mel_dbNoise = mel_dbNoise.squeeze(0)

    return mel_db, mel_dbaug, mel_dbNoise, mel_dbShift
# ---------------------------------------

# -------- MAIN PROCESS --------
if __name__ == "__main__":
    classes = os.listdir(input_audio_dir)

    for class_name in classes:
        class_audio_path = os.path.join(input_audio_dir, class_name)
        class_output_path = os.path.join(output_image_dir, class_name)

        os.makedirs(class_output_path, exist_ok=True)

        audio_files = glob(os.path.join(class_audio_path, "*.wav"))

        for audio_file in audio_files:
            mel_spec, mel_aug, mel_specNoise, mel_specShift = process_audio_file(audio_file)

            # Flip image (otherwise image files appear upside down)
            mel_spec = torch.flip(mel_spec,dims=[1])
            mel_aug = torch.flip(mel_aug,dims=[1])
            mel_specShift = torch.flip(mel_specShift,dims=[1])
            mel_specNoise = torch.flip(mel_specNoise,dims=[1])


            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            save_path = os.path.join(class_output_path, base_name + ".png")
            base_name1 = "mel_dbaug" + base_name
            save_path1 = os.path.join(class_output_path, base_name1 + ".png")
            base_name2 = "mel_dbShift" + base_name
            save_path2 = os.path.join(class_output_path, base_name2 + ".png")
            base_name3 = "mel_dbNoise" + base_name
            save_path3 = os.path.join(class_output_path, base_name3 + ".png")

            plt.imsave(save_path, mel_spec.squeeze().cpu().detach().numpy(), cmap="magma")
            plt.imsave(save_path1, mel_aug.squeeze().cpu().detach().numpy(), cmap="magma")
            plt.imsave(save_path2, mel_specShift.squeeze().cpu().detach().numpy(), cmap="magma")
            plt.imsave(save_path3, mel_specNoise.squeeze().cpu().detach().numpy(), cmap="magma")
            
        print(f"Done processing class: {class_name}")

    print("All Mel spectrogram images saved")
