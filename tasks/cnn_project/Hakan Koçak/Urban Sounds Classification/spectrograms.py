import numpy as np
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download(force_overwrite=True)
dataset.validate()

example_clip = dataset.choice_clip()
print(example_clip)

def create_spectrogram(y):
    spec = librosa.feature.melspectrogram(y=y)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    return spec_conv

# Ana Klasör
spectrogram_klasoru = "spectrograms"
os.makedirs(spectrogram_klasoru, exist_ok=True)

# Ses dosyalarını işleme
for key, clip in dataset.load_clips().items():
    # Ses dosyasını yükle
    data, sr = librosa.load(clip.audio_path)
    spectrogram = create_spectrogram(data)

    # Spektrogramu görselleştirme
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Sınıf: {clip.class_label}")
    plt.tight_layout()

    # Spektrogramu ilgili sınıfın klasörüne kaydetme
    klasor = os.path.join(spectrogram_klasoru, str(clip.class_label))
    os.makedirs(klasor, exist_ok=True)
    dosya_adı = f"{clip.audio_path.split('/')[-1].split('.')[0]}.png"
    dosya_yolu = os.path.join(klasor, dosya_adı)
    plt.savefig(dosya_yolu)
    plt.close()

    print(f"{dosya_yolu} başarıyla kaydedildi.")
