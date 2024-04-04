import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def grayscale_resize_normalize_for_classes(kok_dizin, hedef_boyut):
    for sinif in os.listdir(kok_dizin):
        sinif_dizini = os.path.join(kok_dizin, sinif)
        if os.path.isdir(sinif_dizini):  # Eğer bu bir dizinse
            for dosya in os.listdir(sinif_dizini):
                if dosya.endswith(".png"):
                    yol = os.path.join(sinif_dizini, dosya)
                    img = cv2.imread(yol)
                    img = img[36:544, 79:806]
                    img = cv2.resize(img, hedef_boyut)
                    plt.imshow(img, cmap='gray')

                    spectrogram_klasoru = "Processed_Spectrograms"
                    if not os.path.exists(spectrogram_klasoru):
                        os.makedirs(spectrogram_klasoru)
                    klasor = os.path.join(spectrogram_klasoru, sinif)
                    os.makedirs(klasor, exist_ok=True)
                    dosya_yolu = os.path.join(klasor, dosya)
                    cv2.imwrite(dosya_yolu, img)
                    print(f"{dosya_yolu} başarıyla kaydedildi.")


kok_dizin = "spectrograms"
hedef_boyut = (100, 100)
grayscale_resize_normalize_for_classes(kok_dizin, hedef_boyut)

#sol üst 36.6 , 75.5  sağ alt 542.5 804.5