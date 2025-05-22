import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

class SpectrogramGenerator:
    def __init__(self, sr=22050, n_mels=128):
        self.sr = sr
        self.n_mels = n_mels

    def generate(self, audio_path, output_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return output_path
