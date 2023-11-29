import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

os.getcwd()


os.chdir('genres_original')
os.getcwd()

df=pd.read_csv('features_3_sec.csv')
df.head()

genres='blues classical country rock disco reggae hiphop jazz metal pop'.split()


for i,genre in enumerate(genres):
    y,sr=librosa.load(genre+'/'+genre+'.00005.wav',duration=5.0,sr=None,)
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(y)),ref=1.0),
                         y_axis='log', x_axis='time')
    plt.title(genre.upper()+" Power Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(fname=genre+'_power_spectrogram',dpi=200)
    plt.show()
