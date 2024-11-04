import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Audio
import librosa.display
import librosa
import urllib
import zipfile
from glob import glob 

def embed_video(url):
  return HTML('<center><iframe width="560" height="315" src='+url+' frameborder="0" allowfullscreen></iframe></center>')

def embed_audio(url):
  return Audio(url)

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='Greys')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()   

def download_birdsongs():
  url = "https://github.com/fahadsultan/csc272/raw/refs/heads/main/data/birdsongs.zip"

  zip_path, _ = urllib.request.urlretrieve(url)
  with zipfile.ZipFile(zip_path, "r") as f:
      f.extractall(".")

  robins    = pd.Series(glob("birdsongs/robins/*.wav"), name='robins')
  cardinals = pd.Series(glob("birdsongs/cardinals/*.wav"), name='cardinals')
  unknowns  = pd.Series(glob("birdsongs/unknown/*.wav"), name='unknowns')
  
  return robins, cardinals, unknowns

def encode_wav(file_path):
  y, sr = librosa.load(file_path, sr=None)
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  mfccs_mean = mfccs.mean(axis=1)
  return pd.Series(mfccs_mean)

def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))