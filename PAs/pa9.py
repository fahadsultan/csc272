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
  return HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/HVVpAnKzCs0?si=91LeYqPzmzmXfGUo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></center>')

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
  unknowns  = pd.Series(glob("birdsongs/unknowns/*.wav"), name='unknowns')
  
  return robins, cardinals, unknowns.iloc[0]

def encode_wav(file_path):
  y, sr = librosa.load(file_path, sr=None)
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  mfccs_mean = mfccs.mean(axis=1)
  return pd.Series(mfccs_mean)

def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


import pandas as pd

def download_mnist():
  import tensorflow as tf
  y_codes = {0: 't-shirt/top', 1: 'trouser', 2:'pullover', 3:'dress',\
             4: 'coat', 5:'sandal', 6:'shirt', 7:'sneaker', 8:'bag', \
             9: 'ankle boot', 10: ''}
  data    = tf.keras.datasets.fashion_mnist.load_data()
  X_train = [pd.DataFrame(x) for x in data[0][0]]
  X_test  = [pd.DataFrame(x) for x in data[1][0]]
  y_train = pd.Series(data[0][1]).replace(y_codes)
  y_test  = pd.Series(data[1][1]).replace(y_codes)
  return X_train, y_train, X_test, y_test

def plot_img(img, label=""):
  from matplotlib import pyplot as plt
  plt.figure(figsize=(3, 3))
  plt.imshow(img, cmap='Greys');
  plt.colorbar();
  plt.title(label);

def download_mnist():
  import tensorflow as tf
  from tqdm import tqdm
  (imgs, labels), (imgs2, _) = tf.keras.datasets.mnist.load_data()
  # imgs_df = pd.DataFrame([pd.Series(x.flatten()) for x in tqdm(imgs)])
  # mystery_img = pd.Series(imgs2[20].flatten())
  # labels = pd.Series(labels)
  mystery_img = imgs2[20]
  return imgs, labels, mystery_img


  
def plot_img(img, label=""):
  from matplotlib import pyplot as plt
  plt.figure(figsize=(3, 3))
  plt.imshow(img, cmap='Greys');
  plt.colorbar();
  plt.title(label);

