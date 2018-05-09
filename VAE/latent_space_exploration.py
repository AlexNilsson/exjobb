import os
import cv2 as cv
import numpy as np

# Project Imports
from model import config as C
from model import architecture

PATH_TO_THIS_DIR = os.path.dirname(__file__)

PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

""" MODEL """
# VAE Instance
VAE = architecture.VAE()

# Encoder Model
encoder = VAE.encoder

# Decoder Model
decoder = VAE.decoder

# VAE Model
vae = VAE.model

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  print('vae summary:')
  vae.summary()
  print('encoder summary:')
  encoder.summary()
  print('decoder summary:')
  decoder.summary()


""" LOAD SAVED WEIGHTS """
try:
  before_weight_load = vae.get_weights()
  vae.load_weights(os.path.join(PATH_TO_LOAD_WEIGHTS, 'weight.hdf5'), by_name=False)
  after_weight_load = vae.get_weights()

except Exception as e:
  print('Failed to load stored weights..')
  print(str(e))

else:
  if np.array_equal(before_weight_load, after_weight_load):
    print('Loaded weights are identical?')
  else:
    print('Weights loaded successfully!')


""" SAMPLE POINTS IN LATENT SPACE """
N_SAMPLES = 1

z_samples = np.random.normal(0, 1, (N_SAMPLES, C.Z_LAYER_SIZE))

n_channels = 3 if C.COLOR_MODE == 'rgb' else 1

for z in z_samples:
  # decode each z sample using the trained decoder
  x = decoder.predict(z)
  image = x[0].reshape(C.IMG_SIZE, C.IMG_SIZE, n_channels)
  cv.imshow('Decoded Image', image)
  cv.waitKey(0)